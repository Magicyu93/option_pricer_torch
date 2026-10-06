"""Choose the data and the models, and run a calibration."""

import dataclasses

import pandas as pd
import streamlit as st

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for _shared

import _shared  # noqa: E402
from data_pipeline.coverage import cached_days
from torch_pricer.market.curve.calibration import CURVE_MODELS
from torch_pricer.instruments.listed import LISTED_PRODUCTS
from torch_pricer.market.market_data import QuoteConfig
from torch_pricer.market.surface.calibration import SURFACE_MODELS, fit_surface

st.set_page_config(page_title="Calibrate", layout="wide")
st.title("Calibrate")

# -- what to calibrate -------------------------------------------------------------
european = [t for t, p in LISTED_PRODUCTS.items() if p.is_european]
ticker = st.selectbox("Underlying", list(LISTED_PRODUCTS),
                      index=list(LISTED_PRODUCTS).index(european[0]))
product = LISTED_PRODUCTS[ticker]
st.caption(f"{product.description} — {product.describe()}")
if not product.is_european:
    st.warning(
        f"{ticker} options have {product.style.value} exercise: put-call parity is only an "
        "inequality for them, so forwards need a dividend forecast, which the calibration does "
        "not implement yet."
    )
    st.stop()

days = cached_days("options", ticker)
if not days:
    st.warning(f"No {ticker} options cached. Download a day on the **Home** page.")
    st.stop()

default = QuoteConfig()
with st.form("calibrate"):
    st.subheader("Data")
    c1, c2, c3 = st.columns(3)
    day = c1.selectbox("Day", days, index=len(days) - 1)
    window_start = c2.text_input("Window start (ET)", default.window_start)
    window_end = c3.text_input("Window end (ET, exclusive)", default.window_end)
    c1, c2, c3 = st.columns(3)
    min_days = c1.number_input("Minimum days to expiry", 1, 365, default.min_days)
    min_price = c2.number_input("Minimum trade price", 0.0, 10.0, default.min_price, step=0.05)
    max_abs_k = c3.number_input("Moneyness band: |ln K/F| ≤ band · √T", 0.1, 3.0, default.max_abs_k, step=0.1)

    st.subheader("Rates and forwards")
    c1, c2, c3, c4 = st.columns(4)
    provider = c1.radio("Treasury yields from", ["massive", "fred"], horizontal=True)
    lag_days = c2.number_input("Publication lag (days)", 0, 5, 1,
                               help="Yields for a day are published after the close: lag 1 avoids look-ahead.")
    max_pairs = c3.number_input("Parity: pairs per expiry (nearest the money)", 1, 50, 10)
    min_pairs = c4.number_input("Parity: minimum pairs for a forward", 1, 20, 2)

    st.subheader("Models")
    c1, c2, c3 = st.columns(3)
    curve_model = c1.selectbox("Discount curve model", list(CURVE_MODELS))
    surface_model = c2.selectbox("Vol surface model", list(SURFACE_MODELS))
    max_iter = c3.number_input("Fit: maximum L-BFGS iterations", 50, 5000, 500, step=50)
    run_it = st.form_submit_button("Calibrate", type="primary")

if run_it:
    config = dataclasses.replace(
        default, window_start=window_start, window_end=window_end, min_days=int(min_days),
        min_price=float(min_price), max_abs_k=float(max_abs_k),
    )
    try:
        run = _shared.calibrate(ticker, day, provider, int(lag_days), config, curve_model,
                                surface_model, int(max_pairs), int(min_pairs), int(max_iter))
    except Exception as exc:  # shown, not swallowed: the message is the library's
        st.error(f"Calibration failed: {type(exc).__name__}: {exc}")
        st.stop()
    st.session_state[_shared.RUN_KEY] = run
    st.session_state[_shared.SOURCES_KEY] = {"provider": provider, "lag_days": int(lag_days)}
    st.session_state.pop(_shared.COMPARE_KEY, None)

run = _shared.current_run()
if run is None:
    st.stop()

# -- the result --------------------------------------------------------------------
st.divider()
st.header("Result")
_shared.run_caption(run)
res = run.result
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Quotes fitted", f"{len(run.observations.options):,}")
c2.metric("Expiries", len(run.forwards), f"{(run.forwards['source'] == 'parity').sum()} from parity",
          delta_color="off")
c3.metric("Fit RMSE", f"{100 * res.rmse:.2f} vol pts")
c4.metric("Converged", "yes" if res.converged else "no", f"{res.iterations} iterations", delta_color="off")
c5.metric("Calibration time", f"{sum(s.seconds for s in run.trace):.1f} s")

st.subheader("Steps")
st.dataframe(
    pd.DataFrame([
        {"step": s.name, "outputs": ", ".join(f"{k}: {v}" for k, v in s.outputs.items()),
         "dropped": sum(s.dropped.values()), "ms": round(1000 * s.seconds, 1)}
        for s in run.trace
    ]),
    hide_index=True, width="stretch",
)
st.caption("Every step in detail — docstring, formula, settings, drops, source — on the **Pipeline** page.")

# -- a second model on the same quotes ---------------------------------------------
st.subheader("Compare another surface model")
others = [m for m in SURFACE_MODELS if m != run.settings.surface_model]
if others:
    c1, c2 = st.columns([1, 3])
    other = c1.selectbox("Model", others)
    if c1.button("Fit on the same quotes"):
        surface, result = fit_surface(SURFACE_MODELS[other], run.observations.options, run.forwards,
                                      run.settings.as_of, max_iter=run.settings.max_iter)
        st.session_state[_shared.COMPARE_KEY] = (other, surface, result)
    comparison = st.session_state.get(_shared.COMPARE_KEY)
    if comparison:
        name, _, result = comparison
        c2.dataframe(
            pd.DataFrame([
                {"model": run.settings.surface_model, "RMSE (vol pts)": 100 * res.rmse,
                 "converged": res.converged, "iterations": res.iterations},
                {"model": name, "RMSE (vol pts)": 100 * result.rmse,
                 "converged": result.converged, "iterations": result.iterations},
            ]),
            hide_index=True,
        )
        c2.caption("Overlaid on the **Vol surface** page.")
