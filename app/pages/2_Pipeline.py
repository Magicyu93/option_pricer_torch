"""How the current calibration was computed: every step, its code, and the data it read."""

import pandas as pd
import streamlit as st

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for _shared

import _shared  # noqa: E402
from data_pipeline.coverage import coverage

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("How it was computed")
run = _shared.require_run()
_shared.run_caption(run)
ticker = run.product.underlying

st.markdown(
    "The calibration is one call, `torch_pricer.market.build.build_snapshot`, which runs the steps "
    "below in order and records each one. Every panel shows the library function that ran — its own "
    "documentation and source — with the numbers from this run."
)
_shared.show_source(_shared.resolve("torch_pricer.market.build.build_snapshot"),
                    label="The whole chain")

# -- the steps ---------------------------------------------------------------------
for number, step in enumerate(run.trace, 1):
    with st.expander(f"**{number}. {step.name}** — `{step.function}` · {1000 * step.seconds:.1f} ms",
                     expanded=number == 1):
        fn = _shared.resolve(step.function)
        left, right = st.columns([3, 2])
        with left:
            _shared.show_formulas(step.name)
            _shared.show_doc(fn)
        with right:
            for title, values in (("Settings", step.settings), ("Inputs", step.inputs),
                                  ("Outputs", step.outputs)):
                if values:
                    st.markdown(f"**{title}**")
                    st.dataframe(pd.DataFrame({"": [str(v) for v in values.values()]},
                                              index=list(values)), width="stretch")
            if step.dropped:
                st.markdown("**Dropped, by reason**")
                st.dataframe(pd.DataFrame({"count": step.dropped}), width="stretch")
            st.caption(f"Result kept as `run.{step.artifact}`.")
        _shared.show_source(fn)

# -- data lineage ------------------------------------------------------------------
st.header("Data read")
s = run.settings
files = coverage()
year = str(s.as_of.year)
used = files[
    ((files["date"] == s.as_of.isoformat()) & (files["ticker"].isin([ticker, None]) | files["ticker"].isna())
     & files["dataset"].isin(["options", "indices", "stocks"]))
    | ((files["dataset"] == "rates") & (files["year"] == year))
]
st.dataframe(used[["layer", "dataset", "date", "year", "ticker", "series", "provider", "bytes", "path"]],
             hide_index=True, width="stretch")
published = run.step("Discount curve").inputs["published"]
st.caption(
    f"Raw files are the provider's downloads; parquet files are the filtered caches the loaders "
    f"read. Treasury yields used: published {published} (snapshot date {s.as_of})."
)

# -- reproduce ---------------------------------------------------------------------
st.header("Reproduce in Python")
cfg = s.quote_config
sources = st.session_state.get(_shared.SOURCES_KEY, {"provider": "massive", "lag_days": 1})
st.code(
    f'''import datetime as dt
from data_pipeline import MassiveHistoricalDataLoader, RatesDataLoader
from torch_pricer.instruments.listed import listed_product
from torch_pricer.market.build import SnapshotSettings, build_snapshot
from torch_pricer.market.curve.calibration import par_yield_quotes
from torch_pricer.market.market_data import QuoteConfig

day = dt.date({s.as_of.year}, {s.as_of.month}, {s.as_of.day})
frame = MassiveHistoricalDataLoader().load_market([day.isoformat()], ["{ticker}"])
yields = RatesDataLoader("{sources['provider']}").yields_as_of(day, lag_days={sources['lag_days']})
rates = par_yield_quotes(yields.tenors, yields.yields_pct, as_of=yields.observation_date)

settings = SnapshotSettings(
    as_of=day,
    quote_config=QuoteConfig(window_start="{cfg.window_start}", window_end="{cfg.window_end}",
                             min_days={cfg.min_days}, min_price={cfg.min_price}, max_abs_k={cfg.max_abs_k}),
    curve_model="{s.curve_model}", surface_model="{s.surface_model}",
    max_pairs={s.max_pairs}, min_pairs={s.min_pairs}, max_iter={s.max_iter}, day_count="{s.day_count}",
)
run = build_snapshot(frame, rates, listed_product("{ticker}"), settings)
run.snapshot.vol(strike, t)        # implied vol from the calibrated snapshot
run.trace                          # every step, as on this page
run.inputs                         # CalibrationInputs(snapshot, observations), for a model's calibrate()
''',
    language="python",
)
