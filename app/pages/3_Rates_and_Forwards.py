"""Treasury curve against what the option forwards imply."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for _shared

import _shared  # noqa: E402
from torch_pricer.conventions import tenor_years
from torch_pricer.market.curve.calibration import rates_comparison

st.set_page_config(page_title="Rates & forwards", layout="wide")
st.title("Rates & forwards")
run = _shared.require_run()
_shared.run_caption(run)

curve, carry = run.snapshot.discount, run.snapshot.dividend
table = rates_comparison(run.forwards, curve, carry)
pct = 100

# -- the Treasury curve ------------------------------------------------------------
st.header("Treasury curve")
st.markdown(
    "Par yields as published, bootstrapped into continuously-compounded zero rates "
    "(bills simple-interest up to 6M, semiannual par bonds beyond), and the instantaneous "
    "forward rate those zeros imply."
)
par = pd.DataFrame([{"tenor": q.tenor, "T": tenor_years(q.tenor), "par yield": q.rate,
                     "published": q.as_of} for q in run.observations.rates])
horizon = max(float(par["T"].max()), float(run.forwards["T"].max()))
grid = np.linspace(1 / 365, horizon, 400)
fig = go.Figure()
fig.add_scatter(x=par["T"], y=pct * par["par yield"], mode="markers+text", name="par yields",
                text=par["tenor"], textposition="top center", marker=dict(size=9))
fig.add_scatter(x=grid, y=pct * curve.zero_rate(grid).detach().numpy(), name="zero rate")
fig.add_scatter(x=grid, y=pct * curve.instantaneous_forward(grid).detach().numpy(),
                name="instantaneous forward", line=dict(dash="dot"))
fig.update_layout(xaxis_title="T (years)", yaxis_title="% (continuous)", height=420,
                  xaxis_type="log", legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")
st.dataframe(par, hide_index=True)

# -- what the options imply --------------------------------------------------------
st.header("What option forwards imply")
st.markdown(
    "Put-call parity gives each expiry's forward `F` with no dividend forecast. "
    "`ln(F/S)/T` is the growth rate `r − q` the market prices in; with the Treasury `r`, "
    "the remainder is the implied carry `q`. Hollow markers are expiries whose forward was "
    "interpolated (too few same-minute call/put pairs)."
)
parity = table["source"] == "parity"
fig = go.Figure()
fig.add_scatter(x=table["T"], y=pct * table["treasury_zero"], name="Treasury r(T)", mode="lines")
for label, column in (("implied r − q", "implied_growth"), ("implied carry q(T)", "carry_zero")):
    for mask, symbol, suffix in ((parity, "circle", ""), (~parity, "circle-open", " (interpolated)")):
        fig.add_scatter(x=table.loc[mask, "T"], y=pct * table.loc[mask, column], mode="markers",
                        name=label + suffix, marker=dict(symbol=symbol, size=8),
                        legendgroup=label, customdata=table.loc[mask, ["expiry", "n_pairs"]],
                        hovertemplate="%{customdata[0]}<br>T=%{x:.4f}<br>%{y:.3f}%<br>"
                                      "pairs: %{customdata[1]}<extra></extra>")
fig.update_layout(xaxis_title="T (years)", yaxis_title="% (continuous)", height=450,
                  xaxis_type="log", legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")
st.info(
    "The implied carry sits well below SPX's ~1.2% dividend yield at most maturities. With trade "
    "prices only, parity pins down `r − q` together, not `r` alone, so the gap between the "
    "options' funding rate and Treasuries lands in `q` (execution log P2-5). Short expiries are "
    "noisy: a 1e-4 error in F/S at T = 0.02 is 0.5% of carry (P2-6)."
)

st.subheader("Forward rates between expiries")
fig = go.Figure()
fig.add_scatter(x=table["T"], y=pct * table["treasury_forward"], name="Treasury", line_shape="vh")
fig.add_scatter(x=table["T"], y=pct * table["carry_forward"], name="implied carry", line_shape="vh")
fig.update_layout(xaxis_title="T (years)", yaxis_title="% (continuous)", height=380,
                  xaxis_type="log", legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each step is the curve's forward rate from the previous expiry to this one. The carry "
    "curve's swings between neighbouring short expiries are what the Monte Carlo engine's "
    "step-start drift samples (execution log V-1, fix deferred)."
)

st.subheader("Forward table")
shown = run.forwards.merge(table[["expiry", "treasury_zero", "implied_growth", "carry_zero"]], on="expiry")
st.dataframe(shown, hide_index=True, width="stretch")
_shared.show_source(rates_comparison)
