"""The implied-vol surface: market against model, and how far to trust the fit."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for _shared

import _shared  # noqa: E402
from torch_pricer.market.surface.calibration import fit_surface, option_quotes

st.set_page_config(page_title="Vol surface", layout="wide")
st.title("Implied volatility surface")
run = _shared.require_run()
_shared.run_caption(run)

res = run.result
quotes = res.residuals  # one row per fitted quote: k, T, market iv, model_iv, error, volume, weight
comparison = st.session_state.get(_shared.COMPARE_KEY)
pct = 100


def model_vols(surface, k, t) -> np.ndarray:
    """The library's implied vol on a grid, for drawing."""
    with torch.no_grad():
        return surface.implied_vol(torch.as_tensor(k, dtype=torch.float64), float(t)).numpy()


# -- smiles ------------------------------------------------------------------------
st.header("Smiles")
st.markdown(
    "Market points are one quote per contract: the volume-weighted average of per-minute implied "
    "vols, out-of-the-money options only (puts left of the forward, calls right). Point size is "
    "volume. The line is the fitted surface at that expiry."
)
expiries = quotes.groupby("expiry")["T"].first()
liquid = quotes.groupby("expiry").size().sort_values(ascending=False).index[:6]
chosen = st.multiselect("Expiries", list(expiries.index), default=sorted(liquid),
                        format_func=lambda e: f"{e} ({expiries[e] * 365:.0f}d)")
columns = 3
for start in range(0, len(chosen), columns):
    for col, expiry in zip(st.columns(columns), chosen[start:start + columns]):
        part = quotes[quotes["expiry"] == expiry]
        t = float(expiries[expiry])
        grid = np.linspace(part["k"].min() - 0.02, part["k"].max() + 0.02, 200)
        fig = go.Figure()
        for right, color in (("put", "#d62728"), ("call", "#1f77b4")):
            side = part[part["right"] == right]
            fig.add_scatter(x=side["k"], y=pct * side["iv"], mode="markers", name=right,
                            marker=dict(color=color, size=4 + 10 * np.sqrt(side["volume"] / part["volume"].max())),
                            customdata=side[["strike", "volume", "error"]],
                            hovertemplate="K=%{customdata[0]}<br>k=%{x:.3f}<br>iv=%{y:.2f}%<br>"
                                          "volume %{customdata[1]:.0f}<br>model error "
                                          "%{customdata[2]:.4f}<extra></extra>")
        fig.add_scatter(x=grid, y=pct * model_vols(run.snapshot.vol_surface, grid, t), name=run.settings.surface_model,
                        line=dict(color="black"))
        if comparison:
            fig.add_scatter(x=grid, y=pct * model_vols(comparison[1], grid, t), name=comparison[0],
                            line=dict(color="green", dash="dash"))
        fig.update_layout(title=f"{expiry} · {t * 365:.0f}d · {len(part)} quotes", height=330,
                          xaxis_title="k = ln(K/F)", yaxis_title="implied vol %", showlegend=False,
                          margin=dict(t=40, b=40))
        col.plotly_chart(fig, width="stretch")

# -- term structure and surface ----------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("ATM term structure")
    t_grid = np.geomspace(expiries.min(), expiries.max(), 200)
    with torch.no_grad():
        atm = run.snapshot.vol_surface.implied_vol(torch.zeros(len(t_grid), dtype=torch.float64),
                                      torch.as_tensor(t_grid)).numpy()
    market_atm = quotes.loc[quotes.groupby("expiry")["k"].apply(lambda k: k.abs().idxmin())]
    fig = go.Figure()
    fig.add_scatter(x=t_grid * 365, y=pct * atm, name="surface at k = 0")
    fig.add_scatter(x=market_atm["T"] * 365, y=pct * market_atm["iv"], mode="markers",
                    name="quote nearest the money", customdata=market_atm[["expiry", "k"]],
                    hovertemplate="%{customdata[0]}<br>k=%{customdata[1]:.4f}<br>%{y:.2f}%<extra></extra>")
    fig.update_layout(xaxis_type="log", xaxis_title="days to expiry", yaxis_title="implied vol %",
                      height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Surface")
    k_grid = np.linspace(quotes["k"].quantile(0.02), quotes["k"].quantile(0.98), 60)
    t_axis = np.geomspace(expiries.min(), expiries.max(), 40)
    z = np.stack([model_vols(run.snapshot.vol_surface, k_grid, t) for t in t_axis])
    fig = go.Figure(go.Surface(x=k_grid, y=t_axis * 365, z=pct * z, colorscale="Viridis", opacity=0.85,
                               showscale=False))
    fig.add_scatter3d(x=quotes["k"], y=quotes["T"] * 365, z=pct * quotes["iv"], mode="markers",
                      marker=dict(size=2, color="red"), name="market")
    fig.update_layout(scene=dict(xaxis_title="k", yaxis_title="days", zaxis_title="iv %",
                                 yaxis_type="log"), height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")

# -- fit quality -------------------------------------------------------------------
st.header("Fit quality")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Weighted RMSE", f"{pct * res.rmse:.2f} vol pts")
c2.metric("Quotes beyond 3 vol pts", int((quotes["error"].abs() > 0.03).sum()))
c3.metric("Min butterfly density g(k)", f"{res.diagnostics['min_butterfly_density']:.3g}",
          help="Durrleman's condition; negative would be a butterfly arbitrage.")
c4.metric("Min calendar increment", f"{res.diagnostics['min_calendar_increment']:.2e}",
          help="Smallest step of total variance between maturities at fixed k; negative is a calendar arbitrage.")

left, right = st.columns(2)
with left:
    fig = px.scatter(quotes, x="k", y=quotes["T"] * 365, color=pct * quotes["error"],
                     color_continuous_scale="RdBu_r", range_color=(-3, 3), size=np.sqrt(quotes["volume"]),
                     labels={"y": "days to expiry", "color": "model − market (vol pts)"},
                     hover_data=["expiry", "strike", "right", "iv", "model_iv", "volume"],
                     title="Residuals over (k, T)", height=420, log_y=True)
    st.plotly_chart(fig, width="stretch")
with right:
    per_expiry = res.diagnostics["per_expiry"].reset_index()
    fig = px.bar(per_expiry, x="expiry", y=pct * per_expiry["rmse"], hover_data=["quotes", "max_abs"],
                 labels={"y": "RMSE (vol pts)"}, title="Error by expiry", height=420)
    st.plotly_chart(fig, width="stretch")

# -- the model ---------------------------------------------------------------------
st.header("The fitted model")
left, right = st.columns(2)
with left:
    st.subheader("Parameters")
    described = {k: v for k, v in run.snapshot.vol_surface.describe().items() if not isinstance(v, list)}
    st.dataframe(pd.DataFrame({"value": described}), width="stretch")
    st.markdown("**No-arbitrage bounds** — enforced by the parametrisation, so a value can approach "
                "its bound but not cross it. One pressed against its bound means the data wants more.")
    bounds = pd.DataFrame(run.snapshot.vol_surface.constraints()).T
    st.dataframe(bounds, width="stretch")
    with st.expander("Raw parameters (what the optimizer moves)"):
        st.dataframe(pd.DataFrame({name: [str(np.round(p.detach().numpy(), 6).tolist())]
                                   for name, p in run.snapshot.vol_surface.named_parameters()}).T, width="stretch")
with right:
    st.subheader("Convergence")
    history = res.diagnostics["loss_history"]
    probe = res.diagnostics["probe_start"]
    fig = go.Figure()
    fig.add_scatter(y=history[:probe], name="L-BFGS", mode="lines")
    fig.add_scatter(x=list(range(probe, len(history))), y=history[probe:], name="convergence probe",
                    mode="lines")
    fig.update_layout(yaxis_type="log", xaxis_title="loss evaluation", yaxis_title="weighted loss",
                      height=360, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{res.iterations} iterations; converged = {res.converged}: a further 50-iteration probe "
        "improved the loss by at most 1e-4 relative (execution log P4-3)."
    )

with st.expander("How a quote's vol was computed"):
    st.markdown("Pick a contract to see the minute bars behind its quote.")
    contracts = run.quote_bars.groupby(["expiry", "strike", "right"]).size().reset_index(name="bars")
    label = contracts.apply(lambda r: f"{r.expiry} {r.strike:g} {'C' if r.right > 0 else 'P'} "
                                      f"({r.bars} bars)", axis=1)
    pick = st.selectbox("Contract", contracts.index, format_func=lambda i: label[i])
    c = contracts.loc[pick]
    bars = run.quote_bars[(run.quote_bars["expiry"] == c.expiry) & (run.quote_bars["strike"] == c.strike)
                          & (run.quote_bars["right"] == c.right)]
    st.dataframe(bars[["timestamp", "price", "volume", "spot", "forward", "discount", "T", "k", "iv"]],
                 hide_index=True, width="stretch")
    st.caption("Each bar's vol inverts Black at that minute's forward; the quote is their "
               "volume-weighted average.")
    _shared.show_source(option_quotes, label="Quotes")

_shared.show_source(fit_surface, label="The fit")
_shared.show_source(type(run.snapshot.vol_surface), label="The model")
