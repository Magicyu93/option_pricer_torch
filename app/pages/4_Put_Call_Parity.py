"""Put-call parity, expiry by expiry: the pairs, the line they sit on, the forward they give."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for _shared

import _shared  # noqa: E402
from torch_pricer.market.curve.calibration import parity_forwards, parity_pairs, parity_residuals

st.set_page_config(page_title="Put-call parity", layout="wide")
st.title("Put-call parity")
run = _shared.require_run()
_shared.run_caption(run)

st.latex(r"C - P = D(T)\,(F - K) \quad\Rightarrow\quad F = K + \frac{C - P}{D(T)}")
st.markdown(
    "A call and a put on the same strike and expiry that traded **in the same minute** form a "
    "pair. Each pair gives a forward, expressed as a ratio to that minute's spot so pairs from "
    "different minutes are comparable. The expiry's forward is the median ratio over the pairs "
    "nearest the money, times the snapshot's spot."
)

residuals = parity_residuals(run.pairs, run.forwards)
summary = run.forwards.assign(
    pairs_found=run.forwards["expiry"].map(run.pairs.groupby("expiry").size()).fillna(0).astype(int)
)
c1, c2, c3 = st.columns(3)
c1.metric("Expiries", len(summary))
c2.metric("With a parity forward", int((summary["source"] == "parity").sum()))
c3.metric("Same-minute pairs", f"{len(run.pairs):,}", f"{int(run.pairs['used'].sum())} used", delta_color="off")

st.subheader("Pairs per expiry")
fig = px.bar(summary, x="expiry", y="pairs_found", color="source",
             color_discrete_map={"parity": "#1f77b4", "interpolated": "#bbbbbb"},
             hover_data=["T", "n_pairs", "forward"])
fig.add_hline(y=run.settings.min_pairs, line_dash="dot", annotation_text="minimum for a forward")
fig.update_layout(height=320, xaxis_title=None, yaxis_title="same-minute pairs")
st.plotly_chart(fig, width="stretch")

# -- one expiry --------------------------------------------------------------------
st.subheader("One expiry in detail")
with_pairs = summary[summary["pairs_found"] > 0]
expiry = st.selectbox("Expiry", with_pairs["expiry"],
                      format_func=lambda e: f"{e} ({int(with_pairs.set_index('expiry').loc[e, 'pairs_found'])} pairs)")
row = run.forwards.set_index("expiry").loc[expiry]
one = residuals[residuals["expiry"] == expiry].sort_values("strike")
one = one.assign(minute=one["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M"))

left, right = st.columns(2)
with left:
    fig = go.Figure()
    for used, name, symbol in ((True, "used (nearest the money)", "circle"), (False, "not used", "x")):
        part = one[one["used"] == used]
        fig.add_scatter(x=part["strike"], y=part["c_minus_p"], mode="markers", name=name,
                        marker=dict(symbol=symbol, size=9), customdata=part[["minute", "price_c", "price_p"]],
                        hovertemplate="K=%{x}<br>C−P=%{y:.2f}<br>%{customdata[0]}: "
                                      "C=%{customdata[1]}, P=%{customdata[2]}<extra></extra>")
    fig.add_scatter(x=one["strike"], y=one["fitted"], mode="lines", name="D·(F − K)")
    fig.update_layout(title="C − P against strike", xaxis_title="strike", yaxis_title="C − P",
                      height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.scatter(one, x="strike", y="ratio", color="minute", symbol="used",
                     title="Each pair's forward / spot", height=400)
    fig.add_hline(y=row["ratio"], line_dash="dot",
                  annotation_text=f"expiry ratio {row['ratio']:.6f} ({row['source']})")
    st.plotly_chart(fig, width="stretch")

st.markdown(f"**Worked example — {expiry}**, T = {row['T']:.5f} years, D(T) = {row['discount']:.6f}")
example = one[one["used"]][["minute", "strike", "price_c", "price_p", "spot", "forward", "ratio"]]
st.dataframe(example.rename(columns={"price_c": "C", "price_p": "P", "spot": "spot that minute",
                                     "forward": "F = K + (C − P)/D", "ratio": "F / spot"}),
             hide_index=True, width="stretch")
if row["source"] == "parity":
    st.markdown(
        f"Median of the {int(row['n_pairs'])} ratios = **{row['ratio']:.6f}**; × snapshot spot "
        f"{run.observations.spot.value:,.2f} = forward **{row['forward']:,.2f}**."
    )
else:
    st.markdown(
        f"Only {len(example)} usable pair(s), fewer than the minimum of {run.settings.min_pairs}: "
        f"the forward **{row['forward']:,.2f}** was interpolated from neighbouring parity expiries."
    )

with st.expander("Spot during the window"):
    spot_path = run.bars.drop_duplicates("timestamp")[["timestamp", "spot"]].sort_values("timestamp")
    st.plotly_chart(px.line(spot_path, x="timestamp", y="spot", height=280), width="stretch")
    st.caption("Why pairs must share a minute: the index moves while the window is open.")

with st.expander("All pairs"):
    st.dataframe(residuals, hide_index=True, width="stretch")

_shared.show_source(parity_pairs, label="Pairs")
_shared.show_source(parity_forwards, label="Forwards")
