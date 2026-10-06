"""Research app home: what data is cached, and downloading more.

Run from the repo root, in a terminal that has the provider keys:

    streamlit run app/Home.py
"""

import datetime as dt

import streamlit as st

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # app/, for _shared

import _shared  # noqa: E402,F401  (puts the repo on the path)
from data_pipeline.common import DEFAULT_DATA_DIR
from data_pipeline.download import (
    credentials_status,
    download_dividends,
    download_equity,
    download_rates,
)

st.set_page_config(page_title="torch_pricer research", layout="wide")
st.title("torch_pricer research")
st.markdown(
    "Download market data, calibrate a day, and inspect every step. "
    "Pages: **Calibrate** → run a calibration; **Pipeline** → how it was computed; "
    "**Rates & forwards**, **Put-call parity**, **Vol surface** → the results."
)

# -- what is cached ----------------------------------------------------------------
st.header("Cached data")
st.caption(f"Cache: `{DEFAULT_DATA_DIR}`")
files = _shared.cached_coverage()
if files.empty:
    st.info("Nothing cached yet. Download some data below.")
else:
    processed = files[files["layer"] == "parquet"]
    equity = processed[processed["dataset"].isin(["options", "stocks", "indices"])]
    if not equity.empty:
        st.subheader("Options and underlyings, by day")
        table = equity.pivot_table(index="date", columns=["dataset", "ticker"], values="bytes",
                                   aggfunc="sum").map(lambda b: f"{b / 1e6:.1f} MB" if b == b else "")
        st.dataframe(table, width="stretch")
    yearly = processed[processed["dataset"].isin(["rates", "dividends"])]
    if not yearly.empty:
        st.subheader("Rates and dividends, by year")
        st.dataframe(yearly[["dataset", "year", "series", "ticker", "provider", "bytes"]],
                     hide_index=True, width="stretch")
    raw_mb = files.loc[files["layer"] == "raw", "bytes"].sum() / 1e6
    st.caption(f"{len(files)} files; raw downloads {raw_mb:,.0f} MB. "
               "Cache layout: `<layer>/<dataset>/<key>=<value>/.../<kind>_<provider>.<ext>`.")

# -- downloading -------------------------------------------------------------------
st.header("Download")
status = credentials_status()
for provider, info in status.items():
    icon = "✅" if info["ok"] else "❌"
    st.markdown(f"{icon} **{provider}** — `{info['variables']}`")
if not all(info["ok"] for info in status.values()):
    st.warning(
        "Some keys are missing from this process's environment. Keys set in `~/.zshrc` reach "
        "Streamlit only if it was started from a terminal that loaded them."
    )

with st.form("download"):
    col1, col2, col3 = st.columns(3)
    tickers = col1.text_input("Tickers (comma separated)", "SPX").replace(" ", "").upper().split(",")
    today = dt.date.today()
    start = col2.date_input("From", today - dt.timedelta(days=7))
    end = col3.date_input("To", today - dt.timedelta(days=1))
    col4, col5, col6, col7 = st.columns(4)
    want_equity = col4.checkbox("Options + underlyings", True)
    want_rates = col5.checkbox("Treasury yields", True)
    want_dividends = col6.checkbox("Dividends", False)
    provider = col7.radio("Yields from", ["massive", "fred"], horizontal=True)
    st.caption(
        "Options come as one flat file per day for the whole market (~50 MB) and are "
        "filtered to the tickers; cached days are not downloaded again."
    )
    submitted = st.form_submit_button("Download", type="primary")

if submitted:
    tickers = [t for t in tickers if t]
    if start > end:
        st.error("'From' is after 'To'.")
        st.stop()
    with st.status("Downloading...", expanded=True) as box:
        log = box.write
        failures = []
        if want_equity:
            failures += [r for r in download_equity(tickers, start, end, on_progress=log) if not r.ok]
        if want_rates:
            result = download_rates(start, end, provider=provider, on_progress=log)
            failures += [] if result.ok else [result]
        if want_dividends:
            failures += [r for r in download_dividends(tickers, start, end, on_progress=log) if not r.ok]
        _shared.cached_coverage.clear()
        box.update(label=f"Done: {len(failures)} failed" if failures else "Done",
                   state="error" if failures else "complete")
    st.button("Refresh the table")
