"""What every page shares: library calls behind Streamlit's cache, and the current run.

No computation lives here. Each function calls exactly one library function
(``data_pipeline`` or ``torch_pricer``) and caches its result, so moving between
pages does not repeat work. The formulas below are display text: they describe
what the library functions named next to them compute.
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # run with `streamlit run app/Home.py`, no install needed
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from data_pipeline import MassiveHistoricalDataLoader, RatesDataLoader  # noqa: E402
from data_pipeline.coverage import coverage  # noqa: E402
from torch_pricer.instruments.listed import listed_product  # noqa: E402
from torch_pricer.market.build import SnapshotRun, SnapshotSettings, build_snapshot  # noqa: E402
from torch_pricer.market.curve.calibration import par_yield_quotes  # noqa: E402
from torch_pricer.market.market_data import QuoteConfig  # noqa: E402

RUN_KEY, COMPARE_KEY, SOURCES_KEY = "run", "comparison", "sources"


# -- cached library calls ----------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60)
def cached_coverage() -> pd.DataFrame:
    return coverage()


@st.cache_data(show_spinner="Loading minute bars from the cache...")
def market_frame(ticker: str, day: str) -> pd.DataFrame:
    return MassiveHistoricalDataLoader(verbose=False).load_market([day], [ticker])


@st.cache_data(show_spinner="Loading Treasury yields...")
def par_yields(day: str, provider: str, lag_days: int):
    snap = RatesDataLoader(provider).yields_as_of(day, lag_days=lag_days)
    return snap.tenors.tolist(), snap.yields_pct.tolist(), snap.observation_date


@st.cache_resource(show_spinner="Calibrating...", max_entries=8)
def calibrate(
    ticker: str, day: str, provider: str, lag_days: int, config: QuoteConfig,
    curve_model: str, surface_model: str, max_pairs: int, min_pairs: int, max_iter: int,
) -> SnapshotRun:
    """One calibration per distinct set of settings, kept for the session."""
    tenors, yields_pct, published = par_yields(day, provider, lag_days)
    settings = SnapshotSettings(
        as_of=dt.date.fromisoformat(day), quote_config=config,
        curve_model=curve_model, surface_model=surface_model,
        max_pairs=max_pairs, min_pairs=min_pairs, max_iter=max_iter,
    )
    return build_snapshot(
        market_frame(ticker, day), par_yield_quotes(tenors, yields_pct, as_of=published),
        listed_product(ticker), settings,
    )


# -- the current run ---------------------------------------------------------------

def current_run() -> SnapshotRun | None:
    return st.session_state.get(RUN_KEY)


def require_run() -> SnapshotRun:
    """The run to show, or stop the page with a pointer to where to make one."""
    run = current_run()
    if run is None:
        st.info("No calibration yet: run one on the **Calibrate** page.")
        st.stop()
    return run


def run_caption(run: SnapshotRun) -> None:
    s = run.settings
    st.caption(
        f"{run.product.underlying} on {s.as_of} · window {s.quote_config.window_start}–"
        f"{s.quote_config.window_end} ET · curve: {s.curve_model} · "
        f"surface: {s.surface_model} · spot {run.observations.spot.value:,.2f}"
    )


# -- source transparency -----------------------------------------------------------

def resolve(dotted: str):
    """The library object a trace step names, e.g. ``torch_pricer.market....parity_forwards``."""
    module, _, name = dotted.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):  # a qualname nested one level deeper
        module, _, outer = module.rpartition(".")
        return getattr(getattr(importlib.import_module(module), outer), name)


def show_source(obj, *, label: str = "Source code") -> None:
    """Docstring-free view of an object's source, with where it lives."""
    try:
        path = Path(inspect.getsourcefile(obj)).relative_to(REPO_ROOT)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError, ValueError):
        st.caption("Source not available.")
        return
    with st.expander(f"{label}: `{path}:{start}`"):
        st.code("".join(lines), language="python", line_numbers=True)


def show_doc(obj) -> None:
    doc = inspect.getdoc(obj)
    if doc:
        st.markdown(doc.replace("``", "`"))


# -- what each step computes (display text) ----------------------------------------

FORMULAS: dict[str, list[str]] = {
    "Observation window": [
        r"T = \frac{\text{days}(\text{as\_of} \to \text{expiry})}{365}"
        r"\;-\;\mathbb{1}_{\text{AM}}\,\frac{6.5}{24 \cdot 365}",
    ],
    "Discount curve": [
        r"\text{bill } (T \le \tfrac12):\; z(T) = \frac{\ln(1 + yT)}{T}",
        r"\text{par bond}:\; \sum_i \tfrac{c}{2} D(t_i) + D(T) = 1,\quad z(T) = -\frac{\ln D(T)}{T}",
    ],
    "Put-call pairs": [r"C - P = D(T)\,(F - K) \;\Rightarrow\; F = K + \frac{C - P}{D(T)}"],
    "Parity forwards": [
        r"\frac{F}{S}\Big|_{\text{expiry}} = \operatorname{median}_{\text{nearest pairs}} \frac{F_{\text{pair}}}{S_{\text{minute}}}",
        r"\text{no pairs: } \ln\frac{F}{S} \text{ linear in } T \text{ between parity expiries}",
    ],
    "Implied carry": [r"q(T) = r(T) - \frac{1}{T}\ln\frac{F(T)}{S}"],
    "Option quotes": [
        r"F_{\text{bar}} = \frac{F}{S}\Big|_{\text{expiry}} \cdot S_{\text{minute}},\quad k = \ln\frac{K}{F_{\text{bar}}}",
        r"\text{Black}(F_{\text{bar}}, K, T, \sigma, D) = \text{trade price} \;\Rightarrow\; \sigma_{\text{bar}}",
        r"\sigma_{\text{contract}} = \frac{\sum \text{volume} \cdot \sigma_{\text{bar}}}{\sum \text{volume}}",
    ],
    "Surface fit": [
        r"\min \sum_i w_i\,\big(\sigma_{\text{model}}(k_i, T_i) - \sigma_i\big)^2,"
        r"\quad w_i \propto \sqrt{\text{volume}_i}\; e^{-d_{1,i}^2/2}",
        r"\text{SSVI: } w(k,T) = \frac{\theta}{2}\Big(1 + \rho\varphi k + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2}\Big),"
        r"\; \varphi = \frac{\eta}{\theta^{\gamma}(1+\theta)^{1-\gamma}}",
    ],
    "Market snapshot": [r"F(T) = S\,\frac{D_q(T)}{D_r(T)},\qquad \sigma(K, T) = \sigma_{\text{surface}}\big(\ln\tfrac{K}{F(T)}, T\big)"],
}


def show_formulas(step: str) -> None:
    for formula in FORMULAS.get(step, []):
        st.latex(formula)
