"""Curves calibrated to market data: the discount curve, forwards, and carry.

One function per curve model, each returning a :class:`~torch_pricer.market.curve.base.Curve`.
Curve models differ too much in what they consume -- par yields here, option
pairs below -- for one generic fitter to pay off; a new model (a spread over
Treasuries, a parametric curve) is a new function next to these.

**Discount curve.** :func:`treasury_curve` bootstraps the quote set's Treasury
par yields (:func:`~torch_pricer.market.curve.treasury.par_to_zero`).

**Forwards and carry.** For a European call and put on the same strike and
expiry, ``C - P = D(T) (F(T) - K)``, so every strike where both traded gives a
forward ``F = K + (C - P) / D`` -- with no model, and no dividend forecast. That
is how a snapshot leaves dividends out without getting them wrong: whatever the
market expects the index to pay is already in ``F``, and the *implied carry*
``q(T) = r(T) - ln(F/S) / T`` stands in for the dividend curve.

Trades are not simultaneous, and the index moves, so a call and a put only make
a pair when they traded in the same minute, and each pair is turned into a
forward-to-spot ratio against that minute's spot. The expiry's ratio is the
median over the pairs nearest the money -- where both legs trade most and the
put-call difference is least sensitive to a stale leg.

Expiries without enough pairs -- most of the long-dated ones, where calls barely
trade -- take their ratio from the expiries that have them: ``ln(F/S)``
interpolated linearly in ``T``, which is a carry rate held piecewise constant,
and that rate held flat beyond the last parity expiry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torch_pricer.conventions import tenor_label, tenor_years
from torch_pricer.errors import MarketDataError
from torch_pricer.market.curve.base import Curve
from torch_pricer.market.curve.curves import RateCurve
from torch_pricer.market.curve.treasury import par_to_zero
from torch_pricer.market.market_data import QuoteSet, RateQuote

FORWARD_COLUMNS = ["expiry", "T", "discount", "ratio", "forward", "n_pairs", "source"]


# -- discount curve ----------------------------------------------------------------

def par_yield_quotes(tenors, yields_pct, as_of=None) -> tuple[RateQuote, ...]:
    """Par-yield quotes from nominal tenors in years and yields in percent.

    The shape the data pipeline's ``ParYieldSnapshot`` carries; the caller
    passes its fields, so nothing here imports the pipeline.
    """
    return tuple(
        RateQuote(tenor_label(float(t)), float(y) / 100.0, kind="par", as_of=as_of)
        for t, y in zip(tenors, yields_pct)
    )


def treasury_curve(quotes: QuoteSet, label: str = "UST") -> RateCurve:
    """The zero curve bootstrapped from the quote set's par yields.

    Raises :class:`MarketDataError` when the quote set carries none, rather than
    falling back to a flat rate nobody chose.
    """
    par = [q for q in quotes.rates if q.kind == "par"]
    if not par:
        raise MarketDataError(f"quote set for {quotes.as_of} has no par yields")
    tenors, zeros = par_to_zero([tenor_years(q.tenor) for q in par], [q.rate for q in par])
    return RateCurve.from_zeros(tenors, zeros, label=label)


# -- forwards and carry ------------------------------------------------------------

PAIR_COLUMNS = [
    "expiry", "T", "timestamp", "strike", "price_c", "price_p", "spot",
    "discount", "forward", "ratio", "distance", "used",
]


def parity_pairs(bars: pd.DataFrame, curve: Curve, *, max_pairs: int = 10) -> pd.DataFrame:
    """Every same-minute call/put pair, with the forward each one implies.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output.
    One row per pair: ``F = K + (C - P) / D(T)``, its ratio to that minute's
    spot, and its log-distance from the money. ``used`` marks the ``max_pairs``
    nearest the money in each expiry -- the ones :func:`parity_forwards` takes
    the median of.
    """
    expiries = bars.groupby("expiry", as_index=False)["T"].first()
    expiries["discount"] = curve.discount(expiries["T"].to_numpy()).detach().numpy()

    calls = bars[bars["right"] > 0]
    puts = bars[bars["right"] < 0]
    keys = ["expiry", "timestamp", "strike"]
    pairs = calls.merge(puts, on=keys, suffixes=("_c", "_p"))
    pairs = pairs.merge(expiries[["expiry", "discount"]], on="expiry")
    pairs["forward"] = pairs["strike"] + (pairs["price_c"] - pairs["price_p"]) / pairs["discount"]
    pairs = pairs[pairs["forward"] > 0].rename(columns={"spot_c": "spot", "T_c": "T"})
    pairs["ratio"] = pairs["forward"] / pairs["spot"]
    pairs["distance"] = np.abs(np.log(pairs["strike"] / pairs["forward"]))
    nearest = pairs.groupby("expiry", group_keys=False).apply(
        lambda g: g.nsmallest(max_pairs, "distance"), include_groups=False
    ).index
    pairs["used"] = pairs.index.isin(nearest)
    return pairs[PAIR_COLUMNS].reset_index(drop=True)


def parity_forwards(
    bars: pd.DataFrame,
    curve: Curve,
    spot: float,
    *,
    max_pairs: int = 10,
    min_pairs: int = 2,
    pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One forward per expiry in ``bars``, at the reference ``spot``.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output;
    ``pairs``, if given, is :func:`parity_pairs` output for the same bars, so
    the pairs a caller displays are exactly the ones used here.
    Returns ``expiry, T, discount, ratio, forward, n_pairs, source`` with
    ``source`` either ``"parity"`` or ``"interpolated"``. Raises
    :class:`MarketDataError` if no expiry has ``min_pairs`` pairs, since then
    there is nothing to interpolate from.
    """
    if pairs is None:
        pairs = parity_pairs(bars, curve, max_pairs=max_pairs)
    expiries = bars.groupby("expiry", as_index=False)["T"].first().sort_values("T")
    expiries["discount"] = curve.discount(expiries["T"].to_numpy()).detach().numpy()

    used = pairs[pairs["used"]]
    parity = used.groupby("expiry").agg(ratio=("ratio", "median"), n_pairs=("ratio", "size"))
    parity = parity[parity["n_pairs"] >= min_pairs].reset_index()
    if parity.empty:
        raise MarketDataError(f"no expiry has {min_pairs} same-minute call/put pairs")

    out = expiries.merge(parity, on="expiry", how="left")
    out["source"] = np.where(out["ratio"].notna(), "parity", "interpolated")
    out["n_pairs"] = out["n_pairs"].fillna(0).astype(int)

    known = out[out["source"] == "parity"]
    log_ratio = np.interp(out["T"], known["T"], np.log(known["ratio"]))
    # np.interp holds ln(F/S) flat past the ends; hold the carry *rate* flat instead.
    first, last = known.iloc[0], known.iloc[-1]
    log_ratio = np.where(out["T"] < first["T"], np.log(first["ratio"]) / first["T"] * out["T"], log_ratio)
    log_ratio = np.where(out["T"] > last["T"], np.log(last["ratio"]) / last["T"] * out["T"], log_ratio)
    out["ratio"] = out["ratio"].fillna(pd.Series(np.exp(log_ratio), index=out.index))
    out["forward"] = spot * out["ratio"]
    return out[FORWARD_COLUMNS].reset_index(drop=True)


def implied_carry_curve(forwards: pd.DataFrame, curve: Curve) -> RateCurve:
    """The dividend curve under which ``S D_q(T) / D_r(T)`` is each expiry's forward.

    A zero curve with one pillar per expiry: ``q(T) = r(T) - ln(F/S) / T``.
    """
    t = forwards["T"].to_numpy(dtype=float)
    r = curve.zero_rate(t).detach().numpy()
    q = r - np.log(forwards["ratio"].to_numpy(dtype=float)) / t
    return RateCurve.from_zeros(t, q, label="implied_carry")


#: Curve models a caller can choose by name: each takes the quote set and
#: returns the discount :class:`~torch_pricer.market.curve.base.Curve`.
CURVE_MODELS = {
    "Treasury bootstrap": treasury_curve,
}


# -- views for inspection ----------------------------------------------------------

def parity_residuals(pairs: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Each pair's ``C - P`` against the line its expiry's forward implies.

    The forward at a pair's minute is the expiry's forward-to-spot ratio times
    that minute's spot, so the line is ``D (ratio * spot - K)``; ``residual`` is
    how far the traded ``C - P`` sits from it, in premium.
    """
    fwd = forwards.set_index("expiry")
    out = pairs.copy()
    out["c_minus_p"] = out["price_c"] - out["price_p"]
    out["expiry_forward"] = fwd.loc[out["expiry"], "ratio"].to_numpy() * out["spot"].to_numpy()
    out["fitted"] = out["discount"] * (out["expiry_forward"] - out["strike"])
    out["residual"] = out["c_minus_p"] - out["fitted"]
    return out


def rates_comparison(forwards: pd.DataFrame, curve: Curve, carry: Curve) -> pd.DataFrame:
    """Per expiry: the discount curve's rate, and what the option forwards imply.

    ``implied_growth = ln(F/S) / T`` is the ``r - q`` the forwards price in;
    ``carry = r - implied_growth`` is the implied carry curve's zero rate; the
    forward columns are each curve's forward rate from the previous expiry.
    All rates continuously compounded, as decimals.
    """
    t = forwards["T"].to_numpy(dtype=float)
    prev = np.concatenate([[0.0], t[:-1]])
    out = forwards[["expiry", "T", "source", "n_pairs"]].copy()
    out["treasury_zero"] = curve.zero_rate(t).detach().numpy()
    out["implied_growth"] = np.log(forwards["ratio"].to_numpy(dtype=float)) / t
    out["carry_zero"] = carry.zero_rate(t).detach().numpy()
    out["treasury_forward"] = curve.forward_rate(prev, t).detach().numpy()
    out["carry_forward"] = carry.forward_rate(prev, t).detach().numpy()
    return out
