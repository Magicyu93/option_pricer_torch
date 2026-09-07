"""The market-data layer: adapters, cleaning, and parity-implied forwards.

Everything here runs offline. The chain is synthetic and built from parameters we
choose, which is the point -- a real snapshot can only be checked for
self-consistency, while a manufactured one has a known right answer, so the
parity regression can be asked to recover a forward we already know.
"""

import dataclasses
import datetime as dt
import math

import numpy as np
import pytest

from torch_pricer.data import cache as cache_mod
from torch_pricer.data import synthetic
from torch_pricer.data.clean import CleaningRules, clean, static_arbitrage
from torch_pricer.data.massive import MassiveSource
from torch_pricer.data.source import QuoteSource
from torch_pricer.errors import MarketDataError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.forward import (
    curves_from_forwards,
    implied_forward,
    implied_forwards,
)
from torch_pricer.market.market_data import OptionQuote, QuoteSet, SpotQuote
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.pricer.analytic.black import black_price, implied_vol

AS_OF = dt.date(2026, 9, 4)
SPOT, RATE, DIV = 5500.0, 0.042, 0.013
EXPIRIES = (dt.date(2026, 12, 18), dt.date(2027, 3, 19), dt.date(2027, 6, 18))


def _year_fraction(expiry):
    return (expiry - AS_OF).days / 365.0


def _vol(strike, forward):
    return synthetic.smile(strike, forward)


def synthetic_chain(**kw):
    """The shared generator, pinned to this module's parameters."""
    return synthetic.chain(AS_OF, EXPIRIES, spot=SPOT, rate=RATE, dividend=DIV,
                           ticker="I:SPX", **kw)


# -- adapter and cache ----------------------------------------------------


def _pages():
    return [
        {"results": [
            {"details": {"expiration_date": "2026-12-18", "strike_price": 5500,
                         "contract_type": "call"},
             "last_quote": {"bid": 100.0, "ask": 102.0},
             "underlying_asset": {"price": 5512.25},
             "implied_volatility": 0.19, "greeks": {"delta": 0.5}},
            {"details": {"expiration_date": "2026-12-18", "strike_price": 5500,
                         "contract_type": "put"},
             "last_quote": {"bid": 80.0, "ask": 82.0}},
            {"details": {"expiration_date": "2026-12-18", "strike_price": 9999,
                         "contract_type": "call"}},          # never traded
        ], "next_url": "https://api.polygon.io/next"},
        {"results": [
            {"details": {"expiration_date": "2027-01-15", "strike_price": 5600,
                         "contract_type": "call"},
             "last_trade": {"price": 44.0},
             "underlying_asset": {"price": 5512.25}}]},
    ]


def _source():
    it = iter(_pages())
    return MassiveSource(api_key="test", fetch_json=lambda url, params: next(it))


def test_adapter_satisfies_the_source_protocol():
    assert isinstance(_source(), QuoteSource)


def test_adapter_follows_pagination_and_drops_unpriceable_rows():
    quotes = _source().fetch("I:SPX")
    assert quotes.spot.value == 5512.25
    assert len(quotes.options) == 3                    # the 9999 call is dropped
    assert len(quotes.expiries()) == 2                 # both pages were read
    assert quotes.options[0].price == 101.0            # mid of 100/102


def test_adapter_ignores_vendor_implied_vols_and_greeks():
    """Taking them would calibrate against the vendor's forward and curve, then
    report the agreement as validation of ours."""
    quotes = _source().fetch("I:SPX")
    assert all(q.implied_vol is None for q in quotes.options)


def test_adapter_refuses_without_a_key():
    with pytest.raises(MarketDataError, match="no Massive API key"):
        MassiveSource(api_key=None, fetch_json=lambda u, p: {}).fetch("I:SPX")


def test_adapter_refuses_a_past_date_rather_than_returning_today():
    """The snapshot endpoint has only the latest chain. Silently substituting it
    for a requested history date is the kind of error that dates a whole study."""
    with pytest.raises(MarketDataError, match="cannot be fetched"):
        _source().fetch("I:SPX", as_of=dt.date(2020, 1, 1))


def test_snapshots_round_trip_through_disk(tmp_path):
    original = synthetic_chain()
    path = cache_mod.save(original, tmp_path / "spx.json")
    back = cache_mod.load(path)
    assert back.as_of == original.as_of
    assert len(back.options) == len(original.options)
    assert back.options[5].price == pytest.approx(original.options[5].price)


def test_a_cache_miss_is_an_error_not_an_empty_chain(tmp_path):
    with pytest.raises(MarketDataError, match="no cached snapshot"):
        cache_mod.CachedSource(tmp_path).fetch("I:SPX", dt.date(2020, 1, 1))


def test_cached_source_writes_through_and_then_reads_back(tmp_path):
    cached = cache_mod.CachedSource(tmp_path, source=_source())
    first = cached.fetch("I:SPX", as_of=dt.date.today())
    # The injected transport is exhausted, so a second hit can only come from disk.
    second = cached.fetch("I:SPX", as_of=first.as_of)
    assert len(second.options) == len(first.options)


# -- cleaning -------------------------------------------------------------


def test_cleaning_counts_every_rejection_by_reason():
    good = synthetic_chain().options[:4]
    bad = (
        OptionQuote(EXPIRIES[0], 5500.0, Right.CALL, bid=12.0, ask=10.0),   # crossed
        OptionQuote(EXPIRIES[0], 5600.0, Right.CALL, bid=0.0, ask=4.0),     # zero bid
        OptionQuote(EXPIRIES[0], 5700.0, Right.CALL, bid=1.0, ask=9.0),     # wide
        OptionQuote(EXPIRIES[0], 5800.0, Right.CALL, last=3.0),             # one-sided
        OptionQuote(EXPIRIES[0], 5900.0, Right.CALL, bid=0.01, ask=0.02),   # sub-tick
    )
    chain = QuoteSet(AS_OF, SpotQuote("I:SPX", SPOT), options=tuple(good) + bad)
    cleaned, report = clean(chain)
    assert report.kept == 4
    assert set(report.rejected) == {"crossed", "zero-bid", "wide-spread",
                                    "one-sided", "below-tick"}
    assert report.total == len(chain.options)
    assert len(cleaned.options) == 4


def test_a_generated_chain_is_free_of_static_arbitrage():
    assert static_arbitrage(synthetic_chain()).clean


def test_static_arbitrage_catches_a_broken_call_ladder():
    """Monotonicity is model-free: a call cannot cost more as the strike rises."""
    chain = synthetic_chain()
    options = list(chain.options)
    calls = [i for i, q in enumerate(options)
             if q.right is Right.CALL and q.expiry == EXPIRIES[0]]
    victim = options[calls[10]]
    options[calls[10]] = dataclasses.replace(
        victim, bid=victim.bid * 5, ask=victim.ask * 5
    )
    report = static_arbitrage(dataclasses.replace(chain, options=tuple(options)))
    assert not report.clean
    assert report.monotonicity


# -- the parity forward ---------------------------------------------------


def test_parity_recovers_the_forward_and_discount_it_was_built_from():
    """The headline. Nothing but option quotes goes in -- no rate curve, no
    dividend estimate -- and both come back."""
    chain = synthetic_chain()
    for expiry in EXPIRIES:
        t = _year_fraction(expiry)
        fitted = implied_forward(chain, expiry, t)
        assert fitted.forward == pytest.approx(SPOT * math.exp((RATE - DIV) * t), rel=1e-9)
        assert fitted.discount == pytest.approx(math.exp(-RATE * t), rel=1e-9)
        assert fitted.max_residual < 1e-8
        assert fitted.zero_rate == pytest.approx(RATE, rel=1e-6)


def test_parity_survives_a_noisy_two_sided_market():
    """Real mids are not exact. Independent per-quote noise should move the fit
    by far less than it moves any single quote, which is the point of regressing
    rather than reading one straddle."""
    rng = np.random.default_rng(0)
    chain = synthetic_chain()
    noisy = tuple(
        dataclasses.replace(q, bid=q.bid + rng.normal(0, 0.4),
                            ask=q.ask + rng.normal(0, 0.4))
        for q in chain.options
    )
    fitted = implied_forward(dataclasses.replace(chain, options=noisy), EXPIRIES[0],
                             _year_fraction(EXPIRIES[0]))
    true_forward = SPOT * math.exp((RATE - DIV) * _year_fraction(EXPIRIES[0]))
    assert fitted.forward == pytest.approx(true_forward, rel=2e-4)


def test_too_few_paired_strikes_is_an_error():
    chain = synthetic_chain()
    two = tuple(q for q in chain.options
                if q.expiry == EXPIRIES[0] and q.strike in
                {chain.options[0].strike})
    with pytest.raises(MarketDataError, match="two-sided strike pairs"):
        implied_forward(dataclasses.replace(chain, options=two), EXPIRIES[0], 0.3)


def test_curves_reproduce_the_parity_forwards_through_a_snapshot():
    """The objective of the whole exercise: after this, the log-moneyness a
    calibration sees is the market's own, so a residual is the model's fault and
    not the curve's."""
    chain = synthetic_chain()
    fits = implied_forwards(chain, _year_fraction)
    assert len(fits) == len(EXPIRIES)

    discount, dividend = curves_from_forwards(fits, SPOT)
    snapshot = dataclasses.replace(
        MarketSnapshot.flat(AS_OF, spot=SPOT, ticker="I:SPX"),
        discount=discount, dividend=dividend,
    )
    import torch

    for fit in fits:
        rebuilt = float(snapshot.forward(torch.tensor(fit.t, dtype=torch.float64)))
        assert rebuilt == pytest.approx(fit.forward, rel=1e-10)


def test_end_to_end_recovers_the_volatility_the_chain_was_built_from():
    """Chain -> clean -> parity forward -> invert premiums -> the input smile.

    Every step of the pipeline runs, and the answer is known in advance because
    the chain was manufactured. A real snapshot could only be checked for
    self-consistency; this can be checked for correctness.
    """
    cleaned, report = clean(synthetic_chain(), CleaningRules(min_price=0.5))
    assert report.kept > 100

    expiry = EXPIRIES[1]
    t = _year_fraction(expiry)
    fit = implied_forward(cleaned, expiry, t)

    worst = 0.0
    for q in cleaned.slice(expiry):
        if q.right is not Right.CALL:
            continue
        vol = float(implied_vol(q.price, fit.forward, q.strike, t, fit.discount, 1))
        assert not math.isnan(vol)
        worst = max(worst, abs(vol - _vol(q.strike, fit.forward)))
    assert worst < 1e-6


# -- the whole pipeline, into a calibration -------------------------------


def test_a_chain_calibrates_an_svi_surface_that_reproduces_its_smile():
    """Snapshot -> clean -> parity forwards -> curves -> implied vols -> SVI fit.

    Every layer built for Phase C runs here, ending in the calibrator that
    already existed and had no way to be fed. The acceptance test is in *vol
    points* against the smile the chain was generated from, because that is the
    unit the market quotes and the unit a fit is judged in.
    """
    import torch

    from torch_pricer.calibration.inputs import CalibrationInputs
    from torch_pricer.calibration.svi_fit import fit_surface
    from torch_pricer.data.implied import attach_implied_vols, otm_only
    from torch_pricer.market.svi import SVISlice, SVISurface

    cleaned, _ = clean(synthetic_chain(), CleaningRules(min_price=0.5))
    fits = implied_forwards(cleaned, _year_fraction)
    discount, dividend = curves_from_forwards(fits, SPOT)
    snapshot = dataclasses.replace(
        MarketSnapshot.flat(AS_OF, spot=SPOT, ticker="I:SPX"),
        discount=discount, dividend=dividend,
    )

    with_vols, inversion = attach_implied_vols(cleaned, fits)
    assert inversion.inverted == inversion.total          # nothing unusable
    quotes = otm_only(with_vols, fits)

    surface = SVISurface(
        slices=[SVISlice(0.02, 0.08, -0.4, 0.0, 0.2, snapshot.time_to(e))
                for e in EXPIRIES],
        forward=snapshot.forward,
        as_of=AS_OF,
    )
    result = fit_surface(surface, CalibrationInputs(market=snapshot, quotes=quotes))

    # Residuals are in vol, by construction of the calibrator.
    assert float(result.residuals.abs().max()) < 5e-3

    # And the fitted surface reproduces the generating smile away from the fit's
    # own knots, which a residual alone would not show.
    for fit in fits:
        for moneyness in (0.9, 1.0, 1.1):
            strike = fit.forward * moneyness
            fitted = float(surface.vol(
                torch.tensor(strike, dtype=torch.float64), fit.t
            ).detach())
            assert fitted == pytest.approx(_vol(strike, fit.forward), abs=6e-3)
