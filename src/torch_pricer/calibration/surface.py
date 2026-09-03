"""Implied-vol surfaces: clean, arbitrage-free quoted vol as a function of strike and expiry.

A surface is a *description of the market's quotes*. It is not the dynamics --
that is a :class:`~torch_pricer.calibration.vol_model.base.VolModel`, which is
fitted to a surface and knows how to produce an SDE. Local vol is exactly the
map between the two, which is why it deserves both a surface and a model.

Everything below works in **total implied variance** ``w(k, T) = sigma(k, T)^2
T`` over log-moneyness ``k = log(K / F(T))``, and not in implied vol over
strike. Three reasons, all of which matter downstream:

* Dupire's formula is a ratio of derivatives of ``w`` in exactly these
  coordinates, and writing it any other way buys a page of chain rule;
* the no-arbitrage conditions are clean here -- calendar arbitrage is
  ``dw/dT >= 0`` at fixed ``k``, butterfly arbitrage is one inequality in ``w``
  and its two ``k``-derivatives;
* interpolating linearly in ``w`` between expiries preserves the first of those
  automatically, while interpolating in vol does not.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..errors import CalibrationError, ValidationError
from ..tensors import EPS, as_tensor

if TYPE_CHECKING:  # pragma: no cover
    from .inputs import CalibrationInputs


class VolSurface(ABC):
    """Implied volatility as a function of strike and expiry."""

    @abstractmethod
    def vol(self, strike, expiry) -> Tensor:
        """Implied vol at ``strike`` for ``expiry`` years out."""

    @property
    @abstractmethod
    def reference_date(self) -> dt.date:
        """The date the surface's expiries are measured from."""

    def total_variance(self, strike, expiry) -> Tensor:
        """``sigma^2 T``, the quantity Dupire and the arbitrage checks are stated in.

        The default reads through :meth:`vol`. A surface parameterised in total
        variance to begin with should override it rather than take the square
        root and square it back.
        """
        expiry = as_tensor(expiry)
        return self.vol(strike, expiry) ** 2 * expiry


class FlatVolSurface(VolSurface):
    """One number everywhere: the Black-Scholes textbook surface."""

    def __init__(self, vol: float, as_of: dt.date):
        self._vol = float(vol)
        self._ref = as_of

    def vol(self, strike, expiry) -> Tensor:
        # Broadcast against the strike so callers get the shape they passed in.
        return as_tensor(self._vol) * as_tensor(strike) ** 0

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"FlatVolSurface({self._vol:.4f}, {self._ref})"


#: The butterfly penalty aims a little inside the feasible set rather than at
#: its boundary: a soft penalty settles where its gradient balances the fit's,
#: which for a target of exactly zero is just below zero.
_BUTTERFLY_MARGIN = 1e-3

#: Where the arbitrage penalties are evaluated: far wider than any quoted strike,
#: because the conditions have to hold where the surface is extrapolated too.
_GUARD_GRID = torch.linspace(-1.5, 1.5, 101, dtype=torch.float64)


def _logit(x: float) -> Tensor:
    """Inverse of ``sigmoid``, with the endpoints nudged inside the open interval."""
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return as_tensor(x).logit()


class SVISlice(nn.Module):
    """Gatheral's raw SVI for one expiry.

        w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

    A hyperbola in ``k``: two straight asymptotes, with ``b`` the overall
    steepness, ``rho`` the tilt between the wings, ``m`` the horizontal shift and
    ``sigma`` the curvature at the bottom. Five parameters reproduce an equity
    smile to within its bid-ask, and -- unlike a spline through the quotes --
    the wings stay linear in ``k``, which is Lee's moment formula and the
    condition for the surface to be extrapolatable at all.

    The parameters are stored unconstrained in the same style as a
    :class:`~torch_pricer.calibration.vol_model.base.VolModel`, and two of the
    substitutions do more than keep an optimiser in bounds -- they build the
    static no-arbitrage conditions into the parameterisation, so that no
    sequence of optimiser steps can reach a slice that is arbitrageable in its
    wings:

    * rather than ``a``, the *minimum* of the hyperbola ``w_min = a + b sigma
      sqrt(1 - rho^2)`` is stored, as its logarithm, so total variance is
      positive at every ``k``;
    * rather than ``(b, rho)``, the two wing slopes ``p = b(1 - rho)`` and
      ``c = b(1 + rho)`` are stored, each as ``2 sigmoid(.)`` and so confined to
      ``(0, 2)``. Those slopes are what ``w`` grows like as ``k -> -+inf``, and
      Lee's moment formula caps them at 2: a slice steeper than that implies an
      underlying with no second moment, and Durrleman's ``g`` goes negative in
      the wing. Since ``b = (p + c)/2`` and ``rho = (c - p)/(c + p)``, this also
      keeps ``b > 0`` and ``|rho| < 1`` for free.

    What the constraints cannot decide is the middle of the smile, where
    butterfly arbitrage is a property of the quotes rather than of the wings;
    :meth:`butterfly_margin` is what checks that.
    """

    def __init__(
        self,
        b: float = 0.1,
        rho: float = -0.5,
        m: float = 0.0,
        sigma: float = 0.2,
        w_min: float = 0.01,
    ):
        super().__init__()
        if b < 0 or sigma <= 0 or w_min <= 0:
            raise ValidationError(
                f"SVI needs b >= 0, sigma > 0, w_min > 0; got {b}, {sigma}, {w_min}"
            )
        if not -1.0 < rho < 1.0:
            raise ValidationError(f"SVI rho must lie strictly inside (-1, 1), got {rho}")
        put_wing, call_wing = b * (1.0 - rho), b * (1.0 + rho)
        if not 0.0 <= put_wing < 2.0 or not 0.0 <= call_wing < 2.0:
            raise ValidationError(
                f"SVI wing slopes b(1 -+ rho) = ({put_wing:.4f}, {call_wing:.4f}) "
                "must lie in [0, 2); "
                "steeper than 2 is Lee's moment bound, and arbitrageable"
            )
        self._logit_p = nn.Parameter(_logit(put_wing / 2.0))
        self._logit_c = nn.Parameter(_logit(call_wing / 2.0))
        self.m = nn.Parameter(as_tensor(float(m)))
        self._log_sigma = nn.Parameter(as_tensor(float(sigma)).log())
        self._log_w_min = nn.Parameter(as_tensor(float(w_min)).log())

    # -- parameters -----------------------------------------------------
    @property
    def put_wing(self) -> Tensor:
        """``b (1 - rho)``: the slope of ``w`` as ``k -> -inf``, in ``(0, 2)``."""
        return 2.0 * torch.sigmoid(self._logit_p)

    @property
    def call_wing(self) -> Tensor:
        """``b (1 + rho)``: the slope of ``w`` as ``k -> +inf``, in ``(0, 2)``."""
        return 2.0 * torch.sigmoid(self._logit_c)

    @property
    def b(self) -> Tensor:
        """Wing steepness, non-negative."""
        return 0.5 * (self.put_wing + self.call_wing)

    @property
    def rho(self) -> Tensor:
        """Tilt between the wings, in ``(-1, 1)``."""
        return (self.call_wing - self.put_wing) / (self.call_wing + self.put_wing)

    @property
    def sigma(self) -> Tensor:
        """Curvature of the vertex."""
        return self._log_sigma.exp()

    @property
    def w_min(self) -> Tensor:
        """Total variance at the bottom of the smile, positive."""
        return self._log_w_min.exp()

    @property
    def a(self) -> Tensor:
        """Vertical shift, implied by :attr:`w_min` and the wings."""
        return self.w_min - self.b * self.sigma * (1.0 - self.rho**2).sqrt()

    # -- the smile ------------------------------------------------------
    def total_variance(self, k) -> Tensor:
        """``w(k)``."""
        k = as_tensor(k)
        z = k - self.m
        return self.a + self.b * (self.rho * z + (z * z + self.sigma**2).sqrt())

    def derivatives(self, k) -> tuple[Tensor, Tensor, Tensor]:
        """``(w, dw/dk, d2w/dk2)``, analytically.

        Autograd would give the same numbers, but this is called inside the
        butterfly check on a dense grid where the closed form is both cheaper
        and free of the graph.
        """
        k = as_tensor(k)
        z = k - self.m
        root = (z * z + self.sigma**2).sqrt()
        w = self.a + self.b * (self.rho * z + root)
        dw = self.b * (self.rho + z / root)
        d2w = self.b * self.sigma**2 / root**3
        return w, dw, d2w

    def butterfly_margin(self, k) -> Tensor:
        """Durrleman's ``g(k)``, non-negative exactly where the slice is butterfly-free.

        ``g`` is the density of the implied distribution up to a positive
        factor, so ``g(k) < 0`` somewhere means the slice prices a butterfly
        negatively -- a static arbitrage, and a local vol surface built on it
        will ask for a negative variance at that strike.
        """
        w, dw, d2w = self.derivatives(k)
        w = w.clamp_min(EPS)
        return (1.0 - k * dw / (2.0 * w)) ** 2 - (dw**2 / 4.0) * (1.0 / w + 0.25) + 0.5 * d2w

    def is_butterfly_free(self, k_range: float = 1.5, n: int = 201) -> bool:
        """Whether :meth:`butterfly_margin` stays non-negative across the wings."""
        k = torch.linspace(-abs(k_range), abs(k_range), int(n), dtype=self.m.dtype)
        with torch.no_grad():
            return bool((self.butterfly_margin(k) >= 0).all())

    # -- fitting --------------------------------------------------------
    def fit(
        self,
        k,
        vol,
        expiry: float,
        weight=None,
        iterations: int = 500,
        lr: float = 0.05,
        arbitrage_penalty: float = 10.0,
        k_range: float = 1.5,
        calendar_floor: Tensor | None = None,
    ) -> float:
        """Least squares in *volatility*, with a butterfly penalty. Returns the RMSE.

        Fitting ``w`` directly would weight the long-dated quotes and the far
        wings by the square of what a trader cares about; the quotes are vols,
        and the residual should be in vols.

        The penalty is on Durrleman's ``g`` falling below a small positive
        margin anywhere in ``[-k_range, k_range]``, which is wider than the
        quotes reach. A slice fitted only to the quotes is unconstrained between
        the last strike and its asymptote, and that gap is exactly where an
        unpenalised least-squares fit likes to put a small negative density --
        harmless to the fit, fatal to the Dupire denominator that will later
        divide by it. Since the region carries no quotes, holding it
        arbitrage-free costs the fit essentially nothing.

        ``calendar_floor`` is the previous expiry's total variance on the same
        guard grid, penalised the same way. Butterfly arbitrage is a property of
        one slice, but calendar arbitrage is a property of a pair of them: total
        variance must not fall as expiry grows, at any moneyness. Nothing in a
        slice-by-slice fit enforces that, so it is enforced here, one slice at a
        time, in the order the expiries come.
        """
        # Detached on the way in: the strikes reach here through a forward built
        # from curve parameters, and a live graph would make the second
        # optimiser step backward through a graph that the first one freed.
        k = as_tensor(k).detach()
        target = as_tensor(vol).detach()
        t = max(float(expiry), EPS)
        weight = torch.ones_like(target) if weight is None else as_tensor(weight).detach()
        weight = weight / weight.sum().clamp_min(EPS)
        guard = (
            _GUARD_GRID.to(target.dtype)
            if k_range == 1.5
            else torch.linspace(-abs(k_range), abs(k_range), 101, dtype=target.dtype)
        )

        def residual() -> Tensor:
            model = (self.total_variance(k).clamp_min(EPS) / t).sqrt()
            return (weight * (model - target) ** 2).sum()

        def objective() -> Tensor:
            deficit = (_BUTTERFLY_MARGIN - self.butterfly_margin(guard)).clamp_min(0.0)
            penalty = (deficit**2).mean()
            if calendar_floor is not None:
                crossed = (calendar_floor - self.total_variance(guard)).clamp_min(0.0)
                penalty = penalty + (crossed**2).mean()
            return residual() + arbitrage_penalty * penalty

        adam = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(int(iterations)):
            adam.zero_grad()
            loss = objective()
            loss.backward()
            adam.step()

        lbfgs = torch.optim.LBFGS(self.parameters(), max_iter=100, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            lbfgs.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            return float(residual().sqrt())

    def extra_repr(self) -> str:  # pragma: no cover
        f = lambda x: float(x.detach())  # noqa: E731
        return (
            f"a={f(self.a):+.5f}, b={f(self.b):.5f}, rho={f(self.rho):+.4f}, "
            f"m={f(self.m):+.4f}, sigma={f(self.sigma):.4f}"
        )


class SVISurface(VolSurface):
    """SVI fitted per expiry, interpolated linearly in total variance between them.

    The slices are independent fits; what ties them together is the
    interpolation. Linear in ``w`` at fixed log-moneyness makes ``w`` monotone in
    ``T`` between two calendar-consistent slices, so no calendar arbitrage can be
    introduced by the interpolation itself -- only by the quotes. Outside the
    quoted range ``w`` is scaled proportionally to ``T``, which holds the implied
    vol of the nearest slice flat rather than inventing a term structure.

    The surface needs the forward to convert a strike into log-moneyness, so it
    carries the same forward function the fit was built on.
    """

    def __init__(
        self,
        as_of: dt.date,
        expiries: Sequence[float],
        slices: Sequence[SVISlice],
        forward: Callable[[Tensor], Tensor],
    ):
        if len(expiries) != len(slices):
            raise ValidationError(f"{len(expiries)} expiries but {len(slices)} slices")
        if not slices:
            raise ValidationError("a surface needs at least one slice")
        order = sorted(range(len(expiries)), key=lambda i: float(expiries[i]))
        self._expiries = as_tensor([float(expiries[i]) for i in order])
        if bool((self._expiries <= 0).any()):
            raise ValidationError("slice expiries must be positive year fractions")
        self.slices = tuple(slices[i] for i in order)
        self._forward = forward
        self._ref = as_of

    # -- construction ---------------------------------------------------
    @classmethod
    def fit(
        cls,
        inputs: CalibrationInputs,
        iterations: int = 500,
        check_arbitrage: bool = True,
    ) -> SVISurface:
        """Fit one slice per quoted expiry.

        Quotes are used at their implied vol, backed out of the premium when the
        quote carries only a price. A slice needs at least five quotes to
        determine five parameters; expiries with fewer are skipped rather than
        fitted to something the data cannot pin down.

        Residuals are weighted by Black vega. A far wing quote on a short expiry
        is a premium of a few basis points, and the vol implied by it is mostly
        the rounding of that premium; weighting by vega says so, and stops one
        such point from tilting a whole slice into arbitrage.
        """
        from ..analytics.black import black_vega as _black_vega
        from ..analytics.black import implied_vol as _implied_vol

        quotes = inputs.quotes
        if quotes is None or not quotes.options:
            raise CalibrationError("fitting a surface needs option quotes")

        expiries: list[float] = []
        slices: list[SVISlice] = []
        for expiry in quotes.expiries():
            t = inputs.time_to(expiry)
            if t <= 0:
                continue
            group = quotes.slice(expiry)
            forward = inputs.forward(t).detach()
            discount = inputs.discount.discount(t).detach()

            strikes = as_tensor([float(q.strike) for q in group])
            rights = as_tensor([float(q.right.sign) for q in group])
            quoted = as_tensor(
                [float("nan") if q.implied_vol is None else q.implied_vol for q in group]
            )
            premium = as_tensor([float("nan") if q.price is None else q.price for q in group])
            solved = _implied_vol(premium, forward, strikes, t, discount, rights)
            vols = torch.where(torch.isnan(quoted), solved, quoted)

            usable = ~torch.isnan(vols)
            if int(usable.sum()) < 5:
                continue
            k = torch.log(strikes[usable] / forward)
            vol = vols[usable]
            weight = _black_vega(forward, strikes[usable], t, vol, discount).clamp_min(EPS)

            slice_ = SVISlice(
                b=0.1, rho=-0.5, m=0.0, sigma=0.2, w_min=max(float(vol.min()) ** 2 * t, 1e-6)
            )
            floor = None
            if slices:
                with torch.no_grad():
                    floor = slices[-1].total_variance(_GUARD_GRID.to(k.dtype))
            slice_.fit(k, vol, t, weight=weight, iterations=iterations, calendar_floor=floor)
            if check_arbitrage and not slice_.is_butterfly_free():
                raise CalibrationError(
                    f"the fitted slice at T={t:.4f} admits butterfly arbitrage; "
                    "the quotes for that expiry are not arbitrage-free, or are too sparse to fit"
                )
            expiries.append(t)
            slices.append(slice_)

        if not slices:
            raise CalibrationError("no expiry carried five usable quotes")
        return cls(inputs.as_of, expiries, slices, inputs.forward)

    # -- queries --------------------------------------------------------
    def log_moneyness(self, strike, expiry) -> Tensor:
        """``log(K / F(T))``, the coordinate the slices are stated in."""
        strike, expiry = torch.broadcast_tensors(as_tensor(strike), as_tensor(expiry))
        return torch.log(strike / self._forward(expiry))

    def total_variance(self, strike, expiry) -> Tensor:
        strike, expiry = torch.broadcast_tensors(as_tensor(strike), as_tensor(expiry))
        shape = strike.shape
        k = self.log_moneyness(strike, expiry).reshape(-1)
        t = expiry.reshape(-1).clamp_min(EPS)

        pillars = self._expiries.to(t.dtype)
        ws = torch.stack([s.total_variance(k) for s in self.slices], dim=0)  # (n_slices, n_points)
        if pillars.numel() == 1:
            # One slice: hold its implied vol flat in time, so w scales with T.
            return (ws[0] * t / pillars[0]).reshape(shape)

        idx = torch.searchsorted(pillars, t.detach().contiguous()).clamp(1, pillars.numel() - 1)
        t0, t1 = pillars[idx - 1], pillars[idx]
        w0 = ws.gather(0, (idx - 1).unsqueeze(0)).squeeze(0)
        w1 = ws.gather(0, idx.unsqueeze(0)).squeeze(0)
        inside = w0 + (w1 - w0) * (t - t0) / (t1 - t0)

        # Flat implied vol beyond the quoted range, in both directions.
        before = ws[0] * t / pillars[0]
        after = ws[-1] * t / pillars[-1]
        w = torch.where(t < pillars[0], before, torch.where(t > pillars[-1], after, inside))
        return w.clamp_min(EPS).reshape(shape)

    def vol(self, strike, expiry) -> Tensor:
        expiry = as_tensor(expiry)
        return (self.total_variance(strike, expiry) / expiry.clamp_min(EPS)).sqrt()

    def calendar_margin(self, k_range: float = 1.0, n_k: int = 41) -> Tensor:
        """``w(k, T_{i+1}) - w(k, T_i)`` across the slices; negative means calendar arbitrage."""
        k = torch.linspace(-abs(k_range), abs(k_range), int(n_k), dtype=self._expiries.dtype)
        with torch.no_grad():
            ws = torch.stack([s.total_variance(k) for s in self.slices], dim=0)
        return ws[1:] - ws[:-1]

    @property
    def expiries(self) -> Tensor:
        """The fitted slice expiries, in years."""
        return self._expiries

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        ts = ", ".join(f"{float(t):.3f}" for t in self._expiries)
        return f"SVISurface({self._ref}, {len(self.slices)} slices at [{ts}])"
