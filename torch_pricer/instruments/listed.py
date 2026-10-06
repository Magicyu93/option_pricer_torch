"""Listed option products: the contract terms shared by every series on an underlying.

A product is what an exchange lists under one underlying -- often several
option roots, which can differ in how they settle. For SPX, the ``SPX`` root
(monthlies, LEAPS) settles at the open of expiry day on a special opening
quotation, while ``SPXW`` settles at the close; both are European. These are
facts about the contract, not about a calibration, and anything that reads
market data for a product -- the observation window, put-call parity -- takes
them from here.

The mapping from a provider's option root to its underlying (``SPXW`` ->
``SPX``) belongs to the data pipeline, which parses the provider's symbols;
``roots`` here must name the same roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch_pricer.conventions import DEFAULT_CONTRACT_MULTIPLIER
from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Style


class Settlement(str, Enum):
    """When on expiry day an option's settlement value is fixed."""

    AM = "am"  # at the open, on a special opening quotation
    PM = "pm"  # at the close

    def hours_before_close(self, open_to_close_hours: float = 6.5) -> float:
        """How much earlier than a PM-settled option this one stops living."""
        return open_to_close_hours if self is Settlement.AM else 0.0


@dataclass(frozen=True, slots=True)
class OptionRoot:
    """One option root of a product, e.g. ``SPXW``, and how it settles."""

    root: str
    settlement: Settlement = Settlement.PM

    def __post_init__(self) -> None:
        object.__setattr__(self, "settlement", Settlement(self.settlement))


@dataclass(frozen=True, slots=True)
class OptionProduct:
    """The options listed on one underlying."""

    underlying: str
    style: Style
    roots: tuple[OptionRoot, ...]
    description: str = ""
    multiplier: float = DEFAULT_CONTRACT_MULTIPLIER

    def __post_init__(self) -> None:
        object.__setattr__(self, "style", Style(self.style))
        if not self.roots:
            raise ValidationError(f"{self.underlying}: a product needs at least one option root")

    @property
    def is_european(self) -> bool:
        """Whether put-call parity holds as an equality for these options."""
        return self.style is Style.EUROPEAN

    @property
    def root_names(self) -> frozenset[str]:
        return frozenset(r.root for r in self.roots)

    def settlement(self, root: str) -> Settlement:
        """How options under ``root`` settle. Unknown roots raise rather than guess."""
        for r in self.roots:
            if r.root == root:
                return r.settlement
        raise ValidationError(f"{root!r} is not a root of {self.underlying}")

    def am_settled_roots(self) -> frozenset[str]:
        return frozenset(r.root for r in self.roots if r.settlement is Settlement.AM)

    def describe(self) -> str:
        roots = ", ".join(f"{r.root} ({r.settlement.value.upper()})" for r in self.roots)
        return f"{self.underlying}: {self.style.value}; roots {roots}"


#: The products whose options the library knows how to read, by underlying.
LISTED_PRODUCTS: dict[str, OptionProduct] = {
    "SPX": OptionProduct(
        underlying="SPX",
        style=Style.EUROPEAN,
        roots=(OptionRoot("SPX", Settlement.AM), OptionRoot("SPXW", Settlement.PM)),
        description="S&P 500 index options: SPX monthlies and LEAPS (AM), SPXW weeklies (PM)",
    ),
    "SPY": OptionProduct(
        underlying="SPY",
        style=Style.AMERICAN,
        roots=(OptionRoot("SPY", Settlement.PM),),
        description="SPDR S&P 500 ETF options: American exercise, discrete dividends",
    ),
}


def listed_product(underlying: str) -> OptionProduct:
    """The product listed on ``underlying``, or a :class:`ValidationError` naming the known ones."""
    try:
        return LISTED_PRODUCTS[underlying]
    except KeyError:
        raise ValidationError(
            f"no listed product for {underlying!r}; known: {sorted(LISTED_PRODUCTS)}"
        ) from None

