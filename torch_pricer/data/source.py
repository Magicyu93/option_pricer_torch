"""Where quotes come from, and the shape they arrive in.

Every adapter's job is to produce a
:class:`~torch_pricer.market.market_data.QuoteSet` and nothing else. That type
was in the package from the beginning -- deliberately dumb, no interpolation, no
torch -- with no construction site anywhere; this is the layer that was missing
under it.

The boundary matters more than it looks. Nothing downstream of a ``QuoteSet``
knows which vendor produced it, so a second source is a new file here rather
than a change anywhere else, and a snapshot on disk is indistinguishable from a
live fetch. That is what makes a calibration reproducible: the test suite reads a
frozen chain and never touches the network.

**Vendor greeks and vendor implied vols are ignored on purpose.** A snapshot
usually carries both. Fitting to them would mean calibrating against the vendor's
model -- their forward, their rate curve, their dividend assumption -- and then
reporting the agreement as though it validated ours. Only bids, asks, and last
trades cross this boundary; every derived quantity is computed here.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

from torch_pricer.market.market_data import QuoteSet


@runtime_checkable
class QuoteSource(Protocol):
    """Anything that can hand back a chain for one underlying at one instant."""

    def fetch(self, ticker: str, as_of: dt.date | None = None) -> QuoteSet:
        """The option chain for ``ticker``.

        Args:
            ticker: the underlying's symbol in this source's own convention
            as_of: the snapshot date; ``None`` means the latest available

        Returns:
            Every quote the source has for that underlying, uncleaned. Filtering
            is :mod:`torch_pricer.data.clean`'s job, and keeping the two apart is
            what lets a rejection be attributed to a rule rather than to the feed.
        """
        ...
