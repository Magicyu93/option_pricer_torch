"""Thin client for Massive's daily Treasury constant-maturity yields."""

from __future__ import annotations

from typing import Any

from ...common.massive import MassiveRestClient


class MassiveTreasuryYieldsSource(MassiveRestClient):
    """Retrieve provider-native Treasury yield records, one per date.

    Each record carries every tenor published that day as its own field
    (``yield_1_month`` ... ``yield_30_year``); tenors not published are absent
    rather than null.
    """

    ENDPOINT = "/fed/v1/treasury-yields"

    def fetch(self, start_date: str, end_date: str) -> dict[str, Any]:
        params = {
            "date.gte": start_date,
            "date.lte": end_date,
            "limit": 50_000,
            "sort": "date.asc",
        }
        return self.fetch_all(self.ENDPOINT, params, label="treasury yields")
