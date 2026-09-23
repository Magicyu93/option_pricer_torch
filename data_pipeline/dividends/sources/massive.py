"""Thin client for Massive's stock-dividend reference endpoint."""

from __future__ import annotations

from typing import Any

from ...common.massive import MassiveRestClient


class MassiveDividendsSource(MassiveRestClient):
    """Retrieve provider-native dividend records, following all pages."""

    ENDPOINT = "/stocks/v1/dividends"

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        params = {
            "ticker": ticker,
            "ex_dividend_date.gte": start_date,
            "ex_dividend_date.lte": end_date,
            "limit": 5000,
            "sort": "ex_dividend_date.asc",
        }
        return self.fetch_all(self.ENDPOINT, params, label="dividends")
