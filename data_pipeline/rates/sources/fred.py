"""Thin client for raw FRED series and observations."""

from __future__ import annotations

from typing import Any

import requests

from ...common.http import JsonApiClient


class FredRatesSource(JsonApiClient):
    """Retrieve provider-native FRED observations without transformations."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(
            api_key,
            api_key_env="FRED_API_KEY",
            provider_name="FRED",
            request_timeout=request_timeout,
            session=session,
        )

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = self.get_json(
            f"{self.BASE_URL}/{endpoint}",
            params={**params, "file_type": "json"},
            api_key_param="api_key",
        )
        if "error_message" in payload:
            raise RuntimeError(f"FRED API error: {payload['error_message']}")
        return payload

    def fetch_series_metadata(self, series_id: str) -> dict[str, Any]:
        payload = self._get("series", {"series_id": series_id})
        records = payload.get("seriess") or []
        if not records:
            raise LookupError(f"FRED series {series_id!r} was not found")
        return {"response": payload, "record": records[0]}

    def fetch_observations(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Fetch current-vintage observations in the requested date range."""
        limit = 100_000
        offset = 0
        pages: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        while True:
            payload = self._get(
                "series/observations",
                {
                    "series_id": series_id,
                    "observation_start": start_date,
                    "observation_end": end_date,
                    "sort_order": "asc",
                    "limit": limit,
                    "offset": offset,
                    "units": "lin",
                    "output_type": 1,
                },
            )
            pages.append(payload)
            batch = payload.get("observations") or []
            observations.extend(batch)
            count = int(payload.get("count", len(observations)))
            offset += len(batch)
            if not batch or offset >= count:
                break

        return {"pages": pages, "observations": observations}
