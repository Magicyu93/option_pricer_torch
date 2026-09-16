"""Thin client for Massive's stock-dividend reference endpoint."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests

from ...common.http import JsonApiClient


class MassiveDividendsSource(JsonApiClient):
    """Retrieve provider-native dividend records, following all pages."""

    BASE_URL = "https://api.massive.com"
    ENDPOINT = "/stocks/v1/dividends"

    @staticmethod
    def _redact_api_key(url: str) -> str:
        parsed = urlparse(url)
        query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key.lower() != "apikey"
            ]
        )
        return urlunparse(parsed._replace(query=query))

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(
            api_key,
            api_key_env="MASSIVE_API_KEY",
            provider_name="Massive REST",
            request_timeout=request_timeout,
            session=session,
        )

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        url: str | None = f"{self.BASE_URL}{self.ENDPOINT}"
        params: dict[str, Any] | None = {
            "ticker": ticker,
            "ex_dividend_date.gte": start_date,
            "ex_dividend_date.lte": end_date,
            "limit": 5000,
            "sort": "ex_dividend_date.asc",
            "apiKey": self.require_api_key(),
        }
        pages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        while url:
            payload = self.get_json(
                url,
                params=params,
            )
            if str(payload.get("status", "OK")).upper() not in {"OK", "DELAYED"}:
                raise RuntimeError(
                    f"Massive dividends API error: {payload.get('error') or payload}"
                )
            stored_payload = dict(payload)
            if payload.get("next_url"):
                stored_payload["next_url"] = self._redact_api_key(
                    str(payload["next_url"])
                )
            pages.append(stored_payload)
            records.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            if next_url:
                url = urljoin(self.BASE_URL, str(next_url))
                if urlparse(url).netloc != urlparse(self.BASE_URL).netloc:
                    raise RuntimeError(
                        "Massive returned a pagination URL on another host"
                    )
                # Current responses generally omit apiKey from next_url.  Do
                # not add a duplicate if the provider included it.
                params = (
                    None
                    if "apiKey" in parse_qs(urlparse(url).query)
                    else {"apiKey": self.require_api_key()}
                )
            else:
                url = None

        return {"pages": pages, "records": records}
