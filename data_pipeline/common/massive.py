"""Shared transport for Massive's paginated REST endpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests

from .http import JsonApiClient


class MassiveRestClient(JsonApiClient):
    """Follow Massive ``next_url`` pagination and keep every page for audit.

    Subclasses own the endpoint, its query parameters, and what a record means.
    """

    BASE_URL = "https://api.massive.com"

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

    def fetch_all(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        """GET ``endpoint`` and every following page.

        Returns ``{"pages": [...], "records": [...]}`` with the API key removed
        from any stored ``next_url``.
        """
        url: str | None = f"{self.BASE_URL}{endpoint}"
        query: dict[str, Any] | None = {**params, "apiKey": self.require_api_key()}
        pages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        while url:
            payload = self.get_json(url, params=query)
            if str(payload.get("status", "OK")).upper() not in {"OK", "DELAYED"}:
                raise RuntimeError(
                    f"Massive {label} API error: {payload.get('error') or payload}"
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
                query = (
                    None
                    if "apiKey" in parse_qs(urlparse(url).query)
                    else {"apiKey": self.require_api_key()}
                )
            else:
                url = None

        return {"pages": pages, "records": records}
