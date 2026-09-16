"""Shared HTTP setup for read-only market-data sources."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_retry_session(
    *,
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> requests.Session:
    """Return a session that retries transient GET failures."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class JsonApiClient:
    """Shared transport behavior for API-key-authenticated JSON sources.

    Provider-specific clients still own endpoint paths, pagination, response
    validation, and domain semantics.  This class only centralizes credentials,
    timeout validation, retry-session construction, and JSON decoding.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        api_key_env: str,
        provider_name: str,
        request_timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.provider_name = provider_name
        self.request_timeout = float(request_timeout)
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.session = session or build_retry_session()

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"{self.provider_name} API key is missing. Set "
                f"{self.api_key_env} or pass api_key=."
            )
        return self.api_key

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        api_key_param: str | None = None,
    ) -> dict[str, Any]:
        query = dict(params or {})
        if api_key_param is not None:
            query[api_key_param] = self.require_api_key()
        response = self.session.get(
            url,
            params=query or None,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"{self.provider_name} returned a non-object JSON response")
        return payload
