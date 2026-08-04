"""Minimal signed DNSE OpenAPI REST client.

Reimplements the signing scheme from the official SDK
(https://github.com/dnse-tech/openapi-sdk `python/dnse/api/common.py`) because
that package is NOT published on PyPI (pypi.org/pypi/openapi-sdk -> 404).

Signature contract, verbatim from the SDK:

    signature_string = "(request-target): {method_lower} {path}\\n"
                       "{date_header_lower}: {date_value}"
                       ["\\nnonce: {nonce}"]
    mac              = HMAC-SHA256(api_secret, signature_string)
    signature        = urlquote(base64(mac), safe="")

Sent as::

    Date:        {date_value}
    version:     {api_version}
    X-Signature: Signature keyId="{api_key}",algorithm="hmac-sha256",
                 headers="(request-target) date",signature="{sig}",nonce="{nonce}"

Note the signature covers the PATH ONLY — the query string is excluded.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse
from uuid import uuid4

DEFAULT_API_VERSION = "2026-05-07"  # SDK common.py default (README says 2026-01-01)
DATE_HEADER = "Date"
ALGORITHM = "hmac-sha256"


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read a .env file into a dict. No dependency on python-dotenv."""
    if path is None:
        # Walk up from this file to the first .env — survives the package
        # being installed from anywhere.
        for parent in Path(__file__).resolve().parents:
            candidate = parent / ".env"
            if candidate.exists():
                path = candidate
                break
    values: dict[str, str] = {}
    if path is None or not path.exists():
        raise SystemExit(".env not found (searched upward from the package)")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class DNSEClient:
    """Signed REST client. Read-only helpers plus an explicit order surface."""

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = "https://openapi.dnse.com.vn",
                 api_version: str | None = None, timeout: float = 30.0):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version or os.getenv("DNSE_API_VERSION") or DEFAULT_API_VERSION
        self._timeout = timeout

    # --- signing ---

    def _sign(self, method: str, path: str, date_value: str, nonce: str) -> str:
        header_key = DATE_HEADER.lower()
        signature_string = (
            f"(request-target): {method.lower()} {path}\n"
            f"{header_key}: {date_value}\n"
            f"nonce: {nonce}"
        )
        mac = hmac.new(self._api_secret.encode(), signature_string.encode(), hashlib.sha256)
        escaped = parse.quote(base64.b64encode(mac.digest()).decode(), safe="")
        return (f'Signature keyId="{self._api_key}",algorithm="{ALGORITHM}",'
                f'headers="(request-target) {header_key}",signature="{escaped}",'
                f'nonce="{nonce}"')

    def request(self, method: str, path: str, *, query: dict | None = None,
                body: dict | None = None, headers: dict | None = None
                ) -> tuple[int, object]:
        """Send a signed request. Returns (status, parsed_body_or_text)."""
        url = self._base_url + path
        if query:
            clean = {k: v for k, v in query.items() if v is not None}
            if clean:
                url += "?" + parse.urlencode(clean)

        date_value = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        nonce = uuid4().hex

        send_headers = dict(headers or {})
        send_headers[DATE_HEADER] = date_value
        send_headers["version"] = self._api_version
        # ``x-api-key`` is required alongside the signature — present in the
        # SDK's ``_request`` but absent from the ``common.py`` sample.
        send_headers["x-api-key"] = self._api_key
        # Signature covers the PATH ONLY, never the query string.
        send_headers["X-Signature"] = self._sign(method, path, date_value, nonce)

        data = None
        if body is not None:
            data = json.dumps(body).encode()
            send_headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(url, data=data, method=method.upper())
        for key, value in send_headers.items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                text = resp.read().decode()
                status = resp.status
        except urllib.error.HTTPError as err:
            text = err.read().decode() if err.fp else ""
            status = err.code
        except urllib.error.URLError as err:
            return 0, f"connection error: {err.reason}"

        try:
            return status, json.loads(text) if text else None
        except json.JSONDecodeError:
            return status, text

    # --- read-only ---

    def get_accounts(self):
        return self.request("GET", "/accounts")

    def get_balances(self, account_no: str):
        return self.request("GET", f"/accounts/{account_no}/balances")

    def get_positions(self, account_no: str, market_type: str):
        return self.request("GET", f"/accounts/{account_no}/positions",
                            query={"marketType": market_type})

    def get_orders(self, account_no: str, market_type: str, order_category: str | None = None):
        return self.request("GET", f"/accounts/{account_no}/orders",
                            query={"marketType": market_type,
                                   "orderCategory": order_category})

    def get_order_detail(self, account_no: str, order_id: str, market_type: str,
                         order_category: str | None = None):
        return self.request("GET", f"/accounts/{account_no}/orders/{order_id}",
                            query={"marketType": market_type,
                                   "orderCategory": order_category})

    def get_loan_packages(self, account_no: str, market_type: str):
        return self.request("GET", f"/accounts/{account_no}/loan-packages",
                            query={"marketType": market_type})

    def get_instruments(self, symbol: str | None = None):
        return self.request("GET", "/instruments", query={"symbol": symbol})

    def get_secdef(self, symbol: str):
        return self.request("GET", f"/price/{symbol}/secdef")

    def get_ohlc(self, bar_type: str, query: dict | None = None):
        """bar_type: STOCK | DERIVATIVE | INDEX. query: symbol, resolution, from, to
        (from/to are unix SECONDS)."""
        request_query = dict(query or {})
        request_query["type"] = bar_type
        return self.request("GET", "/price/ohlc", query=request_query)

    # --- trading token (OTP) ---

    def send_email_otp(self):
        return self.request("POST", "/registration/send-email-otp")

    def create_trading_token(self, otp_type: str, passcode: str):
        return self.request("POST", "/registration/trading-token",
                            body={"otpType": otp_type, "passcode": passcode})

    # --- order surface (requires trading token) ---

    def post_order(self, market_type: str, payload: dict, trading_token: str,
                   order_category: str = "NORMAL"):
        return self.request("POST", "/accounts/orders",
                            query={"marketType": market_type,
                                   "orderCategory": order_category},
                            body=payload,
                            headers={"trading-token": trading_token})

    def cancel_order(self, account_no: str, order_id: str, market_type: str,
                     trading_token: str, order_category: str | None = None):
        return self.request("DELETE", f"/accounts/{account_no}/orders/{order_id}",
                            query={"marketType": market_type,
                                   "orderCategory": order_category},
                            headers={"trading-token": trading_token})


def client_from_env() -> DNSEClient:
    env = load_env()
    key, secret = env.get("DNSE_API_KEY", ""), env.get("DNSE_API_SECRET", "")
    if not key or not secret or key.startswith("your-"):
        raise SystemExit("DNSE_API_KEY / DNSE_API_SECRET not set in .env")
    return DNSEClient(key, secret,
                      base_url=env.get("DNSE_BASE_URL", "https://openapi.dnse.com.vn"),
                      api_version=env.get("DNSE_API_VERSION"))
