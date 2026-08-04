"""Record REAL DNSE responses as golden fixtures for the fake venue.

A hand-written mock drifts from the API and then passes while the plugin would
fail live. These fixtures are captured verbatim from the real service so the
fake can replay byte-identical shapes.

PII (name, custody code, investor id, account number) is redacted with
SAME-LENGTH, same-type placeholders so schemas and field widths stay faithful.

Usage:  python record_fixtures.py [out.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pynecore_dnse.client import client_from_env

REDACT = {
    "name": "NGUYEN VAN A",
    "custodyCode": "064C000000",
    "investorId": "1000000000",
    "accountNo": "0001000000",
    "id": None,          # only redacted at account level, handled explicitly
}


def redact(node, account_no: str, real_id: str | None):
    """Replace PII in-place, preserving type and length."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ("name", "custodyCode", "investorId"):
                out[key] = REDACT[key]
            elif key in ("accountNo",) or (key == "id" and value == real_id):
                out[key] = REDACT["accountNo"]
            else:
                out[key] = redact(value, account_no, real_id)
        return out
    if isinstance(node, list):
        return [redact(v, account_no, real_id) for v in node]
    if isinstance(node, str) and account_no and account_no in node:
        return node.replace(account_no, REDACT["accountNo"])
    return node


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                    Path(__file__).parent / "dnse_fixtures.json")
    client = client_from_env()

    status, accounts = client.get_accounts()
    if status != 200:
        raise SystemExit(f"cannot reach DNSE: {status} {accounts}")
    account_no = accounts["accounts"][0]["id"]
    now = int(time.time())

    calls: list[tuple[str, str, dict | None]] = [
        ("GET", "/accounts", None),
        ("GET", f"/accounts/{account_no}/balances", None),
        ("GET", f"/accounts/{account_no}/positions", {"marketType": "DERIVATIVE"}),
        ("GET", f"/accounts/{account_no}/positions", {"marketType": "STOCK"}),
        ("GET", f"/accounts/{account_no}/orders", {"marketType": "DERIVATIVE",
                                                   "orderCategory": "NORMAL"}),
        ("GET", f"/accounts/{account_no}/orders", {"marketType": "DERIVATIVE",
                                                   "orderCategory": "CONDITIONAL"}),
        ("GET", f"/accounts/{account_no}/orders/history", {"marketType": "DERIVATIVE"}),
        ("GET", f"/accounts/{account_no}/loan-packages", {"marketType": "DERIVATIVE"}),
        ("GET", f"/accounts/{account_no}/loan-packages", {"marketType": "STOCK"}),
        ("GET", f"/accounts/{account_no}/ppse", {"marketType": "DERIVATIVE",
                                                 "symbol": "41I1G8000", "price": 1925,
                                                 "loanPackageId": 1306}),
        ("GET", "/instruments", {"limit": 20}),
        ("GET", "/price/41I1G8000/secdef", None),
        ("GET", "/price/HPG/secdef", None),
        ("GET", "/price/HPG/close", None),
        ("GET", "/price/HPG/quotes/latest", None),
        ("GET", "/price/HPG/trades/latest", None),
        ("GET", "/market/trading-session", None),
        ("GET", "/market/working-dates", None),
        ("GET", "/price/ohlc", {"type": "DERIVATIVE", "symbol": "VN30F1M",
                                "resolution": "15", "from": now - 3 * 86400, "to": now}),
        ("GET", "/price/ohlc", {"type": "STOCK", "symbol": "HPG",
                                "resolution": "1D", "from": now - 30 * 86400, "to": now}),
        # error shapes worth freezing
        ("GET", f"/accounts/{account_no}/orders/999999999", {"marketType": "DERIVATIVE"}),
        ("GET", "/price/ohlc", {"type": "DERIVATIVE", "symbol": "VN30F2612",
                                "resolution": "15", "from": now - 86400, "to": now}),
    ]

    fixtures = []
    for method, path, query in calls:
        status, body = client.request(method, path, query=query)
        fixtures.append({
            "method": method,
            "path": path.replace(account_no, REDACT["accountNo"]),
            "query": query,
            "status": status,
            "body": redact(body, account_no, account_no),
        })
        print(f"  [{status}] {method} {path}{'?' + str(query) if query else ''}")

    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1))
    print(f"\nwrote {len(fixtures)} fixtures -> {out_path}")


if __name__ == "__main__":
    main()
