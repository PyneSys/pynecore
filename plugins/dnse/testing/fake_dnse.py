"""Local DNSE API clone — a fake venue for testing the pynecore-dnse plugins.

Replays GOLDEN FIXTURES captured verbatim from the real service
(``record_fixtures.py`` -> ``dnse_fixtures.json``) so response shapes are 1-1
with production, and layers a stateful order book on top so the full
place -> read -> amend -> cancel lifecycle works.

The point is the ``session_open`` switch: real VN30F1M accepts orders only
09:00-11:30 / 13:00-14:45 ICT, which blocks order testing ~19 hours a day.
Here the session is a flag.

    python fake_dnse.py [--port 8888] [--closed]

Then point the plugin at it — no plugin code changes:

    base_url = "http://127.0.0.1:8888"
    ws_url   = "ws://127.0.0.1:8888"

Control endpoints (not part of the real API, prefixed ``_control``):

    POST /_control/session   {"open": true|false}
    GET  /_control/state
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

FIXTURES_PATH = Path(__file__).parent / "dnse_fixtures.json"
ACCOUNT_NO = "0001000000"          # matches the redacted fixtures
VALID_OTP = "123456"
TRADING_TOKEN = "faketoken-0000-1111-2222-333344445555"[:36]

#: Error bodies copied verbatim from the real service. A fake that invents its
#: own wording tests nothing — the plugin's handling keys off these.
ERRORS = {
    "no_api_key": (401, {"status": "error", "message": "X-API-Key header required",
                         "code": "OA-401"}),
    "closed_session": (400, {"status": 400, "code": "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION",
                             "message": "Can not place an order in the closed session"}),
    "price_undefined": (400, {"status": 400, "code": "STOCK_PRICE_UNDEFINED",
                              "message": "Cannot determine stock price"}),
    "bad_otp": (500, {"status": 500, "code": "OA-301",
                      "message": "Failed to obtain trading token: status=0 code=INVALID_OTP, "
                                 "message=The SMS OTP is invalid; is expired; have not been "
                                 "requested or have been used."}),
    "no_token": (401, {"status": 401, "code": "OA-401",
                       "message": "trading-token header required"}),
}


def not_found(order_id: str):
    return 400, {"status": 400, "code": "RESOURCE_NOT_FOUND",
                 "message": f"OrderRepository: cannot find object with id: {order_id}"}


#: DNSE resolution -> ccxt timeframe, for the BTC-backed feed.
_RESOLUTION_TO_CCXT = {"1": "1m", "3": "3m", "5": "5m", "15": "15m",
                       "30": "30m", "1H": "1h", "1D": "1d", "1W": "1w"}

#: Where the synthetic VN30F1M series is anchored, in index points.
VN30F1M_ANCHOR = 1925.0


class BTCFeed:
    """Live BTC bars, rescaled to look like VN30F1M.

    VN30F1M trades 09:00-14:45 ICT; BTC trades continuously. Serving rescaled
    BTC lets a real strategy run against the plugin at any hour while every
    price stays inside VN30F1M's plausible range and on its 0.1 tick.

    The transform is a pure scale factor pinned so the LAST close lands on
    :data:`VN30F1M_ANCHOR` — relative moves, and therefore every indicator, are
    preserved; only the absolute level changes.
    """

    def __init__(self, pair: str = "BTC/USDT", ttl: float = 20.0):
        self.pair = pair
        self.ttl = ttl
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def ohlc(self, resolution: str, limit: int = 500) -> dict:
        import time as _time
        key = resolution
        with self._lock:
            hit = self._cache.get(key)
            if hit and _time.time() - hit[0] < self.ttl:
                return hit[1]

        import ccxt
        timeframe = _RESOLUTION_TO_CCXT.get(resolution, "15m")
        rows = ccxt.binance().fetch_ohlcv(self.pair, timeframe, limit=limit)
        if not rows:
            return {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}

        scale = VN30F1M_ANCHOR / float(rows[-1][4])

        def px(value: float) -> float:
            # VN30F1M ticks at 0.1 index points.
            return round(float(value) * scale, 1)

        payload = {
            "t": [int(r[0] // 1000) for r in rows],
            "o": [px(r[1]) for r in rows],
            "h": [px(r[2]) for r in rows],
            "l": [px(r[3]) for r in rows],
            "c": [px(r[4]) for r in rows],
            "v": [float(r[5]) for r in rows],
            "nextTime": None,
        }
        with self._lock:
            self._cache[key] = (_time.time(), payload)
        return payload


class VenueState:
    """Mutable venue state, guarded by a lock (ThreadingHTTPServer)."""

    def __init__(self, session_open: bool = True, btc_feed: BTCFeed | None = None):
        self.lock = threading.Lock()
        self.session_open = session_open
        self.orders: dict[str, dict] = {}
        self.next_id = 100000
        self.otp_outstanding: str | None = None
        #: Reference prices per tradable symbol, used for the +/-7% band check.
        self.reference: dict[str, float] = {"41I1G8000": VN30F1M_ANCHOR, "HPG": 22.15}
        #: When set, /price/ohlc is served from rescaled live BTC.
        self.btc_feed = btc_feed

    def new_order_id(self) -> str:
        self.next_id += 1
        return str(self.next_id)


class Fixtures:
    """Static GET responses replayed from the recorded corpus."""

    def __init__(self, path: Path):
        self.by_key: dict[tuple[str, str], list[dict]] = {}
        if not path.exists():
            raise SystemExit(f"fixtures not found: {path}\nRun record_fixtures.py first.")
        for entry in json.loads(path.read_text()):
            self.by_key.setdefault((entry["method"], entry["path"]), []).append(entry)

    def match(self, method: str, path: str, query: dict) -> tuple[int, object] | None:
        """Best fixture for (method, path), preferring one whose recorded query
        agrees with the request on the keys that select a response."""
        candidates = self.by_key.get((method, path))
        if not candidates:
            return None
        best, best_score = candidates[0], -1
        for entry in candidates:
            recorded = entry.get("query") or {}
            score = sum(1 for k in ("marketType", "orderCategory", "type", "symbol",
                                    "resolution")
                        if k in recorded and str(recorded[k]) == query.get(k))
            if score > best_score:
                best, best_score = entry, score
        return best["status"], best["body"]


class Handler(BaseHTTPRequestHandler):
    fixtures: Fixtures
    state: VenueState

    # --- plumbing ---

    def log_message(self, fmt, *args):
        print(f"  fake-dnse: {fmt % args}")

    def _send(self, status: int, body: object) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def _authed(self) -> bool:
        """The real service requires x-api-key ALONGSIDE X-Signature."""
        return bool(self.headers.get("x-api-key"))

    # --- dispatch ---

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        body = self._read_body() if method in ("POST", "PUT") else {}

        if path.startswith("/_control"):
            return self._control(method, path, body)
        if not self._authed():
            return self._send(*ERRORS["no_api_key"])

        status, response = self._route(method, path, query, body)
        self._send(status, response)

    def _control(self, method: str, path: str, body: dict) -> None:
        state = self.state
        if path == "/_control/session" and method == "POST":
            with state.lock:
                state.session_open = bool(body.get("open", True))
                return self._send(200, {"session_open": state.session_open})
        if path == "/_control/state":
            with state.lock:
                return self._send(200, {"session_open": state.session_open,
                                        "orders": len(state.orders),
                                        "otp_outstanding": bool(state.otp_outstanding)})
        self._send(404, {"code": "NOT_FOUND", "message": path})

    # --- routing ---

    def _route(self, method: str, path: str, query: dict, body: dict):
        state = self.state

        # --- auth / OTP ---
        if path == "/registration/send-email-otp" and method == "POST":
            with state.lock:
                # A new request INVALIDATES any previous code — real behaviour.
                state.otp_outstanding = VALID_OTP
            return 200, {"status": 200, "code": "OA-000", "message": "OK"}

        if path == "/registration/trading-token" and method == "POST":
            otp_type, passcode = body.get("otpType"), body.get("passcode")
            if otp_type == "sms_otp":
                return 400, {"status": 400, "code": "INVALID_INPUT",
                             "message": "Invalid input: otpType is invalid"}
            if otp_type == "smart_otp":
                return 400, {"status": 400, "code": "INVALID_INPUT",
                             "message": "Invalid input: otpType is not registered for this account"}
            with state.lock:
                ok = passcode == state.otp_outstanding
                state.otp_outstanding = None      # single use
            return (200, {"tradingToken": TRADING_TOKEN}) if ok else ERRORS["bad_otp"]

        # --- orders ---
        order_match = re.fullmatch(r"/accounts/([^/]+)/orders(?:/([^/]+))?", path)
        if path == "/accounts/orders" and method == "POST":
            return self._place(query, body)
        if order_match:
            _, order_id = order_match.groups()
            if method == "GET" and order_id == "history":
                if "from" not in query:
                    return 400, {"status": 400, "code": "INVALID_INPUT",
                                 "message": "from is required"}
                return 200, {"orders": []}
            if method == "GET" and order_id:
                with state.lock:
                    row = state.orders.get(order_id)
                return (200, row) if row else not_found(order_id)
            if method == "GET":
                category = query.get("orderCategory")
                with state.lock:
                    rows = [o for o in state.orders.values()
                            if not category or o["_category"] == category]
                return 200, {"orders": [self._public(o) for o in rows]}
            if method == "PUT" and order_id:
                return self._amend(order_id, body)
            if method == "DELETE" and order_id:
                return self._cancel(order_id)

        # --- BTC-backed OHLC (test mode) ---
        if path == "/price/ohlc" and state.btc_feed is not None:
            resolution = query.get("resolution", "15")
            payload = state.btc_feed.ohlc(resolution)
            if payload["c"]:
                # Keep the band check aligned with the series we are serving,
                # so orders priced off these bars are accepted.
                with state.lock:
                    state.reference["41I1G8000"] = payload["c"][-1]
            return 200, payload

        # --- everything else: replay the recorded corpus ---
        replayed = self.fixtures.match(method, path, query)
        if replayed is not None:
            return replayed
        return 404, {"status": 404, "code": "NOT_FOUND", "message": f"no fixture for {path}"}

    # --- order lifecycle ---

    @staticmethod
    def _public(order: dict) -> dict:
        return {k: v for k, v in order.items() if not k.startswith("_")}

    def _place(self, query: dict, payload: dict):
        state = self.state
        if not self.headers.get("trading-token"):
            return ERRORS["no_token"]
        with state.lock:
            if not state.session_open:
                return ERRORS["closed_session"]
            symbol = payload.get("symbol", "")
            reference = state.reference.get(symbol)
            # An unknown symbol (e.g. the VN30F1M *symbolType* alias) and an
            # out-of-band price both surface as STOCK_PRICE_UNDEFINED.
            if reference is None:
                return ERRORS["price_undefined"]
            price = float(payload.get("price") or 0)
            if not (reference * 0.93 <= price <= reference * 1.07):
                return ERRORS["price_undefined"]
            if not payload.get("loanPackageId"):
                return 400, {"status": 400, "code": "INVALID_INPUT",
                             "message": "loanPackageId is required"}

            order_id = state.new_order_id()
            quantity = int(payload.get("quantity") or 0)
            order = {
                "id": order_id, "side": payload.get("side"), "accountNo": ACCOUNT_NO,
                "symbol": symbol, "price": price, "priceSecure": price,
                "averagePrice": 0.0, "quantity": quantity, "fillQuantity": 0,
                "canceledQuantity": 0, "leaveQuantity": quantity,
                "orderType": payload.get("orderType"), "orderStatus": "NEW",
                "loanPackageId": payload.get("loanPackageId"),
                "marketType": query.get("marketType"),
                "transDate": "", "createdDate": "", "modifiedDate": "",
                "_category": query.get("orderCategory") or "NORMAL",
            }
            state.orders[order_id] = order
            return 200, self._public(order)

    def _amend(self, order_id: str, payload: dict):
        state = self.state
        if not self.headers.get("trading-token"):
            return ERRORS["no_token"]
        with state.lock:
            order = state.orders.get(order_id)
            if not order:
                return not_found(order_id)
            if order["orderStatus"] in ("FILLED", "CANCELLED"):
                return 400, {"status": 400, "code": "INVALID_INPUT",
                             "message": "order is not amendable"}
            if "price" in payload:
                order["price"] = float(payload["price"])
            if "quantity" in payload:
                order["quantity"] = int(payload["quantity"])
                order["leaveQuantity"] = order["quantity"] - order["fillQuantity"]
            return 200, self._public(order)

    def _cancel(self, order_id: str):
        state = self.state
        if not self.headers.get("trading-token"):
            return ERRORS["no_token"]
        # NOTE: cancel is deliberately NOT session-gated — verified against the
        # real service, where DELETE works while placement is refused.
        with state.lock:
            order = state.orders.get(order_id)
            if not order:
                return not_found(order_id)
            order["orderStatus"] = "CANCELLED"
            order["canceledQuantity"] = order["leaveQuantity"]
            order["leaveQuantity"] = 0
            return 200, self._public(order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--closed", action="store_true",
                        help="start with the session CLOSED")
    parser.add_argument("--btc", action="store_true",
                        help="serve /price/ohlc from LIVE BTC rescaled to VN30F1M "
                             "levels, so a strategy can run 24/7")
    args = parser.parse_args()

    Handler.fixtures = Fixtures(FIXTURES_PATH)
    Handler.state = VenueState(session_open=not args.closed,
                               btc_feed=BTCFeed() if args.btc else None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fake DNSE on http://127.0.0.1:{args.port}  "
          f"session={'OPEN' if not args.closed else 'CLOSED'}  "
          f"fixtures={len(Handler.fixtures.by_key)} paths  "
          f"ohlc={'LIVE BTC -> VN30F1M' if args.btc else 'fixtures'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
