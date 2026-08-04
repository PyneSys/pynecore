"""WebSocket half of the fake DNSE venue — streams bars and order events.

Runs alongside ``fake_dnse.py`` (REST). Two servers, two ports, one shared
:class:`VenueState`, so a resting order placed over REST is filled by the
price stream and the fill is pushed back over WS.

    python fake_dnse_ws.py [--port 8888] [--ws-port 8889] [--btc] [--speed 5]

Plugin config:

    base_url = "http://127.0.0.1:8888"
    ws_url   = "ws://127.0.0.1:8889"

FIDELITY NOTE: the REST bodies are golden fixtures recorded from the real
service, but the **WS message envelope is inferred**, not recorded — a live
DNSE stream was never captured (the market was shut). The channel NAMES come
from DNSE's documented list; the payload wrapper is our best reading and must
be re-checked against a real session before trusting parity.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from http.server import ThreadingHTTPServer

from fake_dnse import (BTCFeed, Fixtures, Handler, VenueState,
                       FIXTURES_PATH, VN30F1M_ANCHOR)


class Streamer:
    """Drives bars and fills for all connected websocket clients."""

    def __init__(self, state: VenueState, speed: float, resolution: str = "1"):
        self.state = state
        self.speed = speed            # seconds between synthetic bars
        self.resolution = resolution
        self.clients: set = set()
        self.subscriptions: dict[int, set[str]] = {}
        self._index = 0

    # --- fan-out ---

    async def broadcast(self, prefix: str, message: dict) -> None:
        dead = []
        for ws in list(self.clients):
            channels = self.subscriptions.get(id(ws), set())
            if not any(c.startswith(prefix) for c in channels):
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # --- bar loop ---

    async def run(self) -> None:
        """Emit one closed bar every ``speed`` seconds, then settle orders."""
        while True:
            await asyncio.sleep(self.speed)
            bar = self._next_bar()
            if bar is None:
                continue
            await self.broadcast("ohlc", {"channel": f"ohlc_closed.{self.resolution}.json",
                                          "data": bar})
            for event in self._settle(bar):
                await self.broadcast("order", {"channel": "order.DERIVATIVE.json",
                                               "data": event})

    def _next_bar(self) -> dict | None:
        feed = self.state.btc_feed
        if feed is None:
            return None
        series = feed.ohlc(self.resolution)
        if not series["c"]:
            return None
        # Walk forward through the BTC series, wrapping when exhausted.
        self._index = (self._index + 1) % len(series["c"])
        i = self._index
        bar = {"t": int(time.time()), "o": series["o"][i], "h": series["h"][i],
               "l": series["l"][i], "c": series["c"][i], "v": series["v"][i],
               "symbol": "41I1G8000"}
        with self.state.lock:
            self.state.reference["41I1G8000"] = bar["c"]
        return bar

    # --- fill simulation ---

    def _settle(self, bar: dict) -> list[dict]:
        """Fill any resting order the bar's range touched."""
        events: list[dict] = []
        with self.state.lock:
            for order in self.state.orders.values():
                if order["orderStatus"] not in ("NEW", "PARTIALLY_FILLED"):
                    continue
                price = float(order["price"])
                if not (bar["l"] <= price <= bar["h"]):
                    continue
                order["fillQuantity"] = order["quantity"]
                order["leaveQuantity"] = 0
                order["averagePrice"] = price
                order["orderStatus"] = "FILLED"
                events.append({k: v for k, v in order.items()
                               if not k.startswith("_")})
        return events


def make_app(state: VenueState, streamer: Streamer):
    from aiohttp import web, WSMsgType

    async def stream(request):
        ws = web.WebSocketResponse(heartbeat=None)
        await ws.prepare(request)
        streamer.clients.add(ws)
        streamer.subscriptions[id(ws)] = set()
        try:
            async for msg in ws:
                if msg.type is not WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                action = payload.get("action")
                if action == "auth":
                    # The real service verifies HMAC(api_key:timestamp:nonce);
                    # the fake only checks the envelope is well formed.
                    ok = all(k in payload for k in
                             ("api_key", "signature", "timestamp", "nonce"))
                    await ws.send_json({"action": "auth", "success": ok,
                                        "session_id": f"fake-{id(ws)}"})
                elif action == "subscribe":
                    for channel in payload.get("channels") or []:
                        streamer.subscriptions[id(ws)].add(channel.get("name", ""))
                    await ws.send_json({"action": "subscribe", "success": True})
                elif action == "pong":
                    pass
        finally:
            streamer.clients.discard(ws)
            streamer.subscriptions.pop(id(ws), None)
        return ws

    app = web.Application()
    app.router.add_get("/v1/stream", stream)
    return app


def serve_rest(state: VenueState, port: int) -> None:
    Handler.fixtures = Fixtures(FIXTURES_PATH)
    Handler.state = state
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


async def amain(args) -> None:
    from aiohttp import web

    state = VenueState(session_open=not args.closed,
                       btc_feed=BTCFeed() if args.btc else None)
    threading.Thread(target=serve_rest, args=(state, args.port), daemon=True).start()

    streamer = Streamer(state, speed=args.speed, resolution=args.resolution)
    runner = web.AppRunner(make_app(state, streamer))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.ws_port).start()

    print(f"fake DNSE  REST http://127.0.0.1:{args.port}  "
          f"WS ws://127.0.0.1:{args.ws_port}/v1/stream")
    print(f"  session={'OPEN' if not args.closed else 'CLOSED'}  "
          f"bars={'LIVE BTC -> VN30F1M' if args.btc else 'none'}  "
          f"every {args.speed}s  resolution={args.resolution}")
    await streamer.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--ws-port", type=int, default=8889)
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--btc", action="store_true")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="seconds between synthetic closed bars")
    parser.add_argument("--resolution", default="1")
    args = parser.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
