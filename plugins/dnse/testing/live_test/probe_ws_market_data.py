#!/usr/bin/env python3
"""#50 capture probe — Market-Data WS with the DOCUMENTED auth handshake.

The earlier "WS channels are silent" measurement predates this probe and is
SUSPECT METHODOLOGY (operator, 2026-08-26): the server requires an explicit
HMAC auth message within 30 s and refuses subscribes before auth succeeds
(sdk-build_websocket.md) — a probe that skipped or failed that handshake sees
exactly "silent channels". This probe follows the documented flow precisely:

    connect -> ready -> auth (HMAC-SHA256 over "{api_key}:{ts}:{nonce}")
            -> subscribe -> capture frames

Read-only market data; credentials come from the broker toml and are NEVER
printed. Usage:
    .venv/bin/python plugins/dnse/testing/live_test/probe_ws_market_data.py [--seconds N]
"""
import argparse
import asyncio
import hashlib
import hmac
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBrokerConfig                   # noqa: E402

import websockets                                                    # noqa: E402

WS_URL = "wss://ws-openapi.dnse.com.vn/v1/stream?encoding=json"  # path from the vendored SDK (websocket/client.py)
CHANNELS = [
    {"name": "tick.G1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "tick_extra.G1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "ohlc.1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "top_price.G1.json", "symbols": ["41I1G9000"]},
]


def _mask(value: str) -> str:
    return value[:4] + "…" + value[-3:] if len(value) > 10 else "***"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=45)
    args = ap.parse_args()

    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    ts = int(time.time())
    nonce = str(int(time.time() * 1_000_000))
    signature = hmac.new(cfg.api_secret.encode(), f"{cfg.api_key}:{ts}:{nonce}".encode(),
                         hashlib.sha256).hexdigest()

    counts: Counter = Counter()
    samples: dict[str, str] = {}

    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
        print(f"ready: action={ready.get('action')} session={str(ready.get('session_id'))[:12]}…")

        await ws.send(json.dumps({"action": "auth", "api_key": cfg.api_key,
                                  "signature": signature, "timestamp": ts,
                                  "nonce": nonce}))
        auth = json.loads(await asyncio.wait_for(ws.recv(), 15))
        print(f"auth: action={auth.get('action')} code={auth.get('code')} "
              f"rate_limit={auth.get('rate_limit')}")
        if auth.get("action") != "auth_success":
            print(f"AUTH FAILED: {auth.get('message')}")
            return 2

        await ws.send(json.dumps({"action": "subscribe", "channels": CHANNELS}))
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), max(0.5, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            try:
                frame = json.loads(raw)
            except (ValueError, TypeError):
                counts["<binary/unparsed>"] += 1
                continue
            action = frame.get("action")
            if action in ("subscribed", "error", "ping", "pong"):
                print(f"control: {json.dumps(frame)[:160]}")
                if action == "ping":
                    await ws.send(json.dumps({"action": "pong"}))
                continue
            key = (str(frame.get("T") or "") + ":" + str(frame.get("symbol") or frame.get("s") or "?")) if frame.get("T") else (frame.get("channel") or action or "<data>")
            counts[str(key)] += 1
            samples.setdefault(str(key), json.dumps(frame)[:240])

    print(f"\n=== capture ({args.seconds}s) ===")
    for key, n in counts.most_common():
        print(f"{n:5d}  {key}")
    for key, sample in samples.items():
        print(f"\nsample [{key}]: {sample}")
    if not counts:
        print("ZERO data frames — silent WITH correct auth (methodology now sound)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
