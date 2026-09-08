#!/usr/bin/env python3
"""#50 capture probe — Market-Data WS with the DOCUMENTED auth handshake.

The earlier "WS channels are silent" measurement predates this probe and is
SUSPECT METHODOLOGY (operator, 2026-08-26): the server requires an explicit
HMAC auth message within 30 s and refuses subscribes before auth succeeds
(sdk-build_websocket.md) — a probe that skipped or failed that handshake sees
exactly "silent channels". This probe follows the documented flow precisely:

    connect -> ready -> auth (HMAC-SHA256 over "{api_key}:{ts}:{nonce}")
            -> subscribe -> capture frames

#92 (measured 2026-09-08): the server CLOSED a trading-channel session
mid-capture (1000 OK) and the probe crashed with a raw traceback, losing
the close timing — so we could not tell whether the close predated the
account activity. A server close is now a REPORTED measurement (wall time,
elapsed, code/reason, frames-so-far) with exit 3. ``--dual`` measures the
single-session hypothesis (the engine's own WS may contend with a probe):
two sequential authenticated sessions on the same channels; the report
shows which one the server drops.

Read-only; credentials come from the broker toml and are NEVER printed.
Usage:
    .venv/bin/python plugins/dnse/testing/live_test/probe_ws_market_data.py \
        [--seconds N] [--trading] [--dual]

Exit codes: 0 capture ran to its deadline; 2 auth/setup failed; 3 the
server closed the connection early (the close itself is the finding).
"""
import argparse
import asyncio
import datetime
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
TRADING_CHANNELS = [
    {"name": "order.derivative.json", "symbols": []},
    {"name": "order.stock.json", "symbols": []},
    {"name": "position.derivative.json", "symbols": []},
]


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=7))).strftime("%H:%M:%S")


async def _capture(cfg, channels, seconds: int, label: str) -> int:
    """One authenticated capture session. Returns the probe exit code."""
    counts: Counter = Counter()
    samples: dict[str, str] = {}
    started = time.monotonic()
    closed_early: "str | None" = None

    ts = int(time.time())
    nonce = str(int(time.time() * 1_000_000))
    signature = hmac.new(cfg.api_secret.encode(),
                         f"{cfg.api_key}:{ts}:{nonce}".encode(),
                         hashlib.sha256).hexdigest()

    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        try:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print(f"[{label}] ready: action={ready.get('action')} "
                  f"session={str(ready.get('session_id'))[:12]}…")
            await ws.send(json.dumps({"action": "auth", "api_key": cfg.api_key,
                                      "signature": signature, "timestamp": ts,
                                      "nonce": nonce}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print(f"[{label}] auth: action={auth.get('action')} "
                  f"code={auth.get('code')} rate_limit={auth.get('rate_limit')}")
            if auth.get("action") != "auth_success":
                print(f"[{label}] AUTH FAILED: {auth.get('message')}")
                return 2
            await ws.send(json.dumps({"action": "subscribe",
                                      "channels": channels}))
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), max(0.5, deadline - time.monotonic()))
                except asyncio.TimeoutError:
                    break
                try:
                    frame = json.loads(raw)
                except (ValueError, TypeError):
                    counts["<binary/unparsed>"] += 1
                    continue
                action = frame.get("action")
                if action in ("subscribed", "error", "ping", "pong"):
                    print(f"[{label}] control: {json.dumps(frame)[:160]}")
                    if action == "ping":
                        await ws.send(json.dumps({"action": "pong"}))
                    continue
                key = ((str(frame.get("T") or "") + ":"
                        + str(frame.get("symbol") or frame.get("s") or "?"))
                       if frame.get("T")
                       else (frame.get("channel") or action or "<data>"))
                counts[str(key)] += 1
                text = json.dumps(frame)
                for field in ("accountNo", "custodyCode", "investorId"):
                    value = frame.get(field)
                    if value:
                        text = text.replace(str(value), "<masked>")
                samples.setdefault(str(key), text[:240])
        except websockets.exceptions.ConnectionClosed as exc:
            # #92: the close IS the measurement — report, never traceback.
            elapsed = time.monotonic() - started
            closed_early = (f"SERVER CLOSED the connection at {_now()} "
                            f"(+{elapsed:.1f}s into the session): "
                            f"code={exc.rcvd.code if exc.rcvd else '?'} "
                            f"reason={(exc.rcvd.reason if exc.rcvd else '') or '<none>'}")
            print(f"[{label}] {closed_early}")

    print(f"\n[{label}] === capture ({seconds}s requested, "
          f"{time.monotonic() - started:.1f}s actual) ===")
    for key, n in counts.most_common():
        print(f"[{label}] {n:5d}  {key}")
    for key, sample in samples.items():
        print(f"\n[{label}] sample [{key}]: {sample}")
    if closed_early:
        print(f"[{label}] frames before the close: {sum(counts.values())} — "
              f"a close BEFORE account activity means the channel was never "
              f"tested; re-run and note WHAT ELSE holds a session "
              f"(engine WS? another probe?)")
        return 3
    if not counts:
        print(f"[{label}] ZERO data frames — silent WITH correct auth "
              f"(methodology sound; for order./position. channels this is "
              f"EXPECTED unless account events occurred during capture)")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--trading", action="store_true",
                    help="subscribe the TRADING channels (order/position "
                         "events) instead of market data — passive, read-only")
    ap.add_argument("--dual", action="store_true",
                    help="#92 single-session measurement: run TWO "
                         "authenticated sessions on the same channels, "
                         "second joining 5s late — the report shows which "
                         "one the server drops")
    args = ap.parse_args()

    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    channels = TRADING_CHANNELS if args.trading else CHANNELS

    if not args.dual:
        return await _capture(cfg, channels, args.seconds, "ws")

    async def _late_second():
        await asyncio.sleep(5)
        return await _capture(cfg, channels, args.seconds, "ws-B")

    rc_a, rc_b = await asyncio.gather(
        _capture(cfg, channels, args.seconds + 5, "ws-A"), _late_second())
    print(f"\n=== dual verdict: A rc={rc_a}, B rc={rc_b} — "
          f"3 marks the session the server dropped; two 0s = no "
          f"single-session limit observed ===")
    return max(rc_a, rc_b)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
