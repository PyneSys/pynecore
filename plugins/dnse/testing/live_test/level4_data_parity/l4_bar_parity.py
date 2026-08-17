#!/usr/bin/env python3
"""Live-L4 — bar-data parity + latency (rubycell/pynecore#24). PASSIVE: no orders.

Records every CLOSED bar the plugin's real ``watch_ohlcv`` yields at @1/@3/@5
concurrently, then referees the same window against a fresh ``/price/ohlc`` fetch
(the venue's slightly-delayed authoritative data) and prints PASS/FAIL.

Mechanism (audited 2026-08-17): ``watch_ohlcv`` polls ``/price/ohlc`` at the REQUESTED
resolution and skips the forming bar — live 3m/5m bars are the venue's own rows, so
parity is expected EXACT; any diff is a finding (revision, forming-bar leak, or an
engine-side idle-synth bar that does not exist at the venue).

NOTE: one broker instance PER timeframe — ``_last_bar_ts`` is instance state and
sharing an instance across timeframes would cross-suppress bars.

  Live-L4-T01-BarParity : per-bar timestamp/O/H/L/C/V equality + 1m->3m/5m aggregation
                          cross-check (separates venue-side vs our-side divergence).
  Live-L4-T02-BarLatency: arrival_delay = wallclock - bar_close_time per bar
                          (median/p95/max per TF) + per-poll REST round-trip.
                          RED LINES: a bar arriving after the NEXT bar closed, or
                          delay drifting upward across the run.

    .venv/bin/python .../l4_bar_parity.py --minutes 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig       # noqa: E402

ICT = timezone(timedelta(hours=7))
SYMBOL = "VN30F1M"
TFS = ("1", "3", "5")
TF_SEC = {"1": 60, "3": 180, "5": 300}


def new_broker(tf: str) -> DNSEBroker:
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    return DNSEBroker(symbol=SYMBOL, timeframe=tf, config=cfg)


async def record(tf: str, until: float, out: list, rtts: list) -> None:
    """Capture every closed bar watch_ohlcv yields for ``tf`` until the deadline."""
    b = new_broker(tf)
    # instrument the REST round-trip without touching plugin code
    real = b.client.get_ohlc

    def timed(*a, **k):
        t0 = time.monotonic()
        r = real(*a, **k)
        rtts.append(time.monotonic() - t0)
        return r

    b.client.get_ohlc = timed                                       # type: ignore
    # watch_ohlcv returns ONE closed bar per await (it is a coroutine, not a generator)
    try:
        while time.time() < until:
            try:
                bar = await asyncio.wait_for(b.watch_ohlcv(SYMBOL, tf), timeout=15)
            except asyncio.TimeoutError:
                continue
            out.append({
                "ts": bar.timestamp // 1000, "o": bar.open, "h": bar.high,
                "l": bar.low, "c": bar.close, "v": bar.volume,
                "arrived": time.time(),
            })
            close_t = bar.timestamp / 1000 + TF_SEC[tf]
            print(f"  [@{tf}] bar {datetime.fromtimestamp(bar.timestamp/1000, ICT):%H:%M} "
                  f"O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume:.0f} "
                  f"delay={time.time()-close_t:+.1f}s", flush=True)
    finally:
        pass


def referee(tf: str, recorded: list) -> list[str]:
    """Re-fetch the window from /price/ohlc and diff each recorded bar. -> failures"""
    if not recorded:
        return [f"@{tf}: recorded no bars at all"]
    b = new_broker(tf)
    lo, hi = recorded[0]["ts"], recorded[-1]["ts"] + TF_SEC[tf]
    st, body = b.client.get_ohlc(b.market_type, {
        "symbol": SYMBOL, "resolution": b.to_exchange_timeframe(tf),
        "from": lo - TF_SEC[tf], "to": hi + TF_SEC[tf]})
    if st != 200 or not isinstance(body, dict) or not body.get("t"):
        return [f"@{tf}: referee fetch failed HTTP {st}"]
    venue = {int(t): (body["o"][i], body["h"][i], body["l"][i], body["c"][i], body["v"][i])
             for i, t in enumerate(body["t"])}
    fails = []
    for r in recorded:
        row = venue.get(r["ts"])
        if row is None:
            fails.append(f"@{tf} {datetime.fromtimestamp(r['ts'], ICT):%H:%M}: we emitted a "
                         f"bar the venue does not have (idle-synth or ghost)")
            continue
        for k, i in (("o", 0), ("h", 1), ("l", 2), ("c", 3), ("v", 4)):
            if abs(float(r[k]) - float(row[i])) > 1e-9:
                fails.append(f"@{tf} {datetime.fromtimestamp(r['ts'], ICT):%H:%M}: "
                             f"{k.upper()} ours={r[k]} venue={row[i]} (revision or "
                             f"forming-bar leak)")
    # gap check inside our own sequence (session-aware only trivially: consecutive
    # recorded bars should be TF_SEC apart unless a session boundary sits between)
    for a, c in zip(recorded, recorded[1:]):
        if c["ts"] - a["ts"] != TF_SEC[tf]:
            fails.append(f"@{tf}: gap {datetime.fromtimestamp(a['ts'], ICT):%H:%M} -> "
                         f"{datetime.fromtimestamp(c['ts'], ICT):%H:%M} "
                         f"(missed bar or session edge — verify manually)")
    return fails


def aggregate_check(rec1: list, recN: list, n: int, tf: str) -> list[str]:
    """Aggregate recorded 1m bars into n-minute bars and diff against recorded @tf."""
    fails = []
    ones = {r["ts"]: r for r in rec1}
    for r in recN:
        members = [ones.get(r["ts"] + 60 * i) for i in range(n)]
        got = [m for m in members if m]
        if len(got) != n:
            continue        # incomplete coverage of this window — skip, not a failure
        agg = {"o": got[0]["o"], "h": max(m["h"] for m in got),
               "l": min(m["l"] for m in got), "c": got[-1]["c"],
               "v": sum(m["v"] for m in got)}
        for k in ("o", "h", "l", "c", "v"):
            if abs(float(agg[k]) - float(r[k])) > 1e-9:
                fails.append(f"agg 1m x{n} vs @{tf} "
                             f"{datetime.fromtimestamp(r['ts'], ICT):%H:%M}: "
                             f"{k.upper()} agg={agg[k]} native={r[k]}")
    return fails


def latency_report(tf: str, recorded: list, rtts: list) -> tuple[list[str], dict]:
    """Latency stats EXCLUDING the first recorded bar: at startup watch_ohlcv returns
    the most recent ALREADY-CLOSED bar, which for @5 can be minutes old — that is
    backfill, not feed latency, and it polluted max/p95 in the first smoke run."""
    fails, delays = [], []
    for r in recorded[1:]:
        d = r["arrived"] - (r["ts"] + TF_SEC[tf])
        delays.append(d)
        if d > TF_SEC[tf]:
            fails.append(f"@{tf} {datetime.fromtimestamp(r['ts'], ICT):%H:%M}: arrived "
                         f"{d:.1f}s after close — AFTER the next bar closed (red line)")
    stats = {}
    if delays:
        stats = {"n": len(delays), "median": round(statistics.median(delays), 2),
                 "max": round(max(delays), 2),
                 "p95": round(sorted(delays)[max(0, int(len(delays)*0.95) - 1)], 2)}
        if len(delays) >= 4:
            half = len(delays) // 2
            drift = statistics.median(delays[half:]) - statistics.median(delays[:half])
            stats["drift"] = round(drift, 2)
            if drift > 2.0:
                fails.append(f"@{tf}: delay drifting up ({drift:+.1f}s late-half vs "
                             f"early-half) — loop falling behind")
    if rtts:
        stats["rest_rtt_median"] = round(statistics.median(rtts), 3)
        stats["rest_rtt_max"] = round(max(rtts), 3)
    return fails, stats


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    args = ap.parse_args()

    until = time.time() + args.minutes * 60
    print(f"=== Live-L4 bar parity+latency @ {datetime.now(ICT):%Y-%m-%d %H:%M:%S} "
          f"— recording @1/@3/@5 for {args.minutes:g} min (passive, no orders) ===")
    rec = {tf: [] for tf in TFS}
    rtt = {tf: [] for tf in TFS}
    await asyncio.gather(*(record(tf, until, rec[tf], rtt[tf]) for tf in TFS))

    print("\n--- referee pass (fresh /price/ohlc fetch) ---")
    fails: list[str] = []
    stats: dict = {}
    for tf in TFS:
        fails += referee(tf, rec[tf])
        f, s = latency_report(tf, rec[tf], rtt[tf])
        fails += f
        stats[tf] = s
        print(f"  @{tf}: bars={len(rec[tf])} latency={s}")
    fails += aggregate_check(rec["1"], rec["3"], 3, "3")
    fails += aggregate_check(rec["1"], rec["5"], 5, "5")

    ev = Path(__file__).parent / "logs"
    ev.mkdir(exist_ok=True)
    out = ev / f"l4_{datetime.now(ICT):%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"recorded": rec, "latency": stats, "failures": fails},
                              indent=1))
    print(f"evidence: {out}")

    print("\n=== RESULT ===")
    for f in fails:
        print(f"  [FAIL] {f}")
    print("VERDICT:", "FAIL" if fails else
          "PASS — live bars match the venue's delayed OHLC; latency within red lines")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
