#!/usr/bin/env python3
"""#37 baseline live-half — READ-ONLY probe of ``/price/{contract}/trades/latest``.

Measures what the panel left UNVERIFIED before tick-mode constants freeze:
response envelope (list vs dict wrapper), ARRAY DEPTH (the doc example shows 1
element — if the venue only ever returns the single newest print, a 2 s poll
under-samples bursts and the cursor-dedup design carries the whole load),
``boardId`` values on real prints (put-through filtering seam), and the
monotonicity of ``totalVolumeTraded`` across polls.

Usage (any session phase; zero orders, zero writes):
    .venv/bin/python plugins/dnse/testing/live_test/probe_trades_latest.py [--polls N] [--interval S]

Pre-open it snapshots yesterday's tail prints (schema + boardId ground truth);
during the open burst (~09:00-09:05) it measures depth and print density.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from venue import broker  # noqa: E402  (the toolkit's constructor — no hand-rolled client)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polls", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    b = broker("VN30F1M")
    contract = b.resolve_contract()
    print(f"contract={contract} polls={args.polls} interval={args.interval}s")

    depths, boards, totals = [], Counter(), []
    envelope = None
    for i in range(args.polls):
        status, body = b.client.get_latest_trade(contract)
        if status != 200:
            print(f"poll {i}: http={status} body={str(body)[:120]}")
            time.sleep(args.interval)
            continue
        rows = (body if isinstance(body, list)
                else (body.get("trades") or body.get("data") or [])
                if isinstance(body, dict) else [])
        if envelope is None:
            envelope = ("list" if isinstance(body, list)
                        else f"dict keys={sorted(body.keys())[:6]}" if isinstance(body, dict)
                        else type(body).__name__)
            if rows:
                print(f"poll 0 first-row fields: {sorted(rows[0].keys())}")
                print(f"poll 0 first-row sample: {json.dumps(rows[0], default=str)[:240]}")
        depths.append(len(rows))
        for r in rows:
            boards[str(r.get("boardId"))] += 1
            t = r.get("totalVolumeTraded")
            if t is not None:
                totals.append(float(t))
        time.sleep(args.interval)

    print(f"envelope: {envelope}")
    print(f"array depth per poll: min={min(depths) if depths else '-'} "
          f"max={max(depths) if depths else '-'} all={depths}")
    print(f"boardId values seen: {dict(boards)}")
    monotone = all(a <= b_ for a, b_ in zip(totals, totals[1:]))
    print(f"totalVolumeTraded: n={len(totals)} monotone-nondecreasing={monotone} "
          f"first={totals[0] if totals else '-'} last={totals[-1] if totals else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
