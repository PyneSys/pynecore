"""API flatten for the FILL tier (operator decision 2026-09-07).

Closes the current net position through the plugin's own ``execute_close``
path (marketable band-edge LO) using the bot's API token — NO app
involvement, so the #51 window (app trades poison conditional-book writes)
never opens during the F-ladder. Verifies FLAT afterward.

Exit codes (venue.py convention): 0 flat (flattened or already flat),
1 flatten failed / still holding, 2 could not determine.

Usage: .venv/bin/python plugins/dnse/testing/live_test/flatten_api.py [--dry-run]
"""
import asyncio
import sys
import time
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore.core.broker.models import (                            # noqa: E402
    CloseIntent, DispatchEnvelope,
)
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402


def main() -> int:
    dry = "--dry-run" in sys.argv
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir/config/plugins/dnse_broker.toml")
    b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    symbol = b.symbol or "VN30F1M"
    try:
        pos = asyncio.run(b.get_position(symbol))
    except Exception as exc:                                         # noqa: BLE001
        print(f"COULD NOT READ position: {type(exc).__name__}: {exc}")
        return 2
    if pos is None or float(pos.size or 0) == 0.0:
        print("FLAT already — nothing to do")
        return 0
    size = abs(float(pos.size))
    close_side = "sell" if (pos.side or "").lower() == "long" else "buy"
    print(f"position: {pos.side} {size} -> closing via execute_close "
          f"({close_side} {int(size)} @ band edge)")
    if dry:
        print("dry-run: no order sent")
        return 0
    envelope = DispatchEnvelope(
        intent=CloseIntent(pine_id="F-FLATTEN", symbol=symbol,
                           side=close_side, qty=size, immediately=True),
        run_tag="flat", bar_ts_ms=int(time.time() * 1000),
        retry_seq=0, coid_max_len=30)
    order = asyncio.run(b.execute_close(envelope))
    print(f"close order placed: id={order.id} {order.side} {order.qty}")
    deadline = time.time() + 25
    while time.time() < deadline:
        time.sleep(2)
        try:
            pos = asyncio.run(b.get_position(symbol))
        except Exception:                                            # noqa: BLE001
            continue
        if pos is None or float(pos.size or 0) == 0.0:
            print("FLAT confirmed")
            return 0
    print("NOT FLAT after 25s — check venue.py status / the app NOW")
    return 1


if __name__ == "__main__":
    sys.exit(main())
