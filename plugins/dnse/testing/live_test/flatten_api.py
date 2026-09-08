"""API flatten for the FILL tier — thin CLI over ``tools/flatten.py`` (#91).

Close-first, then sweep the bot's own protection (journal-rooted, read-only
attribution) so a flat account is never left with an armed entry-stop
(measured 2026-09-08, operator-caught). Foreign orders are reported, never
cancelled — the netting account is shared. No app involvement, so the #51
window never opens (operator decision 2026-09-07).

Exit codes (venue.py convention): 0 flat AND owned orders swept/resolved,
1 not flat / sweep unresolved, 2 could not determine.

Usage: .venv/bin/python plugins/dnse/testing/live_test/flatten_api.py [--dry-run]
"""
import sys
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402
from flatten import flatten, owned_live_ids                          # noqa: E402

STORE = REPO / "workdir" / "output" / "logs" / "broker.sqlite"


def main() -> int:
    dry = "--dry-run" in sys.argv
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir/config/plugins/dnse_broker.toml")
    broker = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    symbol = broker.symbol or "VN30F1M"
    owned = owned_live_ids(STORE)
    if owned is None:
        print(f"attribution UNAVAILABLE (store: {STORE})")
    else:
        print(f"attribution: {len(owned)} journalled live id(s)")
    if dry:
        print("dry-run: no orders sent, no cancels")
        return 0
    return flatten(broker, symbol, owned)


if __name__ == "__main__":
    sys.exit(main())
