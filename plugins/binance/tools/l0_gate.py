"""L0 venue gate for the Binance broker — MANDATORY before every live run.

Mirrors the DNSE plugin's L0 discipline: a read-only pre-flight that must
exit 0 before any ``pyne run … --broker`` launch. Checks, in order:

1. Config sane: ``binance_broker.toml`` exists; sandbox=true (or explicit
   allow_mainnet), credentials present.
2. Auth: ``GET /api/v3/account`` succeeds with the configured keys.
3. Clock: local vs venue skew below Binance's default recvWindow (5000 ms).
4. Symbol: market exists; LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL readable.
5. Book: count of open orders on the symbol (reported; nonzero is a warning,
   not a failure — a resumed run may own them).
6. Balance: quote-asset balance covers at least one MIN_NOTIONAL probe.

Usage: ``.venv/bin/python plugins/binance/tools/l0_gate.py [SYMBOL]``
(default ``BTC/USDT``). Never prints credentials.
"""
from __future__ import annotations

import sys
import time
import tomllib
from pathlib import Path

CONFIG_PATH = Path('workdir/config/plugins/binance_broker.toml')


def fail(message: str) -> None:
    print(f"L0 FAIL: {message}")
    sys.exit(1)


def main() -> None:
    import ccxt

    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'

    if not CONFIG_PATH.exists():
        fail(f"{CONFIG_PATH} missing")
    with open(CONFIG_PATH, 'rb') as config_file:
        config = tomllib.load(config_file)
    sandbox = bool(config.get('sandbox'))
    if not sandbox and not config.get('allow_mainnet'):
        fail("sandbox=false without allow_mainnet=true")
    if not config.get('apiKey') or not config.get('secret'):
        fail("apiKey/secret missing in binance_broker.toml")
    print(f"L0 [1/6] config OK (mode={'testnet' if sandbox else 'MAINNET'})")

    client = ccxt.binance({'apiKey': config['apiKey'],
                           'secret': config['secret'],
                           'enableRateLimit': True})
    if sandbox:
        client.set_sandbox_mode(True)

    try:
        balance = client.fetch_balance()
    except Exception as exc:                                        # noqa: BLE001
        fail(f"auth probe failed: {type(exc).__name__}: {exc}")
    print("L0 [2/6] auth OK")

    venue_ms = client.fetch_time()
    skew_ms = abs(venue_ms - int(time.time() * 1000))
    if skew_ms > 4000:
        fail(f"clock skew {skew_ms} ms vs recvWindow 5000 ms")
    print(f"L0 [3/6] clock skew {skew_ms} ms OK")

    markets = client.load_markets()
    if symbol not in markets:
        fail(f"symbol {symbol!r} not on the venue")
    market = markets[symbol]
    limits = market.get('limits', {})
    min_amount = (limits.get('amount', {}) or {}).get('min')
    min_cost = (limits.get('cost', {}) or {}).get('min')
    tick = market.get('precision', {}).get('price')
    if not min_amount or not tick:
        fail(f"filters unreadable: LOT_SIZE={min_amount} tick={tick}")
    print(f"L0 [4/6] {symbol}: LOT_SIZE min={min_amount} tick={tick} "
          f"MIN_NOTIONAL={min_cost}")

    open_orders = client.fetch_open_orders(symbol)
    print(f"L0 [5/6] open orders on {symbol}: {len(open_orders)}"
          + (" (WARNING: nonzero — a resumed run may own them)"
             if open_orders else ""))

    quote = market['quote']
    quote_free = float((balance.get('free') or {}).get(quote) or 0)
    need = float(min_cost or 10.0) * 2
    if quote_free < need:
        fail(f"{quote} free balance {quote_free:.2f} below {need:.2f} "
             f"(2x MIN_NOTIONAL probe budget)")
    print(f"L0 [6/6] {quote} free={quote_free:.2f} OK")

    print("L0 PASS")
    sys.exit(0)


if __name__ == '__main__':
    main()
