# Binance broker plugin — implementation plan

Goal: `pyne run <script>.py binance:BTC/USDT@<tf> --broker` places real orders on
Binance. Today the ccxt plugin is data-only (`--live` streaming already works,
verified 2026-08-17 against BTC/USDT@1); the order path needs a `BrokerPlugin`.

## Scope decision: SPOT first

The funded account is a Binance **spot** wallet (~495 USDT). Consequences:

- Long-only: `short_selling = UNSUPPORTED`. Scripts that may go short are
  rejected at startup by `validate_at_startup` — that is correct behaviour.
- No venue position object → the plugin opts into the core spot inventory
  layer (`spot_inventory_port = self`); core owns the fill ledger, the
  balance invariant and `get_position` synthesis (`synthesize_position`).
- Reference implementation for the port surface:
  `../pynecore-plugin-bybit/src/pynecore_bybit/inventory.py`.

USD-M futures (shorts, leverage, native positions — a much simpler netting
venue) is a later phase and would need a wallet transfer; out of scope here.

## Placement and shape

```
plugins/binance/
  pyproject.toml            # dist opencode-pynecore-binance, editable install
  pynecore_binance/
    __init__.py
    config.py               # BinanceConfig(LiveProviderConfig): apiKey, secret, sandbox
    provider.py             # BinanceProvider(CCXTProvider) pinned to exchange "binance"
    broker.py               # BinanceBroker(BinanceProvider, BrokerPlugin)
    execution.py            # execute_entry/exit/close/cancel — ccxt order calls
    inventory.py            # SpotInventoryPort: fetch_executions / fetch_base_balance
    reconcile.py            # per-bar snapshot poll + DisappearanceTracker
    errors.py               # ccxt exception -> broker taxonomy mapping
  tests/                    # pytest, __test_*__ naming, fake-ccxt seam in conftest.py
```

Entry points (`pyne.plugin` group), mirroring the dnse pair:

```toml
[project.entry-points."pyne.plugin"]
binance = "pynecore_binance.provider:BinanceProvider"
binance_broker = "pynecore_binance.broker:BinanceBroker"
```

Transport is **ccxt/ccxt.pro** (already installed, 4.5.71): signing, endpoints,
rate-limit bookkeeping and the sandbox switch come free; the plugin's job is
purely the intent → order translation and the event mapping. Config continues
to be read from `workdir/config/plugins/` (`binance.toml`, self-healing like
`ccxt.toml`) — never from `.env`, never echoed to stdout.

## Intent → Binance mapping (execution.py)

| Engine call | Binance spot call (via ccxt) |
|---|---|
| entry MARKET | `create_order(sym,'market',side,qty)` |
| entry LIMIT | `create_order(sym,'limit',side,qty,price)` |
| entry STOP (buy-stop above / sell-stop below) | spot has no pure stop-market on all pairs → `STOP_LOSS_LIMIT` with a marketable limit price (band = last±slippage guard), params `{'stopPrice':…}` |
| exit TP+SL bracket | **native spot OCO** — `privatePostOrderOco` / ccxt `createOrder` with `params={'stopPrice','stopLimitPrice'}`; one leg fills, venue cancels the sibling (`tp_sl_bracket = NATIVE`, `oca_cancel = NATIVE`) |
| exit TP-only / SL-only | single LIMIT / STOP_LOSS_LIMIT sell |
| close | market sell of tracked net inventory |
| cancel | `cancel_order` (by our `newClientOrderId`); OCO cancel via `cancelOrderList` |
| cancel_all | `DELETE /api/v3/openOrders` — call `native_cancel_all_expected_sink` FIRST |
| modify | Binance spot has no amend → keep default cancel+replace (`amend_order = SOFTWARE`) |

Idempotency: `client_order_id_max_len = 36`; pass
`envelope.client_order_id(KIND_*)` as `newClientOrderId` — Binance echoes it
and rejects duplicates (`idempotency = NATIVE`). Pre-flight every order against
`LOT_SIZE` / `PRICE_FILTER` / `MIN_NOTIONAL` from `load_markets()`; below-min →
raise `OrderSkippedByPlugin` (non-halting), never send.

Error mapping (`errors.py`): ccxt `AuthenticationError`→`AuthenticationError`,
`InsufficientFunds`→`InsufficientMarginError`, `InvalidOrder`/`OrderNotFound`
per-context (`ExchangeOrderRejectedError` / benign no-op on cancel),
`RateLimitExceeded`→`ExchangeRateLimitError`, `NetworkError`→
`ExchangeConnectionError`, ambiguous send-timeout on a write →
`OrderDispositionUnknownError(client_order_id=…)`.

## Events + inventory

- `watch_orders`: start with **SOFTWARE polling** (dnse pattern — proven in
  this repo): poll open orders + `myTrades` each cycle, synthesize
  `OrderEvent`s. Pine identity recovered by parsing our deterministic
  `newClientOrderId`. `fill_id = str(tradeId)` everywhere (dup gate).
  `fill_qty` is the incremental slice; `order.filled_qty` cumulative.
  Upgrade path: ccxt.pro `watch_orders`/`watch_my_trades` later.
- `fetch_executions(cursor)`: `myTrades(symbol, fromId=cursor)`, per-product
  cursor (`cursor_scope='product'`); `cursor=None` → empty batch anchored at
  the current watermark (prior history is foreign). Decimal-only fields via
  `Decimal(str(x))`; BNB-fee fills touch neither delta (fee-currency rules).
- `fetch_base_balance`: free+locked BTC from `fetch_balance`.
- Disappearance: per-poll snapshot of open orders vs mapping with a grace
  window (`DisappearanceTracker`), quarantine via `quarantine_sink`.

## Capabilities

stop_order SOFTWARE (stop-limit emulation) · tp_sl_bracket NATIVE (OCO) ·
oca_cancel NATIVE · cancel_all NATIVE · idempotency NATIVE · amend SOFTWARE ·
watch_orders SOFTWARE (poll) · fetch_position SOFTWARE (synthesized) ·
reduce_only SOFTWARE (inventory-capped sells) · trailing_stop UNSUPPORTED
(phase 2 — no native spot trailing) · short_selling UNSUPPORTED.

## Test ladder (same discipline as dnse)

1. **pytest** — fake-ccxt client seam in `conftest.py`; unit tests per module +
   fake-venue e2e; `validate_plugin_contract` green in CI from day one.
2. **Backtest oracle** — run the probe strategies over downloaded
   `ccxt_BINANCE_BTC_USDT_*` data; live results are graded against this.
3. **L0-style venue gate** (mandatory, exit 0, before every live run):
   auth OK, clock skew < recvWindow, symbol filters readable, open-order
   count 0 for the run tag, balance ≥ planned notional.
4. **Staged no-fill probe** — far-from-market LIMIT (buy 30% below market,
   min-notional ~10 USDT), verify ack/open/cancel round-trip from the VENUE
   record (`fetch_order`), never the run log alone. 24/7 venue: no session
   phases to dodge, but probes still start with `winStart` after launch.
5. **Staged fill test** — smallest lot market buy → OCO bracket → TP/SL fill,
   graded against the backtest oracle.

Live probes spend real money (min-notional ~10 USDT per fill probe) and are
run only after explicit user go-ahead per stage; sandbox mode
(`sandbox = true`, Binance spot testnet keys) is wired from the start for
free rehearsal.

## Phases

- **P1** scaffold + config + provider subclass + contract validation green.
- **P2** execution.py + errors.py against the fake venue (pytest e2e).
- **P3** inventory port + polling events; full fake-venue lifecycle test.
- **P4** L0 gate + staged no-fill probe live (cancel-only, no fills).
- **P5** staged fill test (OCO bracket) vs backtest oracle; then park.

Prereq before any live stage: **rotate the API key** (it was exposed in
terminal scrollback on 2026-08-17), restrict it to Read + Spot Trading, no
withdrawals, IP-whitelisted.
