# DNSE Broker Plugin v2 — native conditional orders on openapi-sdk

## Overview

v2 rebuilds the DNSE plugin on DNSE's **native conditional orders** (server-side
STOP/OCO) using the official **openapi-sdk 2.0.0**. This replaces v1's plugin-owned
"software-watch" design (and the `dnse-py` SDK), which only reached the legacy
`/accounts/orders` endpoint.

Native conditional orders are **server-side** (fire even if the plugin is offline —
strictly safer for a live stop-loss), **fully API-manageable** (place / list /
cancel), and far simpler than a client-side watch: no armed-leg engine, no intrabar
teardown, no persistence, no WS intrabar trigger. **v2 is REST-only.** All proven
live 2026-08-06.

**Root unlock:** the conditional endpoints are gated behind API **`version >=
2026-07-23`** — sending an older/absent `version` (v1 sent `2026-01-01`/`2026-05-07`)
returns `500` or silently the NORMAL book, which is what made native STOP look
"broken". The vendored SDK defaults to `2026-07-23`.

## SDK

Official **openapi-sdk 2.0.0** vendored at `plugins/dnse/pynecore_dnse/_vendor/dnse`
(tag `v2.0.0` / `1532e33`), imported via `_sdk.py`. Its `DNSEClient` is a single,
complete client (trading **and** market-data) with a `(status, body)` return
contract:

- **Trading:** `post_order` / `put_order` / `cancel_order` (account-scoped,
  `order_category` ∈ NORMAL/STOP/OCO, `version`), `get_orders` (paginated),
  `get_order_detail`, `get_positions`, `get_position_by_id`, `create_trading_token`,
  `send_email_otp`.
- **Market data:** `get_ohlc`, `get_security_definition`, `get_instruments`,
  `get_quotes`/`get_trades`/`get_latest_*`.

Defaults to `version=2026-07-23`; string `orderId`s. REST client is stdlib-only; the
WS `TradingClient` (deps `msgpack`/`websockets`/`certifi`) exists but v2 does not use
it. Vendored, not pip-installed. Docs mirrored at `docs/dnse-openapi-documentation/`
(kept current + version-drift-checked by `fetch_docs.py`).

## Native conditional orders — schema & lifecycle

`POST /accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory={NORMAL|STOP|OCO}`

- **STOP** (base + deriv): `orderType` LO (stop-limit) or MTL (stop-market);
  `stopPrice` + `conditionOperator` (`>=` / `<=`) + `durationType=GTD` +
  `durationDateTime` (RFC3339); `price` = the limit price after trigger.
- **OCO** (deriv only): ONE managed order, not two legs. `price`=TP, `stopPrice`=SL
  trigger, `stopOrderPrice`=SL limit, `durationType=DAY`. On activation the venue
  places a **working NORMAL LO** at the TP price and, if the SL condition hits first,
  **amends that same LO to the SL price** (only one LO on the exchange at a time).
  **The link is `externalOrderId`**: `get_order_detail(oco_id, orderCategory=OCO)`
  returns `externalOrderId` = the working LO id (the LIST omits it), and the LO's
  `metadata.conditionOrderId` points back. So the plugin **tracks the working LO**
  (fills + cancels route by it — cancel the LO, not the umbrella); the `Activated`
  umbrella record is cosmetic and auto-expires EOD.
- **Status lifecycle:** `New` (armed, cancellable) → `Activated` (working / triggered)
  → `Cancelled`/`Expired`/`Rejected`/`Failed`. The OCO umbrella is only cancellable
  while `New`; once `Activated`, manage its working LO (via `externalOrderId`).
- **Cancels are refused during the ATO/ATC auctions**
  (`CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION`) — retry after the session opens.
- **List:** `GET /accounts/{accountNo}/orders?orderCategory=STOP|OCO` (+ `pageIndex`/
  `pageSize`) — a dedicated conditional book. GET-by-id is NORMAL-only.
- **Cancel:** `DELETE /accounts/{accountNo}/orders/{orderId}?marketType&orderCategory`.

## Pine intent → native order mapping

Pine semantics (confirmed): `entry(limit, stop)` = **one stop-limit** order;
`exit(limit, stop)` = an **OCA bracket** (two legs). These map cleanly onto native
STOP and OCO respectively:

| Pine construct | v2 native order |
|---|---|
| `entry(stop=S)` | STOP (`conditionOperator` by side; MTL, or LO at a marketable price) |
| `entry(limit=L)` | NORMAL LO |
| `entry(limit=L, stop=S)` (stop-limit) | STOP, `orderType=LO`, `stopPrice=S`, `price=L` |
| `exit(stop=SL)` | STOP, reduce-side |
| `exit(limit=TP)` | NORMAL LO, reduce-side |
| `exit(limit=TP, stop=SL)` (OCA) | **OCO** (native bracket) |
| `close` / market | marketable LO (band-edge price) |

Caveat: OCO is **deriv-only**. For a stock `exit(limit, stop)`, fall back to a NORMAL
TP LO + a STOP SL and manage the one-cancels-other in the plugin. Target symbol
VN30F1M is a derivative, so OCO applies.

## Build plan (task #4)

1. **Provider** (`provider.py`) — thin over `DNSEClient`: `resolve_contract`
   (`get_instruments`, symbolType→dated KRX contract), `watch_ohlcv` (poll
   `get_ohlc` for closed bars → `lib.close`), `_secdef` (`get_security_definition`,
   ceiling/floor), `market_type`. Port the reusable v1 logic (parked in
   `backup/deleteable/`) onto the new client API.
2. **Broker execution** — the intent→order mapping above: `post_order` with
   `order_category` STOP/OCO/NORMAL; `put_order` to modify while `New`; `cancel_order`
   for `strategy.cancel` / OCA. Market intents priced at the daily band edge
   (ceiling for buy, floor for sell).
3. **Broker state** — poll the conditional book (`New`→`Activated`) + the NORMAL book
   (fills) via `get_orders`; **position self-tracked from the fill ledger** (persisted;
   `/positions` read-only for reconciliation, never as the source of truth — see
   Non-goals); `get_capabilities` declares `stop_order` / `tp_sl_bracket` /
   `oca_cancel` = **NATIVE**, `reduce_only`, `short_selling` (deriv), `idempotency`
   SOFTWARE (no client order id in the place payload).
4. **Trading-token lifecycle** — daily OTP refresh (below).
5. **Verify** — contract probe + integration test + backtest oracle + live smoke.

## Trading-token lifecycle (daily OTP refresh) — prerequisite for the order path

The `trading_token` is short-lived + self-invalidating; **placement/cancel/replace
need a valid one, reads do not**. Path: **email OTP read from Gmail** (no TOTP seed).

**Architecture — a separate cron minter + a shared token file; the plugin is a pure
consumer.** Keeps the fragile Gmail/OTP dance out of the broker loop and the tests
(the plugin only ever reads a token string). Same producer/consumer split as
`pyne data download` → `.ohlcv` → `pyne run`.

- **Two legs, ONE source of truth.** Both the automatic (cron) and manual legs mint
  via the **same script**, which atomically writes the single
  `workdir/state/dnse_trading_token.json` = `{"trading_token": "...", "minted_at":
  <unix>}` (temp + `os.replace`). The plugin reads only that file; a manual refresh
  simply overwrites the cron's file. `config.trading_token` is a bootstrap fallback
  used only when the state file is absent. Never mutate `dnse_broker.toml`.
- **Minter** (`plugins/dnse/tools/refresh_token.py`, reuses the plugin's `DNSEClient`):
  - **auto (cron):** `send_email_otp()` → poll Gmail for the NEWEST DNSE OTP that
    arrived *after* the send timestamp (delivery lags ~30–120s; a prior code is dead)
    → `create_trading_token("email_otp", code)` → atomic write. Retry-send once + alert
    if none arrives. Gmail creds live only with the minter.
  - **manual (operator):** `refresh_token.py --otp <code>` — skips the Gmail scrape
    (optionally `--send` first). The "something went wrong" leg.
- **Schedule:** ONE daily cron at **08:00 ICT**; token **TTL = 8h** (~16:00) covers the
  full session (09:00–11:30 + 13:00–14:45). Manual leg covers a failed morning cron.
- **Expiry coordination:** on a token-expired reject, the plugin **re-reads the state
  file** (cron may have refreshed since startup); still stale → loud operator error /
  quarantine. The plugin **never mints**.
- **Security:** the token file is order-placement authority — perms + gitignored under
  `workdir/state/`, treated like `api_secret`; never logged.

## Verification

1. **Contract probe (step-0 gate):** `validate_plugin_contract()` on the v2 plugin →
   zero findings, in a proper `plugins/dnse/tests/` package.
2. **Native-conditional integration test** against an injected fake `DNSEClient`
   (recorded `(status, body)`): `entry(stop)` → a STOP placed with the correct schema;
   `exit(limit, stop)` → an OCO whose working LO is resolved via `externalOrderId` and
   tracked; a fill on that LO routed to the engine by Pine identity; cancel routed to
   the working LO; the `Activated` umbrella not reported as a phantom open order.
3. **Backtest as oracle:** `t3_long_stop` / `t4_short_stop` / `t5_long_stop_limit` /
   `t6_short_stop_limit` (`plugins/dnse/testing/strategies/`) through the backtest vs
   the fake broker; compare fills & positions. Note: native triggers **intrabar on
   match price** (vs the backtest's bar high/low) — parity holds on *trigger timing*;
   fill price is market-after-trigger, not the exact stop level.
4. **Live smoke (session-gated):** far-from-market STOP + OCO place→cancel on VN30F1M
   (proven 2026-08-06 — place 201/`New`, cancel 200 with `version=2026-07-23`); confirm
   no unexpected exposure.

## Non-goals

- **No pynecore core (`sync_engine.py`) changes.**
- **No software watch / WS intrabar trigger.** Native conditional orders are
  server-side; the WS `TradingClient` and market-data streams are not needed. Bars come
  from REST `/price/ohlc`.
- **No `/deals` endpoint** (undocumented) — fills come from the `/orders` cumulative
  `fillQuantity` diff (`fill_id = order_id:cumulative`). `/positions` is used
  **read-only for reconciliation only**; the position of record is self-tracked from
  the bot's fill ledger, so carry needs no venue adoption (first run with no ledger =
  flat).
- **No SDK from pip** — openapi-sdk is vendored from source.

## Key DNSE facts (reference)

- **API version: pin the EXACT published `2026-07-23`** (unlocks conditional endpoints; string
  `orderId`; `get_orders` requires `pageIndex`/`pageSize`). The SDK defaults to it — **use the default,
  never override.** ⚠️ A floating future date like `2026-08-06` (not a published version) **silently
  breaks the OCO cancel** (DELETE returns `200 "Canceled"` but the order sticks at `Activated`) while
  reads + STOP-cancel still work — proven live 2026-08-07.
- **Two-level symbols:** `VN30F1M` (symbolType) → dated KRX contract (e.g. `41I1G8000`)
  via `get_instruments`. Orders + streams use the dated contract; `/price/ohlc` accepts
  the alias.
- **Session-gated placement** (09:00–11:30 / 13:00–14:45 ICT); conditional orders rest
  as `New`; reads need no token.
- **Derivatives are netted per symbol** → `position_port = None` (no hedged-leg
  emulation).
- The conditional order book is a **separate subsystem** from the NORMAL book — an OCO
  spawns a NORMAL TP leg (cancellable via the NORMAL DELETE) + a conditional SL leg.

## Documentation & sync

- **Mirror:** `docs/dnse-openapi-documentation/` — ~51 markdown pages (every DNSE endpoint + the
  guides + changelog + SDK docs), the machine-readable OpenAPI spec YAMLs per published version
  (`openapi-spec-2026-07-23.yaml`, `openapi-spec-2026-05-07.yaml`), and a `README.md` index. The doc
  schemas are client-rendered on the site, so the mirror is the usable-offline source of truth.
- **Sync tool:** `docs/dnse-openapi-documentation/fetch_docs.py` — re-runnable, stdlib-only:
  - discovers every `/docs/` page from the **sitemap** `https://developers.dnse.com.vn/sitemap.xml`
    (new pages appear automatically — no hardcoded list) and fetches each page's `.md` export from
    `developers.dnse.com.vn` (category-landing pages have no `.md` and are auto-skipped);
  - downloads the per-version OpenAPI spec YAMLs from the CDN
    `https://cdn.entrade.com.vn/dnse-openapi/doc/dnse-openapi-<version>.yaml` (the cache-proof
    authority for which versions are actually published);
  - **verify step:** cross-checks the vendored SDK's `DEFAULT_API_VERSION` against the latest
    published version (from the versioning guide) and flags drift → re-vendor + review the changelog.
- **Key upstream references** (all mirrored locally): changelog
  `developers.dnse.com.vn/docs/changelog`, versioning guide `.../docs/guide/versioning/api`, place
  order `.../docs/dnse/post-accounts-account-no-orders`, cancel `.../docs/dnse/cancel-order`.
