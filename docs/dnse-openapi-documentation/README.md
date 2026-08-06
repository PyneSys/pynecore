# DNSE OpenAPI v2 documentation (mirror)

Local markdown mirror of DNSE's developer docs, from
`https://developers.dnse.com.vn/docs/<slug>` (each page exposes a `.md` export
via its "Copy page" control). ~51 endpoint/guide/SDK pages.

**Keep it in sync:** run `python docs/dnse-openapi-documentation/fetch_docs.py`.
It discovers every `/docs/` page from the site sitemap (so newly published pages
appear automatically — no hardcoded list) and reports new/updated/unchanged.
Category-landing pages (no `.md` export) are auto-skipped. It also downloads the
authoritative **OpenAPI spec YAML** for each published API version from the DNSE
CDN — `openapi-spec-<version>.yaml` (currently `2026-05-07` and `2026-07-23`, the
latest). The CDN 404s any non-published date, so the version list is self-checking.

Base URL: `https://openapi.dnse.com.vn`. Auth = HMAC signing headers
(`X-API-Key`, `X-Signature`, a **date header**) + **`version: YYYY-MM-DD`**
(date-based; conditional STOP/OCO need `version >= 2026-07-23`); order **writes**
additionally need `trading-token`.

> **Why this mirror exists:** discovered mid-development that DNSE has **two order
> backends** (below), and the schemas are client-rendered on the doc site (not in
> the SSG HTML), so the endpoint details had to be mirrored to be usable offline.

## ⚠️ Two order backends — the key finding

| | Legacy | New (latest API) |
|---|---|---|
| Place path | `POST /accounts/orders` (accountNo in **body**) | `POST /accounts/{accountNo}/orders` (accountNo in **path**) |
| Used by | v1 plugin, both Python SDKs (`dnse`, `openapi-sdk`) | **neither SDK** — call directly |
| Order types | NORMAL only | NORMAL + **STOP** + **OCO** |
| STOP behaviour | **silently degraded to NORMAL** (proven live: `orderCategory=STOP` → readback shows `orderCategory=NORMAL`, `stopPrice` dropped) | real conditional order (`orderStatus=New`, resting) |
| Order ids | integer (`16`) | string (`d9q91e21a4skcecm6720`) |

**Consequence:** native STOP/OCO are reachable **only** via the account-scoped
new-backend endpoints, which neither SDK uses. Using them means driving the new
backend directly (schema below).

## Native conditional orders (derivatives) — schema

`POST /accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory={NORMAL|STOP|OCO}`

- **`orderCategory`** (query): `NORMAL` (base/deriv/bond), `STOP` (base+deriv),
  `OCO` (deriv only).
- **`orderType`**: NORMAL → LO / MOK,MAK,MTL / ATO,ATC / PLO; **STOP → LO
  (stop-limit) or MTL (stop-market)**; **OCO → LO**.
- **STOP required fields:** `stopPrice` + **`conditionOperator`** (`">="` trigger
  when market ≥ condition; `"<="` when ≤) + **`durationType="GTD"`** +
  **`durationDateTime`** (RFC3339). `price` = the limit price after trigger.
- **OCO:** a TP+SL bracket; `price` = take-profit price (deriv only).
- Verified live: a `GTD` buy stop (`stopPrice`, `conditionOperator=">="`) returns
  **201** and rests as `orderStatus="New"`.

## Cancel / read of a native STOP — a parallel "conditional order book"

`DELETE /accounts/{accountNo}/orders/{orderId}?marketType=…&orderCategory=…` is
what the doc lists (both query params required). **In practice (live 2026-08-06)
it does NOT work for a STOP order id:** GET-detail and DELETE-by-id both return
`500 REMOTE_SERVER_ERROR`, with either `X-Aux-Date` or `Date` (so the header is
*not* the cause — an earlier note here that claimed it was is wrong).

Why: a native STOP lands in DNSE's **separate conditional order book** (the user
saw the placed order there in the app). `GET /accounts/{accountNo}/orders` returns
only the **NORMAL** book and **silently ignores** the `orderCategory` filter
(NORMAL/STOP/OCO all return the same NORMAL rows), so the STOP id is unknown to
that service → 500 on detail/cancel. Net: native STOP is **place-only** through
the public gateway, and it IS the conditional orderbook — which this plugin
deliberately does not use. See `dnse-cancel-order.md` for the doc'd (NORMAL) shape.

## Index (files here)

- **Trading (writes):** `dnse-post-accounts-account-no-orders.md` (place — NORMAL/STOP/OCO),
  `dnse-replace-order.md` (PUT modify), `dnse-cancel-order.md` (DELETE),
  `dnse-post-positions-position-id-close.md`, `dnse-post-positions-position-id-pnl-configs.md`
  (position TP/SL config), `dnse-send-email-otp.md`, `dnse-2-fa-verification.md` (OTP→token).
- **Account/orders (reads):** `dnse-get-accounts.md`, `dnse-get-account-balances.md`,
  `dnse-get-loan-packages.md`, `dnse-get-ppse.md`, `dnse-get-orders.md`,
  `dnse-get-order-detail.md`, `dnse-get-orders-history.md`, `dnse-get-executions.md`.
- **Positions:** `dnse-get-positions.md`, `dnse-get-positions-position-id.md`,
  `dnse-get-positions-position-id-pnl-configs.md`, `dnse-get-corporate-action-history.md`.
- **Market data:** `dnse-get-instruments.md`, `dnse-get-ohlc-history.md`,
  `dnse-get-latest-quotes.md`, `dnse-get-latest-trades.md`, `dnse-get-quotes.md`,
  `dnse-get-history-trades.md`, `dnse-get-price-symbol-close.md`,
  `dnse-get-symbol-secdef.md`, `dnse-get-session.md`, `dnse-get-expected-price.md`,
  `dnse-get-foreign-trading.md`, `dnse-get-market-working-dates.md`.
- **Guides:** `changelog.md`, `guide-broker.md`, `guide-error_codes.md`,
  `guide-ratelimits.md`, `guide-faq.md`, `guide-enum-market_data.md`,
  `guide-intro-*` (api_platform / authentication / register_guide),
  `guide-market-data-*` (connect / broker_connect / trading_connect),
  `guide-trading-api-*` (trading_account / trading_order / dnse_margin),
  **`guide-versioning-api.md`** (date-based API versioning; latest published
  version = `2026-07-23`) + `guide-versioning-sdk.md`.
- **SDK (official = `github.com/dnse-tech/openapi-sdk`):** `sdk-trading.md`,
  `sdk-market_data.md`, `sdk-build_websocket.md`.

**Not mirrored (auto-skipped by `fetch_docs.py`):** the section-landing pages
(`dnse/account`, `dnse/trading`, `dnse/market-data`) are navigational category
indexes; `openapi-v-2-spec-260730` is a client-side spec-viewer shell — none have
a `.md` export, and the endpoint pages above carry the same schemas.
