# DNSE broker — error-code handling plan

**Goal:** catch every DNSE error (documented + observed-live) and map it to one
deliberate action, so nothing collapses into today's blanket
`ExchangeOrderRejectedError` and no transient fault silently orphans or halts.

## 1. How DNSE reports errors

Errors come as **HTTP status (header) + `code` (body)**, sometimes with `status` /
`message`. Rule: **classify on `code` first, fall back to HTTP status.** Our client's
`status == 0` is synthetic = *no response reached* (network/timeout) → disposition
unknown. The OpenAPI spec enumerates **no** per-endpoint error schemas, and the
conditional-order `CO-ORD-*` family is undocumented — so the code table below is the
source of truth, seeded from `guide-error_codes.md` + codes observed live.

## 2. Engine contract — what raising each exception *does*

The engine already does retry / park / halt; the plugin only has to raise the **right**
exception. Misclassification is the whole risk: a permanent fault marked retryable
loops forever; a transient one marked permanent halts a healthy bot. Asymmetric by
design — **reads park (idempotent), writes halt/verify.**

| Plugin raises | Engine reaction | Use for |
|---|---|---|
| `ExchangeConnectionError` | reconnect, **indefinite** exp backoff; the read parks | recoverable read / connectivity fault |
| `OrderDispositionUnknownError` | **park the dispatch + verify** (never blind-retry) | ambiguous WRITE (no accept/reject) |
| `ExchangeRateLimitError(retry_after)` | wait `retry_after`, then retry | 429 / OA-429 |
| `ExchangeOrderRejectedError` | terminal reject of that order | definitive business/input reject |
| `InsufficientMarginError` (⊂ rejected) | terminal reject | buying-power / margin |
| `AuthenticationError` | stop — creds bad | auth a retry can't fix |
| `OrderSkippedByPlugin` | no-op, nothing sent | plugin proactively declines |
| `BrokerManualInterventionError` / `UnexpectedCancelError` | halt / quarantine | unsafe to continue |

## 3. Action classes (code → exception + any plugin-local step)

### A — Transient: let the engine wait it out
`HTTP 500/503`, `OA-500`, `OA-503`, `SYSTEM_ERROR`, `REMOTE_SERVER_ERROR`,
`THIRD_PARTY_ERROR`, `TIMEOUT`, `BATCH_IN_PROGRESS`, **client `status==0`**.
- **read** path → `ExchangeConnectionError` (engine reconnects indefinitely).
- **write** path → `OrderDispositionUnknownError` (engine parks + verifies; a blind
  place-retry risks a duplicate order — never do it).

### B — Rate limit
`HTTP 429`, `OA-429`. → `ExchangeRateLimitError(retry_after)` from the
`X-RateLimit-Reset` header (client must stop discarding it; default backoff if absent).
Also respect the per-endpoint quotas proactively — **`Send Email OTP` / `Create Trading
Token` are only 100/hr, 1,000/day** (the token cron must not spin on OTP).

### C — Auth / token
`HTTP 401/403`, `OA-401`, `OA-403`, `FORBIDDEN`, `INVALID_TRADING_TOKEN`, `INVALID_OTP`.
- `INVALID_TRADING_TOKEN` (expired mid-run): **re-read `dnse_trading_token.json` once**
  (the cron may have refreshed it) and retry the call; still invalid → `AuthenticationError`.
  The plugin **never mints** — that's the cron's job.
- `OA-401` / `OA-403` / `FORBIDDEN` (bad API key / permission): `AuthenticationError`
  immediately — a retry won't fix it.

### D — Definitive order reject (won't succeed on retry)
→ `ExchangeOrderRejectedError`, or `InsufficientMarginError` for the money ones.
- **Validation:** `OA-400`, `OA-422`, `INPUT_MISSING/INVALID/FORMAT_INVALID`,
  `INVALID_ORDER_TYPE` **and observed `ORDER_TYPE_INVALID`**, `INVALID_ORDER_SIDE`,
  `INVALID_SYMBOL` / `SYMBOL_NOT_EXIST`, `INVALID_PRICE` / `INVALID_PRICE_LOT`,
  `PRICE_MUST_{LESS,GREATER}...{CEILING,FLOOR}_PRICE`, `INVALID_QUANTITY` / `_LOT`.
- **Margin / buying power → `InsufficientMarginError`:** `PURCHASING_POWER_NOT_ENOUGH`,
  `PP0_EXCEED`, `QMAX_EXCEED`, `STOCK_NOT_ENOUGH`, `VIOLATE_POOL_RULE`,
  `VIOLATE_ROOM_RULE`, `OUT_OF_MARGIN_BASKET`.
- **Symbol halted/suspended:** `CAN_NOT_PLACE_ORDER_ON_{HALTED,AOM_HALTED,SUSPENDED,
  UNLISTED}_SYMBOL`, `CAN_NOT_PLACE_ODD_LOT_ORDER_ON_SPECIAL_SYMBOL` → reject (a bar
  retry is pointless; the symbol won't trade).

### E — Session-gated  ← **POLICY DECISION (see §5.1)**
Place: `CAN_NOT_PLACE_ORDER_ON_THIS_SESSION`,
`CAN_NOT_PLACE_ORDER_WITH_THAT_ORDER_TYPE_ON_{ATO,ATC}_SESSION`,
`INVALID_ORDER_TYPE_FOR_THIS_SESSION`, `INVALID_TRADING_SESSION`,
observed `CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION`, `CANNOT_PLACE_OPPOSITE_ORDER[_IN_THIS_SESSION]`,
`CAN_NOT_PLACE_PLO_ORDER_WITHOUT_MATCHED`.
Cancel/replace: `CAN_NOT_CANCEL_THAT_ORDER_ON_THIS_SESSION`, observed
`CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION`, `CAN_NOT_CANCEL_ATO_ORDER`,
`CAN_NOT_CANCEL_MARKET_ORDER`, `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION`,
`CAN_NOT_REPLACE_*`.
- **Place** refused for session → recommend `ExchangeOrderRejectedError` + a loud
  *protection-degraded* log (drop it; the strategy re-decides next bar — a deferred
  order firing later at a different price is worse than a clean miss).
- **Cancel** refused mid-auction (ATO/ATC) → the order is *still there*, so a short
  retry after the open is safe; treat as a brief `ExchangeConnectionError`-style wait,
  **not** a terminal reject, or the cancel is silently lost.

### F — Already-gone / terminal-state → cancel = success
`RESOURCE_NOT_FOUND`, `INVALID_ORDER_ID`, `ORDER_IS_DONE`, `ORDER_STATUS_REJECTED`,
**observed `CO-ORD-013`** (conditional already `Activated`).
- On **cancel**: nothing left to cancel → **report success** (`_cancel_one` already
  does this for `RESOURCE_NOT_FOUND`; extend the set). For a STOP that returns
  `CO-ORD-013` it has **fired** → it's now a NORMAL order; the fill surfaces via
  `watch_orders`, and the linked NORMAL order is what to manage (see the OCO model).

## 4. Implementation

1. **`errors.py`**: `classify(status, body, *, is_write) -> BrokerError | None`
   (`None` = success). Single table keyed by `code`, then HTTP status; `status==0`
   handled first.
2. **`_place` / `_amend`**: replace the blanket `status not in (200,201) ->
   ExchangeOrderRejectedError` with `exc = classify(...); if exc: raise exc`.
3. **`_cancel_one`**: widen the "already gone" set to
   `{RESOURCE_NOT_FOUND, INVALID_ORDER_ID, ORDER_IS_DONE, CO-ORD-013}` → success.
4. **Reads** (`_iter_orders`, `get_open_orders`, `get_position`, `watch_ohlcv`):
   classify transient → `ExchangeConnectionError`; surface rate limits.
5. **Token**: `INVALID_TRADING_TOKEN` → re-read state file once + retry → else
   `AuthenticationError`.
6. **Client**: stop discarding `X-RateLimit-{Limit,Remaining,Reset}`; expose them so
   `ExchangeRateLimitError(retry_after)` and proactive throttling are accurate.
7. **Tests**: table-driven — one case per action class against canned `(status, body)`
   (the same fake-client seam used to prove the cancel fix), plus the observed codes.

## 5. Policy decisions (settled 2026-08-07)

1. **Session-refused place → reject + log** (`ExchangeOrderRejectedError` + a loud
   degraded-protection WARNING). Drop it; the strategy re-decides on the next open bar.
   No parking — a pre-gap level must not fire into the reopen. (Cancels refused by a
   session are *not* rejects: brief retry after the session flips; the order still rests.)
2. **Transient writes → engine park+verify** (`OrderDispositionUnknownError`); no
   in-plugin place-retry (dup-order safe).
3. **`INVALID_TRADING_TOKEN` → re-read the state file once + retry, else
   `AuthenticationError`.** The plugin never mints (the cron owns that).

## 6. Logging — every error, clearly (required)

Every classified error emits **one structured line before the plugin acts**, so an
operator can see what happened and what was done about it — nothing is ever swallowed:

```
[DNSE] <action> code=<CODE> http=<status> -> <disposition> | order=<pine_id/leg|id> intent=<intent_key> msg="<venue message>"
```

- `action` = `place` / `cancel` / `amend` / `read:<endpoint>`
- `disposition` = `rejected` | `park+verify` | `rate-limit wait(retry_after=Ns)` |
  `auth-fail` | `token-reread` | `treated-gone` | `degraded-protection`

Levels:
- **ERROR** — hard reject (validation / margin / symbol), `AuthenticationError`,
  manual-intervention / quarantine.
- **WARNING** — degraded-protection (session-refused place), rate-limit wait, transient
  write park+verify, transient read reconnect, token re-read.
- **INFO** — benign: an already-gone cancel treated as success (audit trail, not silent).

Rules: never swallow an error silently; log at the point of classification (in
`classify`, or the caller with the identity in hand); **never** log the trading token or
`api_secret`.

## 7. Note on doc completeness

`guide-ratelimits.md` is complete; `guide-error_codes.md` covers the general/legacy
surface but **omits the conditional-order `CO-ORD-*` family** and uses different
spellings from some live strings (`CANNOT_…` vs doc `CAN_NOT_…`, `ORDER_TYPE_INVALID`
vs `INVALID_ORDER_TYPE`). Codes observed live but absent from the docs — **treat this
list as authoritative and append as we see more**:
`CO-ORD-013`, `CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION`,
`CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION`, `ORDER_TYPE_INVALID`.
