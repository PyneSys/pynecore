# DNSE plugin — test-suite plan

Scope: first-party modules under `plugins/dnse/pynecore_dnse/` — `broker.py`,
`provider.py`, `client.py`, `errors.py`, `_sdk.py`, `__init__.py`. The vendored
`_vendor/` (official DNSE openapi-sdk) is **excluded** (third-party). This is a
**live-trading** broker; the goal is a real safety net on the money path, not a
coverage number — assertions must catch real bugs.

## Baseline (measured 2026-08-07, `pytest-cov 7.1.0` — already installed)

From the only existing tests (`tests/test_errors.py`, 39 passing):

| Module | LOC | Stmts | Cover |
|---|---|---|---|
| `errors.py` | 173 | 83 | **94%** |
| `broker.py` | 698 | 379 | **36%** |
| `client.py` | 85 | 34 | **35%** |
| `provider.py` | 334 | 148 | **34%** |
| **first-party total** | | 644 | **43%** |

## Conventions (`pytest.ini`)

- Test functions named **`__test_*__`** (not `test_*`); classes/helpers stay `_prefixed`.
- `--import-mode=importlib`, `-m "not live"` default, `-x` stop-on-first-failure.
- **Mock seam:** inject a fake client as `broker._client` / `provider._client` that
  returns canned `(status, body)` tuples (the `_FakeClient` pattern in `test_errors.py`).
  `async` methods → `asyncio.run`; async generators (`watch_orders`, `watch_ohlcv`) →
  collect N events via `__anext__` and **monkeypatch `asyncio.sleep` / `time.sleep`** so
  poll loops don't wait. No live network, no real filesystem (use `tmp_path`).

## Risk ranking

| Module | risk | category | why |
|---|---|---|---|
| `broker.py` | **Critical** | critical-path | the money path; highest LOC/complexity (ΣCC 177, peak `get_position` CC 17), highest churn (8 commits/3mo), 3 `fix(dnse…)` commits. Order-construction, all state reads, fill detection, amend — untested. |
| `errors.py` | Low | critical-path | 94% covered; only small direct-helper gaps remain. |
| `client.py` | High | high-risk | tiny but sits under every call; the version-pin is a **live-proven silent-no-op landmine** (a floating date made cancels return 200 while cancelling nothing). |
| `provider.py` | High | high-risk | "data-only" but `resolve_contract`/`market_type` feed `broker._place`'s order symbol and book selection — a wrong resolution trades the wrong instrument / scans the wrong book. |
| `_sdk.py`, `__init__.py` | Low / None | utility | `sys.path` shim + re-export; docstring. |

## Highest-value cases per module (money-path first)

**`broker.py`** (critical):
- `_place` — payload rounding/`int` qty/lazy `loanPackageId`; **OCO re-tracking** (swap to `_resolve_oco_lo`'s LO, `tracked_category="NORMAL"` — the exact `cc9849b` bug class); non-dict success body → `ExchangeOrderRejectedError`; `_order_ids`/`_identity`/`_order_category` bookkeeping correct.
- `execute_entry/exit/close` routing table — stop→STOP (`conditionOperator` by side), limit→LO, tp+sl→OCO, sl→STOP, tp→LO, neither→`OrderSkippedByPlugin`, close→marketable band edge.
- `get_position` (CC 17) — weighted-avg entry across multiple rows; net-to-zero → **`None`** (not zero-size); `NB/NS` sign; wrong-symbol rows filtered; non-200 → classify+`_emit`+`ExchangeConnectionError`.
- `get_open_orders` — union NORMAL+STOP, terminal excluded; **both books fail → must raise, never return `[]`** (silently-empty exposure is the named regression).
- `watch_orders` (CC 15, async gen) — fill delta + price; **dedup** (no re-yield on identical); **unknown order skipped** (not ours); status→event map; survives a transient `_iter_orders` blip; `max(delta,0)` clamp.
- `_resolve_oco_lo` — monkeypatch `time.sleep`; `externalOrderId` on attempt 1 vs late vs never (`None`); LO-detail non-dict fallback.
- `_write` token-retry — INVALID_TRADING_TOKEN on call 1 → re-read + retry (exactly 2 calls); still-invalid → propagate, no loop.
- `_amend`/cancel wrappers, `account_id`/`_token` (tmp_path state file: valid→wins, missing→config, malformed→caught, none→`RuntimeError`), `_to_exchange_order` (status map + unmapped→PENDING, numeric coercions), and quick wins (`_marketable_price`, `_gtd`, `_loan_package_id`, `get_capabilities`, `get_balance` silent-`{}` contract).

**`client.py`** (high): the **version pin** = `2026-07-23` regardless of config (single highest-value test); `_parse` (bytes decode, JSON parse, invalid-JSON left raw, empty left as-is); `__getattr__` delegation + underscore→`AttributeError`; **TLS-verify swap** asserted (`cert_reqs="CERT_REQUIRED"`, `ca_certs` set).

**`provider.py`** (high): `resolve_contract` (alias→dated, cache, silent-fallback-on-failure); `market_type` all 3 branches (FU→DERIVATIVE / other→STOCK / heuristic); `update_symbol_info` **stock tick-size bands** (<10→0.01, 10–50→0.05, ≥50→0.10); `download_ohlcv` (ms conversion, progress cadence, tz branch, non-200→`RuntimeError`); `client` property fail-fast on missing creds; `is_production` **mixed-endpoint → still "production"** (safety banner).

**`errors.py`** (small gaps): direct `code_of`/`_message_of`/`_retry_after` (incl. non-numeric retryAfter → `DEFAULT_RETRY_AFTER`).

**`_sdk.py`**: one smoke test — import resolves `dnse` to the **vendored** copy (`sys.modules['dnse'].__file__` under `_vendor/`), guarding vendor-shadow order.

## Execution order (steps 3–7)

1. **broker A** — `_write` retry, `_to_exchange_order`, `_place` (+OCO, non-dict), `execute_entry/exit/close` routing.
2. **broker B** — `get_position`, `get_open_orders` (both-fail raises), `watch_orders` (dedup/skip/map).
3. **broker C** — `_resolve_oco_lo`, `_amend`, `execute_cancel(_with_outcome)`, `account_id`/`_token`.
4. **client** — version pin, `_parse`, TLS swap.
5. **provider** — `resolve_contract`, `market_type`, tick bands, `download_ohlcv`, mixed-endpoint.
6. **errors gap-fill** — direct helper tests.
7. **verify & gate** — full run + coverage gate + mutation sample on broker's money path.

## Quality gates

Line **80** / branch **75** / functions **80** / statements **80** (first-party, `_vendor`
omitted). Mutation sample ≥ 60% on `broker.py`'s order/cancel/classify functions
(feasibility-gated). ≥ 2 assertions/test; all I/O mocked; deterministic (monkeypatch
time + sleep); never weaken an assertion to hit a number; anything untestable is
surfaced, not skipped.
