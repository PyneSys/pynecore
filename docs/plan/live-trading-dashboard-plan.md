# Live-trading control plane ("fleet dashboard") — plan

Date: 2026-08-31. Status: PROPOSAL — nothing built yet.
Goal: one place to see and manage every live `pyne run … --broker` bot across
all broker plugins (dnse, binance, and the PyneSys bybit / capitalcom /
ctrader plugins), without inventing state the engine already keeps.

## 1. What exists today (surveyed 2026-08-31 across ../)

**The engine already journals everything durable into ONE file per workdir:**
`workdir/output/logs/broker.sqlite` (WAL, safe for concurrent read-only
readers). Schema (`src/pynecore/core/broker/storage.py`):

| table / view | what a dashboard gets from it |
|---|---|
| `runs` | one row per process instance: `run_id` (logical, stable across restarts = `strategy@account:symbol:tf[#label]`), `run_tag`, `plugin_name`, `account_id`, `symbol`, `timeframe`, `script_path`, `started/last_heartbeat/ended_ts_ms` |
| `live_runs` VIEW | **"what is running"** = `ended_ts_ms IS NULL AND heartbeat < 5 min old` (engine heartbeats 1/min) |
| `orders`, `order_refs` | live + closed orders per instance with Pine identity, SL/TP levels, `extras` JSON |
| `events` | the audit timeline: `dispatch_submitted`, `confirmed`, `rejected`, `unexpected_cancel_quarantine`, `stale_run_cleaned`, `spot_*`, … |
| `spot_executions`, `spot_inventory_epoch`, `spot_asset_owner` | spot fill ledger, epoch state (`active/quarantined/closed`), per-asset lease |

Account truth is available **offline** by instantiating a plugin and calling
`get_position / get_open_orders / get_balance` — exactly what
`plugins/dnse/tools/venue.py` does, with tri-state exit codes
(0 yes / 1 no / **2 could-not-determine**).

**Gaps in the engine (nothing reads the store cross-run today):**
- No `pyne runs/status` CLI; the only reader of `broker.sqlite` is the
  engine's own single-instance guard (`run.py:1504`).
- Quarantine / halt are in-process latches; the only durable trace is an
  `events` row (`*_quarantine`) — no `runs.status` column.
- No PID / command line / host recorded; no per-run log file (stdout only,
  `error.log` is truncated each start). Logs exist only where a shell
  redirected them (`run_t10_dual.sh` pattern).
- Native fail-safe health is not persisted at all.
- No supervisor, systemd unit, scheduler, or kill switch anywhere.

**Strategy side (`../pinescript`):** ~224 `.pine` strategies under a strict
5-field filename convention (`Name - Asset - TF - Fork NNN - Event`), champions
tracked on GitHub ProjectV2 cards + `docs/strategy-dashboard/data.json`.
**There is no "which strategies are live" registry and no live launcher.**
Live trading is not wired into that repo at all.

**Venue plugins:** dnse + binance (this fork; full staged live suites, L0
gates, `venue.py`, token cron) and the three PyneSys plugins (no status CLI;
`broker_lab` conformance suites; ctrader has `pyne ctrader auth`). Notable
operational hazards to surface: DNSE trading token ~8 h TTL + app-trade
breakage of conditional writes; **Bybit API keys without IP whitelist expire
in ~3 months** (silent auth loss for an unattended bot).

**Reusable UI/infra in the workspace:**
- `../vnstocksectorvnindexcorrelation` — FastAPI + SQLite/WAL + static-mounted
  dashboard, `app/api/jobs.py` (submit/status/dedup/prune job registry),
  `app/engine_watch.py` (staleness → Telegram), and a **three-layer Playwright
  golden test net** (`visual_diff_dashboard.py`, `dom_snapshot_dashboard.py
  --prove-read-only`, `gui_test_*.py`). Backend + tests worth lifting; its
  350 KB hand-rolled JS frontend is not.
- `../Vibe-Trading` — `frontend/src/hooks/useSSE.ts` (reconnect, backoff,
  dedup, Last-Event-ID), SSE auth tickets (`security.py`), and the
  `/live/status` + `/live/runner/{start,stop}` + `/live/halt` endpoint
  contract with idempotent responses. Patterns to copy, repo not to fork.
- `../tradingview_crawler` dashboard — a Flask single-table toy; ignore.

## 2. Recommendation

Build a **thin control plane over the existing journal**, not a new trading
runtime. Three layers, smallest first:

### Layer A — engine substrate (inside pynecore, small, upstream-friendly)
1. `pyne runs` CLI (`ls | show <run_id> | events <run_id> | accounts`) — the
   first cross-run reader of `broker.sqlite`. Ships value on day one with no
   UI and doubles as the dashboard's reference implementation + test oracle.
2. Derive a **run status** in a shared reader module
   (`core/broker/fleet_reader.py`): `live | stale | ended | quarantined |
   halted`, computed from `live_runs` + `events` (`*_quarantine`,
   `stale_run_cleaned`) + `spot_inventory_epoch.state`. Persisting a
   `runs.status` column is optional later; deriving keeps the migration list
   untouched.
3. Record `pid` + `hostname` + `argv` on the `runs` row (one additive
   migration) so the supervisor and the store agree on identity.
4. Optional `--log-file <path>` on `pyne run` (or leave capture to the
   supervisor — see B2; decide in P0).

### Layer B — control-plane service (`pyne-control`, new sibling repo)
FastAPI + its own small SQLite (bot definitions, schedules, action audit) +
read-only attach to each configured `broker.sqlite`.
1. **Fleet reader**: polls `live_runs`/`orders`/`events` (WAL read-only
   connection) → SSE stream of run status, heartbeat age, position, equity,
   last event. Multiple workdirs supported (pinescript's `pyne-run` skill
   uses per-run dirs; config lists them).
2. **Supervisor**: launches `pyne run <script> <provider:sym@tf> --broker
   --run-label …` as a child process with CWD pinned to the workdir (DNSE's
   token path is CWD-relative), captures stdout to
   `logs/<run_id>/<instance>.log`, tracks PID, maps PID→`run_instance_id` via
   Layer A.3. Endpoints modelled on Vibe-Trading: `POST /runs/{id}/start`
   (idempotent `already_running`), `/stop` (SIGINT → graceful shutdown,
   `--shutdown-timeout`), `POST /halt` (global kill switch: stop all, then
   per-venue `cancel_all` through the offline plugin path). Later: emit
   systemd units instead of children (same API).
3. **Pre-flight gates wired into start**: refuse to launch unless the
   plugin's L0 gate exits 0 (`plugins/binance/tools/l0_gate.py`,
   DNSE `l0_order_semantics.py`), `venue.py flat`-style cleanliness passes
   where required, and the trading window is sane (VN session phases for
   DNSE, 24/7 for crypto). Gates are per-plugin adapters, not hard-coded.
4. **Accounts**: per venue, offline plugin reads (balance, position, open
   orders) with the tri-state contract — a failed read renders as
   **UNKNOWN**, never as flat (the 2026-08-19 false-flat incident is the
   rule here). Credentials never leave the workdir config files.
5. **Alerts** (copy `engine_watch`): stale heartbeat, quarantine event,
   process died without `run stopped` line, DNSE token TTL < 1 h, Bybit key
   age > 75 days, disk/DB growth. Telegram sink exists in vnstock.
6. **Bot registry** = the missing "what is live" list: strategy file
   (pinescript Fork path) + provider string + plugin + label + schedule +
   link to the champion card. Writes back a `live` verdict into
   `docs/strategy-dashboard/data.json` (its design already reserves a
   `champion | candidate | rejected | archived | inert` taxonomy).

### Layer C — UI (Vite + React + TS + Tailwind, fresh scaffold)
Pages: **Fleet** (one row per run: plugin, symbol/tf, status pill, heartbeat
age, position, equity, last event, start/stop), **Run** (orders table,
events timeline, live log tail via SSE, controls), **Accounts** (per venue
balances/positions/open orders + gate results), **Alerts**. `useSSE.ts`
copied verbatim. Auth: bind 127.0.0.1 by default; single bearer token +
SSE tickets if exposed on LAN. No public exposure in scope.

### Testing (non-negotiable, same discipline as the plugin suites)
- Fixture `broker.sqlite` files generated by the real `BrokerStore` (not
  hand-written SQL) → reader/status-derivation unit tests.
- Supervisor tests against a stub `pyne` executable (heartbeats, crash,
  SIGINT path, stale-guard collision).
- Lift vnstock's **three-layer golden net** wholesale — pixel, DOM, and
  contract-driven `gui_test_*` — with `--prove-read-only` because the app
  can reach order-placing code.
- E2E: Binance **testnet** (already proven) + DNSE fake venue; never mainnet.

## 3. Phases (each ends with a verifiable gate)

| Phase | Deliverable | Verify |
|---|---|---|
| **P0 spike (1–2 d)** | `pyne runs ls/show/events` over the existing 59-run `broker.sqlite`; `fleet_reader.py` status derivation | matches the engine's own `live_runs` + a hand-checked quarantine case from the 08-17 DNSE runs |
| **P1 read-only fleet (3–4 d)** | service + SSE + Fleet/Run pages, log tail from supervisor-captured files | shows the Binance testnet staged run live while it runs; golden net green |
| **P2 control (3–4 d)** | supervisor start/stop, gates, kill switch, audit log, PID/argv migration | start refused on failing L0; stop leaves `ended_ts_ms` set and venue clean; halt sweeps a testnet book |
| **P3 accounts + alerts (2–3 d)** | offline venue reads (tri-state), alert rules, Telegram | a forced read failure renders UNKNOWN, not flat; stale-heartbeat alert fires on a killed stub run |
| **P4 registry + schedule (2–3 d)** | bot definitions, VN session / 24/7 schedules, DNSE token-cron integration, write-back to strategy dashboard | a scheduled DNSE bot starts pre-open behind the gate and stops at close |
| **P5 hardening** | systemd units, multi-workdir, key-expiry tracking, docs | soak: 1 week unattended on testnet + DNSE with zero silent failures |

Out of scope (explicitly): a new order runtime, cross-host fleets, public
internet exposure, strategy optimisation UI (pinescript owns that).

## 4. Decisions needed before P0

1. **Home**: new repo `pyne-control` (recommended — spans pynecore workdirs
   and pinescript) vs `pynecore/tools/`. Layer A lands in pynecore either way.
2. **Process model**: child processes under the service (fast, single host)
   vs systemd units from day one (survives service restarts). Recommend
   children in P2, systemd in P5.
3. **Exposure**: localhost only, or LAN with a token? Drives auth scope.
4. **Alert channel**: Telegram (exists in vnstock) vs email vs both.
5. **Which bots first**: Binance testnet + one DNSE strategy is the natural
   pair — both have measured live suites to validate the dashboard against.
