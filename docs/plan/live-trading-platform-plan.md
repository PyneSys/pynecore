# PyneCore live-trading platform — 24/7 hosted service (bigger picture)

Date: 2026-08-31. Status: PROPOSAL. Supersedes the scope of
`live-trading-dashboard-plan.md` (which becomes the "Fleet" module of this).

## 0. The product in one sentence

A self-hosted TradingView-style strategy runner: open the dashboard, pick an
asset, pick a strategy, edit its Inputs / Properties exactly like the TV
dialog, press *Go live*, then pause / resume / stop it and read its trigger
log — except the "alert → webhook → bot" chain is replaced by pynecore's
broker plugins placing the orders directly, running 24/7 in a datacenter.

## 1. UX contract — the TradingView screens mapped onto what exists

| TV screen (screenshots) | Platform equivalent | Already exists in pynecore? |
|---|---|---|
| **Inputs tab** (Length, Stop Loss %, checkboxes, groups) | Auto-rendered form from the strategy's `input()` declarations | **Yes** — the generated `.toml` carries `input_type / defval / title / minval / maxval / group` per input; `value =` is what the runner reads |
| **Properties tab** (initial capital, order size + type, pyramiding, commission, slippage, calc on bar close…) | Form over the `[script]` block | **Yes** — `initial_capital`, `default_qty_type/value`, `pyramiding`, `commission_type/value`, `slippage`, `calc_on_every_tick`, `process_orders_on_close` are all `.toml` keys |
| **Style / Visibility tabs** | Chart page cosmetics | Later; `--viz-journal` NDJSON is the plot feed |
| **Alerts list** (active/inactive, filter by symbol+interval, *Pause/Restart as per filter*, delete inactive) | **Instances** list with bulk pause/resume/stop | Partly — `live_runs` view gives state; pause needs an engine control channel (§4) |
| **Log** (trigger name, payload, symbol+TF, "webhook successfully delivered", CSV export) | **Events** timeline: dispatch → venue ack (order id) → fill/cancel/reject, filter by instance/symbol/TF/type, CSV export | **Yes** — `broker.sqlite` `events` + `orders`; "delivered" = the venue order id echoed on `dispatched` |
| Import / Export / Defaults buttons | Instance param presets as JSON; "Defaults" = the `.pine` defvals | Trivial over the `.toml` |

## 2. Domain model

- **Strategy** — a `.pine` from `../pinescript` (Fork-named) + its transpiled
  `.py`, versioned by git SHA. Inputs schema extracted at import time.
- **Account** — a broker plugin + its `workdir/config/plugins/<x>.toml`
  credentials (dnse_broker, binance_broker, bybit, capitalcom, ctrader).
  Secrets never enter the platform DB (§6).
- **Instance** ("live strategy", the TV alert) — Strategy version × Asset
  (`provider:symbol@tf`) × Inputs overrides × Properties × Account ×
  run-label × Schedule. States: `draft → armed → running → paused → running
  … → stopped`, plus `quarantined` / `halted` / `stale` derived from the
  engine. One instance = one `pyne run --broker` process in its **own
  workdir** (outputs are named after the script; sharing a workdir clobbers —
  the lesson pinescript already learned) with `PYNE_DATA_DIR` pointed at one
  shared candle cache.
- **Event** — a row in that instance's `broker.sqlite` `events`/`orders`,
  surfaced with TV-log semantics.
- **Schedule** — trading window per venue (DNSE session phases + pre-open
  token mint; crypto 24/7), maintenance windows, auto-stop at contract expiry.

## 3. Control semantics (must be exact — this is real money)

| Action | Meaning | Implementation |
|---|---|---|
| **Start / Go live** | launch process behind the plugin's **L0 gate** + flat-check; instance runs warmup then live | supervisor spawns `pyne run … --broker --run-label <instance>`; refuses on gate ≠ 0 |
| **Pause** (TV "pause alert") | engine keeps running and *observing*; **no new entries**; protective exits, cancels and closes still flow | NEW engine control (§4): reversible operator latch, same block as quarantine's entry gate |
| **Resume** | clear the latch | control channel |
| **Stop** | graceful SIGINT → `run stopped` summary, `ended_ts_ms` set; open position/orders **stay at the venue** and are re-adopted on next start (`run_id` identity + orphan adoption) — matches Hummingbot's *force*-stop button ("cancels orders", leaves position), not its graceful one (§9.6) | supervisor; UI must show "position left open" explicitly, visually distinct from Flatten & stop |
| **Flatten & stop** | `cancel_all` + `close_all` then stop — matches Hummingbot's *graceful* STOP ("closes positions"); the two must be visibly different controls, not one button behind a confirm dialog (§9.6) | via the engine (preferred) or offline plugin call as fallback; requires typed confirmation |
| **Change params** | TV re-arms instantly; here it is a **controlled restart**: save new `.toml` immediately, mark the instance "restart pending" (OctoBot's staged-restart pattern, §9.6) — stop → apply → start whenever the operator chooses; warmup replays with the new params, venue position is adopted at startup. A same-direction re-entry after a param change composes with the engine's existing reversal sizing (an opposite-direction entry against an open position nets against it in one order, never two racing orders) — same in-process atomicity PineConnector's flip commands (`closelongopenshort` etc.) exist to fake over a webhook hop (§9.5) | UI warns: strategy `var` state is recomputed from history and may disagree with the adopted position; offer "flatten first" as the alternative to clearing the pending-restart badge |
| **Kill switch** | stop everything on the host, sweep every account; scalar mark-to-market PnL-rate auto-trigger modeled on Hummingbot's kill switch maps to **Stop**, never **Flatten** — say so explicitly in the UI so a triggered switch is never mistaken for a closed position (§9.6) | global endpoint + physical button; audit-logged |

## 4. Engine work required (pynecore, upstream-friendly, all small)

1. **Operator control channel** — a `run_controls` table (`run_id`,
   `desired_state ∈ {run, pause, flatten, stop}`, `requested_ts_ms`,
   `acked_ts_ms`) polled once per sync cycle alongside the heartbeat. Pause
   reuses the existing entry-dispatch block; unlike quarantine it is
   reversible. Acks make the UI truthful (requested vs in effect).
2. **`runs.pid`, `runs.hostname`, `runs.argv`** (additive migration) and a
   supervisor-side "instance is dead" fast path so a crashed bot can be
   restarted immediately instead of waiting for the 5-min stale window
   ("Active run_id already exists").
3. **Derived status** in one reader module (`live | stale | ended |
   paused | quarantined | halted`) from `live_runs` + `events` +
   `spot_inventory_epoch` + controls — shared by `pyne runs` CLI and the API.
4. **`pyne runs ls|show|events|tail`** — CLI over the same reader (day-one
   value, test oracle for the API).
5. Optional: `--log-file`; per-input `group` already in metadata — verify
   `.toml` emits it so the form can reproduce TV's section headers.

## 5. Platform architecture (single VPS)

```
[browser] —TLS/VPN→ Caddy → api (FastAPI, SSE)  ─┐
                              ↓ own sqlite (instances, presets, audit)
                        supervisor (systemd-run transient units, 1 per instance)
                              ↓ spawns                 ↓ read-only WAL
                 instance workdirs  ─ broker.sqlite ─ fleet reader → SSE
                              ↓ PYNE_DATA_DIR
                        shared candle cache (one .ohlcv per provider/symbol/tf)
                 venue adapters: L0 gates, offline reads (tri-state), token cron
                 alerts → Telegram/email; backups → object storage nightly
```

- **Process model**: `systemd-run` transient units per instance (auto-restart
  policy, journald capture, survives api restarts) — not bare children.
- **Realtime**: SSE (Vibe-Trading `useSSE` pattern) for fleet status, event
  log, live log tail; polling fallback.
- **Frontend**: Vite + React + TS + Tailwind; forms generated from the
  inputs schema; Lightweight-Charts chart page fed by `--viz-journal` (phase 5).
- **Tests**: fixture `broker.sqlite` built by the real `BrokerStore`; stub
  `pyne` binary for supervisor tests; the vnstock three-layer Playwright
  golden net with `--prove-read-only`; e2e on **Binance testnet** (proven)
  and the DNSE fake venue. Mainnet only via the plugin's own guards.

## 6. Hosting, security, operations (the "24/7 in a datacenter" part)

- **Region**: DNSE is Vietnam-only; Binance/Bybit serve Asia from SG/TYO.
  Pick a Singapore or Vietnam VPS (Vultr/Hetzner-SG/Viettel IDC). 2 vCPU /
  4 GB is plenty — bar-close bots are idle 99 % of the time; size for I/O
  and uptime, not CPU. **Always-on NTP** (Binance `recvWindow`, L0 checks skew).
- **Exposure**: no public dashboard port. WireGuard/Tailscale to the box, or
  Cloudflare Access in front of Caddy; TLS everywhere; single-operator login
  + TOTP; every mutating action audit-logged with actor + reason.
- **Secrets**: plugin TOMLs live only on the VPS disk, encrypted at rest
  (sops/age), injected at boot; **venue API keys IP-whitelisted to the VPS**
  (also fixes Bybit's 3-month expiry and the earlier exposed-key exposure);
  DNSE OTP/Gmail minter runs on the VPS pre-open on a timer.
- **Resilience**: systemd `Restart=on-failure` with the §4.2 fast restart;
  the engine already re-adopts positions on restart; nightly `broker.sqlite`
  + config backup to object storage; `engine_watch`-style watchdog paging on
  stale heartbeat, quarantine, dead process, disk, token TTL, key age.
- **Deploy**: Docker Compose or plain systemd + `uv` venv; pinned pynecore
  fork + plugins by SHA; blue/green not needed — instances are stopped in a
  maintenance window that the scheduler owns (never mid-session for DNSE).
- **Blast radius**: one Linux user per venue account; the kill switch is
  reachable without the dashboard (SSH script + Telegram command).

## 7. Phases (each with a verifiable gate)

| Phase | Deliverable | Gate |
|---|---|---|
| **P0 (1–2 d)** engine substrate | `pyne runs` CLI, derived status, pid/argv migration, `run_controls` table + pause latch | pause on a testnet bot blocks a scripted entry, resume lets the next one through; measured, not assumed |
| **P1 (1 wk)** hosted read-only fleet | VPS provisioned (WireGuard, Caddy, NTP, backups), api + SSE + Instances list + Events log with CSV export, systemd-run supervisor **read path** | the Binance testnet staged suite runs on the VPS and is watched from your laptop; golden net green |
| **P2 (1 wk)** instance lifecycle | create instance from strategy + asset, Inputs/Properties forms from `.toml`, start behind L0 gate, stop, pause/resume, flatten, kill switch, audit | every action verified from the venue record on testnet; a failing L0 blocks start; "position left open" shown on stop |
| **P3 (1 wk)** accounts + alerts + schedule | offline tri-state venue reads, alert rules → Telegram, DNSE session schedule + token timer, contract-expiry auto-stop | a killed process pages within 2 min; a DNSE bot starts pre-open and stops at close unattended for 3 sessions |
| **P4 (3–4 d)** param workflow | presets import/export/defaults, controlled-restart param change with adoption warning, "preview backtest" of the instance's params over cached data (equity + trades) | preview equals `pyne run` file-mode output for the same `.toml` |
| **P5** chart + polish | Lightweight-Charts with trades/orders from `--viz-journal`, style tab, mobile layout | — |
| **P6** first real money | one DNSE strategy at minimum size behind everything above | 2-week unattended soak on testnet + DNSE with zero silent failures first |

## 8. Decisions needed

1. **VPS region/provider** (SG vs VN; DNSE latency vs crypto) and budget.
2. **Access model**: WireGuard/Tailscale (recommended) vs Cloudflare Access.
3. **Pause semantics**: confirm "block new entries, keep exits" (§3) vs
   "freeze everything" — the former is what protects an open position.
4. **Param change**: confirm the OctoBot-style staged-restart flow (save
   now, operator-triggered restart, §3/§9.6) vs requiring flat before edits.
5. **Repo home**: new `pyne-platform` repo (api + ui + deploy) with engine
   changes in pynecore — recommended.
6. **Alert channel** and who is on call when the watchdog pages.
7. **TradingView import bridge** (§9.3): worth building now (cheap, retires
   the fragile Tampermonkey userscript) or defer to a later phase?

## 9. What TradingView's own strategy UX actually does (researched 2026-08-31)

Deep research (official Pine Script v6 docs, Help Center, and live inspection
of the user's own TradingView account — the 2026-redesigned Strategy Tester
was confirmed hands-on, not just from docs) turned up the exact shape of the
thing we're modeling ourselves on, several load-bearing gotchas, and a
precise list of what the ecosystem around TradingView does badly that we can
do better by construction. Full detail lives with the research; this section
distills what changes the plan.

### 9.1 The Strategy Tester panel — richer than assumed, and we already compute most of it

TradingView renamed "Strategy Tester" to **strategy report** in a 2026
redesign and added a `Bar detalization` control (`Default (4 ticks per bar)`
/ `High (~60 ticks per bar)`, Premium+) that **replaced** the old "Bar
Magnifier" checkbox — confirmed live: 15m chart showed exactly
`~28 ticks per bar`, matching the predicted math (15m ÷ chosen intrabar TF ×
4) before we ever opened the dropdown. Two toolbar controls, confirmed live:

- **`Script execution`** — a checkbox group replacing the old "Recalculate"
  checkboxes: `On bar close` (always on, cannot disable), `On order fill`,
  `On history bar tick` (new in v6; Premium/Ultimate, standard charts only),
  `On realtime bar tick`.
- **`Bar detalization`** — as above.

Report structure, confirmed live on a real VN30 strategy:

- **Key stats** strip: Total PnL, Max drawdown, Profitable trades, Profit factor.
- **Performance** chart: Cumulative PnL, Buy and hold, Trades excursions,
  Run-ups and drawdowns — toggleable overlays on one chart.
- **Performance analysis**, five sub-tabs: `Breakdown` (gross profit/loss,
  profit factor, commission load, P&L by signal/by side), `Periodical`
  (CAGR, total return, **Sharpe ratio**, **Sortino ratio**, daily/weekly/
  quarterly/yearly PnL bars), `Benchmarking` (strategy vs buy-and-hold
  return, outperformance, correlation), `Margin usage` (margin efficiency,
  average margin used, margin calls, total liquidated volume, a margin-
  utilization time series), `Growth and decline` (avg run-up/drawdown
  duration, max drawdown as % of initial capital, alternating growth/decline
  chart).
- **Trades analysis**: Distribution, Streaks, Time patterns, plus expected
  payoff, outliers PnL, largest win/loss, returns distribution histogram,
  win/loss/breakeven trade counts.
- **List of trades**: a real table (Trade #, Type, Date/time, Price, Size,
  Net PnL, Return) with a **column-setup picker** adding Signal, Commission,
  Favorable/Adverse excursion, Cumulative PnL, Duration (bars) — confirmed
  live via the gear icon next to the CSV-download button.

`src/pynecore/core/strategy_stats.py` **already computes almost this entire
metric set** — net/gross profit/loss, profit factor, percent profitable,
max equity drawdown, max run-up, buy & hold return, Sharpe, Sortino, max
contracts held, margin calls, avg/largest win/loss, avg bars in trades,
consecutive win/loss streaks, long/short splits, and per-trade rows with
the same fields TV's List of Trades exposes. **The Strategy Tester tab in
our runner is primarily a rendering job over data we already produce**, not
new computation. Genuine gaps to add: the equity-curve-vs-buy-and-hold
overlay chart, the Margin usage panel (needs the live margin/leverage model
from §9.2), and the CSV-exportable, column-configurable trade table.

### 9.2 Properties tab / broker-emulator settings — the EMU vs REAL split is now a hard design rule

TradingView's `strategy()` has 34 parameters (v6). Full mapping is in the
research; the finding that changes our settings UI:

**TradingView itself has no live auto-trading** — its own Help Center is
explicit: *"Strategy trading is limited to the backtesting mode only.
Automated strategy trading with a brokerage account is not available on
TradingView yet."* Every Properties setting is therefore a **simulation
assumption**, never an order-routing config. That boundary is one we must
draw explicitly, because our runner *does* place real orders and a setting
that quietly governs both paths is a live-money bug waiting to happen.

**Rule: split the settings screen into two zones.**

| Zone | Settings | Live behavior |
|---|---|---|
| **Execution** (real) | `default_qty_type/value`, `pyramiding` (advisory — see below), `close_entries_rule` (FIFO is an NFA 2‑43b requirement for US-regulated instruments), `calc_bars_count` (warmup depth), `strategy.risk.*` guards | Applied live, re-rounded to the venue's lot/step/min-notional rules |
| **Simulation assumptions** (backtest-only) | `slippage`, `commission_type/value`, `backtest_fill_limits_assumption`, `use_bar_magnifier`/Bar detalization, `fill_orders_on_standard_ohlc`/Heikin Ashi mode, `initial_capital`, `currency`, `margin_long/short` + the emulator's margin-call algorithm | **Grey out and label "backtest-parity only — not applied live"** in live mode; never let them touch a live order. Backtest still uses them for parity with the TV oracle. |

Two settings need special handling because they're real in spirit but fake
in TV's implementation:

- **`initial_capital` / `currency`** — in live mode these must be **read-backs
  from the broker** (`get_balance()`), not user-entered numbers. If the read
  fails, the field shows **could-not-determine**, never a stale/default
  value — this is the same tri-state discipline already in the DNSE/Binance
  `venue.py` tools, now extended to the settings UI.
- **`margin_long`/`margin_short`** — TradingView's margin-call algorithm
  liquidates **4× the calculated cover amount** "to prevent continuous
  margin calls across subsequent bars," and *"short trades … are subject to
  forced liquidation even if `margin_short` is 100."* This is a backtest
  convenience with no venue counterpart — **never simulate a margin call
  live; always read the broker's actual margin state.**

TradingView's own **2026 UI redesign** (confirmed live) renamed several
widgets — useful naming to borrow since users coming from TV will recognize
it: `Script execution` (recalculation checkboxes), `Bar detalization`
(replaces "bar magnifier"), `Heikin Ashi mode` (replaces "fill orders using
standard OHLC" — HA-chart-only), `Limit order execution` (replaces "verify
price for limit orders" — now a lossy 2-option dropdown: reach-level vs
1-tick-beyond; **keep ours as the original numeric ticks field**, since the
dropdown is strictly less expressive than the parameter it fronts),
`Order execution delay` (replaces the bar-close checkbox), `Long/Short
leverage` (replaces margin-percent inputs, auto-converted).

**`strategy.risk.*` is pynecore's real live control surface** (already
implemented — `src/pynecore/lib/strategy/risk.py`): `allow_entry_in`,
`max_cons_loss_days`, `max_drawdown`, `max_intraday_filled_orders`,
`max_intraday_loss`, `max_position_size`. TV's docs stress these commands
*"execute on every tick and order execution event, regardless of any
changes to the strategy's calculation behavior. There is no way to
deactivate any of these commands."* — i.e. they're the one setting family
that is real both in backtest and live, and the natural place to expose
**per-instance risk guards** in the platform UI rather than inventing a
parallel mechanism (§9.4 revises the risk-guard plan on this basis).

### 9.3 Inputs dialog — confirmed buildable from what we already emit

`src/pynecore/core/script.py`'s `InputData` dataclass already captures the
full TradingView input schema in memory: `input_type, defval, title, minval,
maxval, step, tooltip, inline, group, confirm, options, display`. This is
enough to render TV's Inputs tab faithfully, section headers (`group`) and
same-row layout (`inline`) included. The only gap: today it's serialized to
`.toml` as **human-readable comments**, not structured data — a
`pyne inspect --json` (or similar) inputs-schema export is a small, precise
addition that unblocks the whole forms layer, and should land early (P2).

**A TradingView-import bridge is now in scope, cheaply.** The user's own
`../tradingview-settings` Tampermonkey userscript exports TV's settings
dialog as `{inputs: {in_0: value, …}, meta: {in_0: {name, group, type,
defval}}}` by walking TradingView's React fiber internals. The `name +
group + type + defval` tuple is the same field set our `InputData` already
carries, so we can **ingest a JSON export from that userscript to seed an
instance's parameters** — no retyping the 15+ inputs of an already-tuned
strategy, and no drift between what was tuned on TV and what runs live.
Join on `(title, group)` rather than the positional `in_N` key (TV's
internal ids aren't guaranteed stable across script edits). This retires
the userscript's own fragility (it needs a maintenance workflow just to
survive TV's CSS/DOM changes) for any strategy that moves onto our runner.

**Widget mapping, confirmed against the official Pine v6 docs** (each Pine
input type → what to render):

| Pine input | Widget | Note |
|---|---|---|
| `input.int`/`input.float` | number spinner (`minval`/`maxval`/`step`), or dropdown if `options=` given | **TradingView has no slider** for numeric inputs — don't add one just because `minval`/`maxval` invites it; match TV's plain stepper. |
| `input.bool` | checkbox | defaults to hidden from status line/data window (`display.none`) |
| `input.string` | text field, or dropdown if `options=` | |
| `input.text_area` | multiline textbox | hard 40,960-char cap |
| `input.symbol` | symbol-search widget | empty `defval` = current chart symbol |
| `input.timeframe` | dropdown, optionally restricted via `options=` | |
| `input.source` | dropdown: `open/high/low/close/hl2/hlc3/ohlc4` + other plots on the instance | |
| `input.color` | swatch → picker with alpha slider | |
| `input.time` | date+time picker; on-chart vertical-line marker if `confirm=true` | |
| `input.price` | numeric + on-chart horizontal-line marker if `confirm=true` | pairing a `time`+`price` input under the same `inline` merges their two markers into one draggable point — a nice touch worth copying for any chart-page input UX (P5) |
| `input.session` | a plain start/end time-range field | **not** a unified day+time widget on TV itself — day-of-week is bolted on as a second, ordinary `options=` dropdown by convention, so we don't need to build anything fancier than TV does |
| `input.enum` (v6) | dropdown of enum member titles | |

Layout confirms what `InputData` already models: declaration order is
render order (no explicit ordering field needed), `group=` is a section
header, `inline=` puts fields on one row and auto-wraps, `tooltip=` is a
hover icon (only the **last** one shows when several inputs share an
`inline`). Dialog chrome: a **Defaults** dropdown with exactly `Reset
settings` / `Save as default` / `Reset to TradingView defaults` — worth
mirroring verbatim since that's the vocabulary a TV-trained operator
already knows.

**One genuinely useful negative finding:** TradingView's own public API
surface (the Charting Library's `StudyInputInfo`/custom-study `metainfo`)
does **not** expose `group` anywhere — it's Pine-compiler/dialog-internal
only, with no publicly documented interface. Our `InputData.group` is
already ahead of what TradingView itself documents as a stable contract,
which is a good sign for the `pyne inspect --json` export (P2): there's no
external schema we're falling short of matching.

**What TradingView does NOT document, and we therefore get to decide
ourselves:** whether an existing chart's saved input values survive a
script edit/republish (kept, reset to new defaults, or the removed input
silently dropped) has **no official TradingView documentation at all** —
confirmed absent from the publishing, FAQ, and script-structure pages.
Since there's no TV behavior to match, this becomes a first-class product
decision for the "Change params" flow (§3): our answer is already better
than TV's undocumented one — old values are kept verbatim on a param
change unless the input was removed, in which case it's dropped with a
visible note on the instance's pending-restart badge (§9.6), never silently.

### 9.4 What TradingView + the webhook bridge ecosystem does badly — our actual differentiators

This is the most consequential finding for the control-plane design (§4–5
of this plan). TradingView's alert→webhook chain, and every commercial
bridge built on top of it (PineConnector, TradersPost, 3Commas, Alertatron,
WunderTrading, PickMyTrade), share the same structural weaknesses — and
since our runner places orders **in-process**, we eliminate the whole
transport layer that causes them, which is worth stating plainly in the
platform's own design rationale:

1. **TradingView's webhook retry is real but nearly useless.** It resends
   ONLY on HTTP 5xx (except 504), 5-second delay, max 3 resends (4 sends
   total) — and explicitly does **not** retry timeouts (3s), connection
   failures, DNS failures, or 4xx. A slow or unreachable receiver loses the
   signal permanently, zero retries. TradersPost's own "Known Limitations"
   page: *"failed/rejected orders are NEVER automatically retried… we don't
   have a way currently to guarantee that the order creation did not
   actually succeed"* — i.e. **no idempotency keys anywhere in the
   ecosystem.** Our supervisor already has the natural fix: the broker
   store's `client_order_id` scheme (deterministic, run-tag-scoped) is
   exactly the idempotency key this whole industry lacks — we get it for
   free from the existing engine.
2. **No delivery ID, no sequence number, no ordering guarantee.**
   TradingView documents no ordering guarantee at all; TradersPost's answer
   to concurrent signals is "leave 1–5 minutes between signals," not a real
   solution. Our engine's bar-clock + single-process dispatch already gives
   us total ordering per instance for free — call this out as a property to
   preserve, not re-derive.
3. **Alerts silently run stale code.** TradingView explicitly: *"TradingView
   saves a mirror image of the script and its inputs… changes to your
   chart's strategy will have no effect on the operation of its copy
   running on our servers"* — editing the script or its inputs afterward
   does nothing, with **no UI signal** that the alert is stale. This is the
   platform-scale version of the "verify you're testing new code" rule
   already in this repo's global instructions. Fix: **stamp the strategy's
   git SHA + inputs hash into every instance's identity** (already partly
   there via `run_tag`, which hashes `script_source` — extend the pattern to
   surface a visible version fingerprint in the UI itself, not just the coid).
4. **The delivery log conflates "no ACK" with "not delivered."**
   TradingView's 3-second timeout produces false negatives — the vendor
   consensus is *"in most cases the webhook has already reached the
   destination… TradingView just gave up waiting."* Our Events timeline
   (§9.1/§1 of this plan) is sourced from the broker's own confirmed state
   (`dispatched → confirmed/rejected`), never from a client-side timeout —
   already immune to this class of bug by construction.
5. **No signed requests.** TradingView offers only an obscure, largely
   unused client TLS certificate; the entire bridge ecosystem instead
   defaults to secret-in-URL or secret-in-body. N/A for us — there is no
   webhook hop at all between signal and order.
6. **No liveness heartbeat.** TradingView has five *documented* silent
   alert-kill conditions (rate-limited at 15 triggers/3min, plan-downgrade
   auto-stop, 2-month expiry, script runtime error, quota exceeded) with
   **zero notification on any of them** — a dead alert looks identical to a
   quiet market. This directly validates the plan's existing P3 alerting
   phase (stale heartbeat, quarantine, dead process) — TradingView's own
   failure catalogue is effectively the alert-rule backlog for that phase.

**Net effect on scope:** nothing above adds new phases — it validates and
sharpens P0–P3 as already planned, and gives concrete, evidenced language
for *why* (useful when explaining the platform's value over "just use
TradingView + a webhook bridge").

### 9.5 Risk-guard vocabulary — informed by PineConnector's command set and TradersPost's honesty

PineConnector's MT4/5 command language is the richest risk/sizing
vocabulary found in the bridge ecosystem and is a good reference for our
per-instance risk-guard config (§3/§5 revision): position sizing by
`vol_lots` / `vol_dollar` / `vol_pct_bal_loss` / `vol_pct_eq_loss`, SL/TP by
pips/price/pct, breakeven trigger+offset, ATR trailing (timeframe, period,
multiplier, shift, trigger), an entry spread filter, and — most relevantly
— **atomic flip commands** (`closelongopenshort` etc.) that exist
specifically because two separate uncoordinated webhooks race each other.
Our engine's `strategy.exit()`/`strategy.entry()` reversal semantics already
handle this atomically in-process, composed into the Change-params flow (§3) — this
is confirmation the architecture is right, not new work.

TradersPost's **"Known Limitations"** page is the single most useful
negative document found: explicit admissions of no HFT support, no order
retry, no race-condition prevention, and a recommended 1–5 minute gap
between signals as their only mitigation. Worth keeping as a checklist of
"things a webhook-based bridge cannot do that our in-process runner can":
sub-second reaction to a bar close, guaranteed per-instance ordering, and
atomic reversal — all now explicit non-goals-avoided rather than assumed.

### 9.6 Live-bot management UX patterns worth adopting (Hummingbot / Jesse / OctoBot)

Research into three mature self-hosted bot platforms surfaced concrete UX
patterns directly applicable to §3 (control semantics) and §5 (architecture):

- **OctoBot's staged-restart model is the best answer to the "config change
  while running" problem** — better than either extreme. Rather than
  refusing edits (Hummingbot: *"Configuring the strategy while it is
  running is not currently supported"*) or silently deferring them (Jesse:
  changes only apply to a **new** session), OctoBot lets you **save now,
  restart later**, with a visible `Activation required` badge on the
  pending change and an explicit `Apply changes and restart` action when
  the operator chooses to disrupt the live bot. **Adopt this pattern for
  our controlled-restart param-change flow (§3):** save the new `.toml`
  immediately, show a persistent "pending restart" badge on the instance
  card, let the operator pick the moment (not auto-restart), matching the
  plan's existing "flatten first" option as one path to clearing that badge.
- **Hummingbot's kill switch is the right shape for ours**: one scalar
  mark-to-market PnL-rate threshold, evaluated continuously (not just on
  trades — *"the trigger responds to real-time market price fluctuations,
  not just completed trades"*), action = **stop + cancel resting orders**,
  explicitly **not** a flatten. This matches our plan's existing distinction
  between "Stop" and "Flatten & stop" (§3) — Hummingbot's kill switch maps
  to our "Stop," never our "Flatten," and that mapping should be explicit in
  the UI copy so an operator doesn't assume a triggered kill switch closes
  the position.
- **Hummingbot's Instances page has two visually distinct stop actions** —
  a "STOP" that *"gracefully closes positions"* vs a force-stop icon that
  *"force stops and cancels orders"* (leaves position open). This is exactly
  our plan's "Stop" vs "Flatten & stop" distinction (§3) — confirms both
  actions need clearly different affordances in the UI, not a single button
  with a confirm dialog.
- **The orphan-order-on-hard-kill trap is real and documented**: Hummingbot's
  own docs warn that a non-graceful process kill (SIGKILL, container crash,
  VPS reboot) leaves live orders resting at the venue because cancel-on-exit
  is a *cooperative* shutdown handler with no venue-side dead-man's switch.
  Directly validates this plan's supervisor requirement (§5) to run
  instances as `systemd-run` transient units with `Restart=on-failure`
  **and** to reconcile against venue state on every restart (which our
  engine's `run_id`-based orphan-adoption already does — Layer A of the
  original fleet-dashboard plan) rather than assuming a clean shutdown ever
  happened.
- **Jesse gates live trading behind a paid plugin even in a self-hosted,
  open-source project** — worth noting only as a market signal (live-trading
  reliability/support is where these projects choose to charge), not as
  something to imitate; our single-operator, single-account use case has no
  licensing dimension.
- **OctoBot's "Force refresh" (portfolio) is the right operator-facing
  reconciliation affordance** — *"Triggers a total portfolio re-
  synchronization using exchanges as a reference"* — a manual button
  version of the automatic startup reconcile our engine already does. Worth
  adding as an explicit Accounts-page action (§5.4) for the moments an
  operator suspects drift and doesn't want to wait for the next poll cycle.

### 9.8 QuantConnect / Freqtrade / NautilusTrader — one more confirmation and one concrete upgrade

- **Freqtrade's control vocabulary independently confirms our §3 design.**
  It runs a real three-state machine (`running` / `paused` / `stopped`),
  identical over REST and its Telegram bot, where **pause specifically
  keeps managing existing exits while blocking new entries** — exactly our
  Pause semantics (§3, §8 decision 3), arrived at independently rather than
  copied from us. Worth noting the one gap in their design we should NOT
  copy: pause "resets upon bot restart" (not persisted) — ours should
  persist the latch in the broker store so a supervisor restart doesn't
  silently un-pause a bot the operator deliberately paused.
  QuantConnect, by contrast, has only two states (running/stopped) plus a
  destructive Liquidate, with no "pause new entries, keep exits" verb at
  all — concrete evidence that skipping Pause (as QuantConnect did) is a
  real UX gap operators notice, not a nice-to-have we invented.
- **NautilusTrader's startup reconciliation is the most rigorous version of
  something our DNSE/Binance broker plugins already do less formally, and
  is worth adopting as the target shape for Layer A (engine substrate) of
  the original fleet-dashboard plan.** On every start it pulls order/fill/
  position reports from the venue and reconciles the local cache against
  them: missing fills are synthesized, in-flight orders left ambiguous by a
  crash resolve to explicit terminal states (`SUBMITTED`→`REJECTED` on
  timeout; `PENDING_CANCEL`/`PENDING_UPDATE`→`CANCELED`) rather than staying
  unknown, and a real position-size mismatch is corrected via a **price
  preference ladder** (reconciliation price → market mid → current position
  average → MARKET order as the last resort). Most notable: **if a fill's
  commission can't be calculated, the whole reconciliation fails closed
  rather than silently dropping that fill** — startup blocks instead of
  starting with wrong state. This is a stronger, more explicit version of
  the DNSE plugin's existing orphan-adoption / phantom-shell handling
  (`plugins/dnse/pynecore_dnse/broker.py`, `venue.py`'s phantom-shell
  separation) and the engine's `run_id`-based orphan adoption (§4 of this
  plan) — worth reviewing those against Nautilus's explicit-terminal-state
  and fail-closed-on-uncertain-fill rules as a concrete hardening pass, not
  a redesign.
- **QuantConnect's restart-loop guard is the right shape for our systemd
  policy (§5/§6):** automatic restart only re-engages if the process had
  been up **≥5 minutes** before the failure, capped at **5 total attempts**
  — a direct, documented defense against a bad deploy crash-looping and
  spamming the exchange. Translate directly to `systemd-run`'s
  `StartLimitIntervalSec=`/`StartLimitBurst=` rather than a bare
  `Restart=on-failure`.
- **Two documented VPS pitfalls sharpen §6:** Freqtrade's own docs warn that
  **`sd_notify`'s systemd watchdog communication does not work inside
  Docker** — i.e. the standard "run in Docker" and "enable the systemd
  crash watchdog" advices are mutually defeating unless the health-check
  path is built without relying on `sd_notify` (favor an HTTP/socket
  liveness probe from outside the container instead, which §6's watchdog
  design already assumes). And QuantConnect's own operator warning —
  *"avoid manipulating your brokerage account and placing manual orders...
  while your algorithm is running"* — is independent, cross-platform
  confirmation of exactly the DNSE venue fact already in this repo's memory
  ([[dnse-conditional-cancel-session-binding]]: the operator's own
  Smart-OTP app trade breaks the bot's conditional writes). Worth stating
  in the platform's operator runbook as a general rule, not a DNSE quirk:
  **never hand-trade an account a live instance is also trading.**
- **Composer.trade, Kryll.io, and StockSharp's live-UX are gated behind
  their logged-in apps** — no public docs describe pause/edit/risk-control
  UI for the first two, and StockSharp reads as desktop-IDE-first with no
  evidence of a comparable web dashboard. Not worth further research spend;
  none of the three offered a pattern the sources above didn't already cover
  more thoroughly.

### 9.9 Sources

Full verbatim findings, parameter tables, and URL lists for all of the
above are preserved in the session's research notes (Pine Script v6 docs,
TradingView Help Center — noting the "Strategy properties" Help Center
article is **stale** post-2026-redesign, live-UI confirmation is
authoritative over it; TradersPost/PineConnector/Alertatron/3Commas docs;
Hummingbot/Jesse/OctoBot/QuantConnect/Freqtrade/NautilusTrader official
docs and source). Re-derive on demand rather than treating this summary as
exhaustive — cite the primary sources directly if a specific number or
label needs to go in front of the user.
