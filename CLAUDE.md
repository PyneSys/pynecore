# pynecore — repo notes for Claude

> This file is force-tracked (`git add -f`) past the repo's `.gitignore` so that
> cloud/sandbox sessions — which only receive tracked files — still get these rules.
> Upstream deliberately ignores CLAUDE.md and will never modify it, so rebases don't
> conflict. Keep secrets out: this file is public in the repo.

## If you are running in a CLOUD SANDBOX (claude.ai/code, remote agent)

Only tracked files exist there. Consequences:

- **The pine2pyne transpiler installs from PyPI** — `pip install opencode-pine2pyne`,
  then `python -m pine2pyne in.pine -o out.py`. (Transpiled `.py` files are also
  committed next to every `.pine`.) Still NEVER `pyne compile` / `pyne run x.pine` —
  that is the CLOUD compiler and needs an API key we don't have.
- **OHLCV data IS tracked** in `workdir/data/` (force-added `.ohlcv` + `.toml` pairs:
  dnse VN30F1M @1/3/5/15/1D, dnsebroker @1/3/5, HPG 1D, a ccxt BTC sample, replay
  DEMO). Backtests therefore work offline in FILE MODE — pass the dataset name, e.g.
  `pyne run script.py dnse_VN30F1M_1`. Provider mode (`dnse:VN30F1M@1`) still needs
  API credentials and re-downloads, so avoid it in the sandbox. Note the tracked
  files are snapshots from the dev machine; a provider-mode run there rewrites them.
- **Credentials and the DNSE trading token are absent** → anything live (`--broker`,
  the L0 gate) is IMPOSSIBLE. What works offline: `pytest plugins/dnse/tests/`
  (incl. the fake-venue e2e), transpiling, and the staged probes in backtest/oracle
  mode over the tracked data.
- **The dev machine's memory/notes are absent.** The durable venue facts (cancel-ACK,
  no cascade, amend-500, GTD clamp, session phases) live in
  `plugins/dnse/testing/live_test/README.md` — read it before touching the plugin.

## Conventions (distilled from the dev machine's global rules)

- **Never delete** — move unwanted files to `backup/deleteable/` and untrack
  (`git add -A <path>` to stage the deletion); `backup/` is gitignored.
- **Back up before destructive commands** on files with uncommitted changes.
- **Verify you are testing NEW code** — editable installs must resolve to the repo
  (`python -c "import pynecore; print(pynecore.__file__)"`), not site-packages.
- **Fork policy:** `main` = upstream (PyneSys/pynecore) + a linear fork stack,
  maintained by REBASE, never merge. Dist name is `opencode-pyneruntime`; the import
  stays `pynecore`.
- Tests use `__test_*__` naming (`pytest.ini`); full-suite runs need
  `--ignore=tests/t00_pynecore/ast/test_045_lib_import_normalizer_invalid_alias.py`
  (a `@pyne` file that self-triggers at collection — long-standing upstream).

## Pine → Python: use the LOCAL pine2pyne transpiler, NOT the cloud API

To compile a `.pine` strategy to the `.py` that `pyne run` executes, use the
**local pine2pyne transpiler**. Do NOT use `pyne compile` or `pyne run script.pine`
— those call the PyneSys **cloud** compiler and require an API key we don't have
(`workdir/config/api.toml` is empty, and `PYNESYS_API_KEY` is unset).

pine2pyne is a separate repo with its own venv:

```bash
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne /abs/path/to/in.pine -o /abs/path/to/out.py
# globs work too: .venv/bin/python -m pine2pyne "sample/pinescript/*.pine" -o workdir/scripts/
```

Then run the transpiled `.py` (no key needed):

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/pyne run path/to/out.py <data> [--broker]     # e.g. data: dnse:VN30F1M@5
```

Notes:
- The tracked `*.py` test strategies can be **stale** — after editing a `.pine`,
  re-transpile it (they were not regenerated automatically).
- `pyne run` auto-compiles a `.pine` via the cloud API — avoid it; transpile
  locally first, then run the `.py`.

## NEVER use `pyne run` with `--from` (CRITICAL)

Always run **without** `--from`:

```bash
.venv/bin/pyne run path/to/out.py dnse:VN30F1M@1 [--broker]    # correct
.venv/bin/pyne run path/to/out.py dnse:VN30F1M@1 --from -30    # NEVER
```

Why: in provider mode the warmup download **truncates and rewrites the shared
`.ohlcv` file** (`cli/commands/run.py:611`). A run with `--from -30` therefore
*destroys* the accumulated local history for that `(provider, symbol, timeframe)`,
leaving only 30 bars — the next run must re-download, and any deeper history is
gone. `--from` is data-destructive, not a read-only window.

It also silently starves indicators: warmup replays exactly the bars `--from`
fetched (no lookback introspection — `max_bars_back` is a no-op stub), so
`ta.sma(close, 600)` under a small `--from` is `NaN`, every comparison is
`false`, and the strategy quietly never trades while the run looks healthy.

Omitting `--from` uses the built-in default (`-500` real bars, gap-retried up to
4×), which is both safe for the cache and deep enough for most indicators. If a
script genuinely needs more than 500 bars of lookback, raise the default rather
than reaching for `--from`, and gate the strategy on `not na(<series>)` so
insufficient warmup fails loudly instead of silently.

## Plugins in this repo (`plugins/`)

Fork-specific venue plugins, editable-installed (so they import as
`pynecore_<name>`) and discovered via the `pyne.plugin` entry-point group:

- **`plugins/dnse/`** — DNSE (Vietnamese broker) plugin. Two entry points:
  `dnse` (`DNSEProvider` — OHLCV history + metadata) and `dnse_broker`
  (`DNSEBroker` — native STOP/OCO conditional orders). REST-only, built on the
  **vendored** DNSE openapi-sdk v2.0.0 at `plugins/dnse/pynecore_dnse/_vendor/dnse/`
  (do NOT pip-install the SDK). Run: `pyne run <out>.py dnse:VN30F1M@5` for data,
  `… dnse_broker:VN30F1M@5 --broker` for live orders. Tests: `plugins/dnse/tests/`
  — `pytest` with functions named `__test_*__` (see `pytest.ini`); mock via the
  fake-client seam in `conftest.py` — pytest itself NEVER hits the live venue.
  Live testing exists but is its own gated suite (see "DNSE testing" below). Docs mirror + sync tool:
  `docs/dnse-openapi-documentation/` (`fetch_docs.py`); plans in `docs/plan/`.

## DNSE testing (read before touching the plugin or running anything live)

The test suite and its rules are documented in
`plugins/dnse/testing/live_test/README.md` — four types: `pytest plugins/dnse/tests/`
(unit + fake-venue e2e), the L0 venue-semantics gate (MANDATORY, exit 0, before EVERY
live run), the staged no-fill probe (T1–T13 + the `run_t10_dual.sh` dual-strategy
runner), and the staged fill test (F1–F8; its backtest mode over a past window IS the
oracle). Both staged probes are driven by `winStart`/`winEnd`/`startState` in their
`.toml` — for a live run `winStart` must be AFTER launch or warmup consumes the stages.
Grade live results from the VENUE record, never the run log alone. Measured venue facts
(cancel-ACK, no cascade, amend-500, GTD clamp, session phases) are listed in that README.
Trading-token workflow (OTP mint, ~8h TTL, status check): `plugins/dnse/tools/README.md`
— live runs need a GOOD token first (`tools/token_status.py`).

## Pine Script language reference

Authoritative Pine v6 reference (syntax, built-ins, `strategy.*` semantics):
https://www.tradingview.com/pine-script-reference/v6/
Use it when writing test `.pine` files or checking what TradingView-compatible
behaviour SHOULD be (e.g. `strategy.exit` has no `oca_type`; exits form a reduce
group; OCA fires on fill, not on cancel).

## Reference repos (workspace siblings — read for patterns, don't edit)

Cloned next to this repo under `/home/mike/workspace/github/`:

- **`pine2pyne/`** — the local Pine→Python transpiler (see above); its own venv.
- **Official PyneSys broker-plugin samples** — the canonical examples the DNSE
  plugin's design was ground-checked against; copy their module split
  (`execution.py` / `activity.py` / `reconcile.py` / `recovery.py`), the
  `store_ctx` persistence + `DisappearanceTracker` patterns, and the
  `broker_lab/` conformance-lab `VenueProfile`:
  - `pynecore-plugin-bybit/` — native `reduceOnly`, bar-close fills
  - `pynecore-plugin-capitalcom/` — deal-id model, `recovery.py` reopen-on-retry
  - `pynecore-plugin-ctrader/` — cTrader
