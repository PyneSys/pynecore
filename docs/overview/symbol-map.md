<!--
---
weight: 108
title: "Symbol Map"
description: "Translate TradingView-canonical symbols to your provider-native data with a workdir-level symbol map"
icon: "swap_horiz"
date: "2026-07-24"
lastmod: "2026-07-24"
draft: false
toc: true
categories: ["Overview", "Configuration"]
tags: ["symbol-map", "request-security", "configuration", "toml", "workdir", "providers"]
---
-->

# Symbol Map

The **symbol map** is an optional, workdir-level translation table that maps the
TradingView-canonical symbols your script writes (`NASDAQ:AAPL`,
`BINANCE:BTCUSDT`) to the provider-native symbols your own data files carry
(`capitalcom:AAPL`, `ccxt:BYBIT:BTC/USDT:USDT`). It lets both backtest and live
runs find the right data for a [`request.security()`](../lib/request-security.md)
context **without** an explicit `--security` mapping on every call.

## The problem it solves

Pyne code is written to be TradingView-faithful, so a script references symbols
the way TradingView spells them:

```python
spx: Series[float] = lib.request.security("SP:SPX", "1D", lib.close)
```

But the data on your machine came from *your* data provider, and that provider
spells the symbol differently — Capital.com calls it `US500`, a CCXT broker uses
`BYBIT:BTC/USDT:USDT`, and so on. Without a translation step you would have to
pass an explicit `--security "SP:SPX=capitalcom_US500_1D"` for every context, in
every run.

The symbol map declares those translations **once**, in one file, and every run
picks them up automatically.

## File location and schema

The map lives in a single TOML file inside your working directory's config
folder:

```
workdir/config/symbol_map.toml
```

It contains one `[symbol_map]` table of `KEY = "VALUE"` pairs:

```toml
[symbol_map]
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
"NASDAQ:AAPL"     = "capitalcom:AAPL"
"SP:SPX"          = "capitalcom:US500"

# Optional per-timeframe override (see below)
"NASDAQ:AAPL:60"  = "capitalcom:AAPL"
```

### Key — the TradingView symbol

The **key** is the TradingView-style `PREFIX:TICKER` exactly as written in the
script, optionally suffixed with `:TIMEFRAME` for a per-timeframe override.

- `"NASDAQ:AAPL"` — matches any `request.security("NASDAQ:AAPL", …)` call
- `"NASDAQ:AAPL:60"` — matches only the same symbol at the `60`-minute timeframe

### Value — the provider-native symbol

The **value** is a provider-qualified native symbol in the same
`provider:rest` shape used by the `pyne data download` provider string, where
`rest` may itself contain further colons:

| Value                          | Provider     | Native symbol           |
|--------------------------------|--------------|-------------------------|
| `"capitalcom:AAPL"`            | `capitalcom` | `AAPL`                  |
| `"ccxt:BYBIT:BTC/USDT:USDT"`   | `ccxt`       | `BYBIT:BTC/USDT:USDT`   |

The value is **not** a filename. For backtests the expected `.ohlcv` path is
derived deterministically from the provider's own naming rules (each provider's
`get_ohlcv_path`), so per-provider file-naming conventions are honored
automatically.

## Precedence

When a `request.security()` context needs data, resolution consults these
sources in order — the **first hit wins**:

1. **Explicit `--security` mapping** (or the programmatic `security_data` dict) —
   an exact `SYMBOL:TF`, then `SYMBOL`, then `TF` key.
2. **The chart's own feed** — when the security symbol is the chart symbol
   (same symbol at a coarser timeframe, or a `ticker.heikinashi()` chart-type
   request), it is served from the chart data / live stream directly.
3. **The global symbol map** — `config/symbol_map.toml`.
4. **Provider identity fallback (live only)** — the Pine symbol is forwarded to
   the live provider unchanged, on the assumption it is already provider-native.

So an explicit `--security` mapping always overrides the symbol map, and the
symbol map overrides the bare identity fallback.

### Timeframe override

Within the map, a `"SYMBOL:TF"` entry beats a bare `"SYMBOL"` entry for that
timeframe — mirroring the `--security` key-matching rules. This lets you point a
single symbol at different native data per timeframe when needed:

```toml
[symbol_map]
"NASDAQ:AAPL"    = "capitalcom:AAPL"     # default for every timeframe
"NASDAQ:AAPL:1D" = "ccxt:SOMEDEX:AAPL"   # but daily comes from elsewhere
```

## Backtest vs. live behavior

The map is consulted in both modes, but what it produces differs:

- **Backtest** — the mapped provider + native symbol + timeframe are turned into
  the expected `.ohlcv` file path via the provider plugin's `get_ohlcv_path`.
  If that file exists it is used; if the symbol is mapped but the derived file is
  missing, the run fails with a clear error and a ready-to-run download command
  (unless the context set `ignore_invalid_symbol`):

  ```
  Security 'SP:SPX' @ '1D' is mapped to capitalcom:'US500' by
  config/symbol_map.toml, but the derived data file capitalcom_US500_1D.ohlcv
  was not found in <workdir>/data. Download it with:
  pyne data download 'capitalcom:US500@1D'
  ```

- **Live** — the mapped native symbol is used to construct the `PluginSymbol`
  that the security subprocess streams. The per-plugin `config.symbol_map` (the
  provider plugin's own TOML translation table) is consulted **first**, then the
  global map.

## Cross-provider limitation (live mode)

In **backtest** mode a map entry may name any provider — the correct `.ohlcv`
file is derived from that provider's naming rules regardless of which provider
produced the chart data.

In **live** mode, multi-provider resolution is not supported yet: a global-map
entry whose provider differs from the running chart provider is **skipped with a
warning**, and resolution falls through to the identity fallback.

```
Skipping global symbol_map entry 'BINANCE:BTCUSDT' -> 'ccxt:BYBIT:BTC/USDT:USDT':
it targets provider 'ccxt' but the running provider is 'capitalcom'
(multi-provider resolution is not supported yet).
```

## Robustness

The map is designed never to break a run on its own:

- A **missing** `symbol_map.toml` yields an empty map (nothing is translated).
- An **unreadable file** or a **malformed `[symbol_map]` table** degrades to an
  empty map; the parse error is logged as a warning.
- An **individual malformed entry** (a non-string value, or a value with no
  provider/native-symbol) is skipped with a warning; the rest of the map still
  loads.

## Relationship to other mappings

| Mechanism                 | Scope                        | Where                                   |
|---------------------------|------------------------------|-----------------------------------------|
| `--security` flag         | Per-run, per-context         | CLI / `security_data` dict              |
| Global symbol map         | Per-workdir, all runs        | `config/symbol_map.toml`                |
| Per-plugin `symbol_map`   | Per-provider (live)          | The provider plugin's own config TOML   |

Use `--security` for one-off overrides, the per-plugin map for provider-specific
naming quirks, and the global symbol map as the durable, script-agnostic
translation layer between TradingView symbols and your local data.

## See also

- [request.security()](../lib/request-security.md) — multi-symbol / multi-timeframe data
- [Configuring PyneCore](./configuration.md) — the workdir and its config files
- [Live Mode](../advanced/live-mode.md) — running scripts against a live provider
