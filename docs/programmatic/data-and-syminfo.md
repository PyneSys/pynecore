<!--
---
weight: 802
title: "Data & SymInfo"
description: "Loading and creating OHLCV data and symbol information for programmatic use"
icon: "database"
date: "2025-03-31"
lastmod: "2026-03-17"
draft: false
toc: true
categories: ["Programmatic", "Data"]
tags: ["ohlcv", "syminfo", "data-converter", "csv", "custom-data"]
---
-->

# Data & SymInfo

PyneCore needs two things to run a script: **OHLCV data** (candles) and **SymInfo** (symbol
metadata). This page covers all the ways to provide them.

## OHLCV Data

### The OHLCV Type

Every candle in PyneCore is an `OHLCV` namedtuple:

```python
from pynecore.types.ohlcv import OHLCV

candle = OHLCV(
    timestamp=1704067200,   # Unix epoch in SECONDS (not milliseconds!)
    open=42000.0,
    high=42500.0,
    low=41800.0,
    close=42300.0,
    volume=1000.0,
)
```

> **Important:** Timestamps are in **seconds**. Many exchange APIs (CCXT, Binance) return
> milliseconds — divide by 1000.

### Option 1: From a CSV File

Use `DataConverter` to convert CSV data to PyneCore's binary OHLCV format:

```python
from pathlib import Path
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVReader

csv_path = Path("data/BTCUSD_1h.csv")

# Convert CSV → .ohlcv binary + .toml metadata
DataConverter().convert_to_ohlcv(csv_path)

# Read the converted data
ohlcv_path = csv_path.with_suffix(".ohlcv")
with OHLCVReader(ohlcv_path) as reader:
    for candle in reader.read_from(reader.start_timestamp, reader.end_timestamp):
        print(candle.close)
```

The converter automatically detects:
- Column mapping (timestamp, open, high, low, close, volume)
- Timezone from timestamps (DST-aware)
- Tick size, trading hours, symbol type

### Option 2: Create OHLCV Objects Directly

For custom data sources (APIs, databases, websockets), create OHLCV objects directly:

```python
from pynecore.types.ohlcv import OHLCV

# From a REST API
def fetch_from_api():
    response = requests.get("https://api.exchange.com/ohlcv/BTCUSD/1h")
    for bar in response.json():
        yield OHLCV(
            timestamp=bar["time"],           # must be seconds
            open=bar["o"], high=bar["h"],
            low=bar["l"], close=bar["c"],
            volume=bar["v"],
        )

# From a pandas DataFrame
def from_dataframe(df):
    for row in df.itertuples():
        yield OHLCV(
            timestamp=int(row.Index.timestamp()),
            open=row.open, high=row.high,
            low=row.low, close=row.close,
            volume=row.volume,
        )

# From a database
def from_database(cursor):
    cursor.execute("SELECT ts, o, h, l, c, vol FROM candles ORDER BY ts")
    for row in cursor:
        yield OHLCV(timestamp=row[0], open=row[1], high=row[2],
                     low=row[3], close=row[4], volume=row[5])
```

`ScriptRunner` accepts any `Iterable[OHLCV]` — lists, generators, and readers all work.

### Option 3: From an Exchange (CCXT)

```python
import ccxt
from pynecore.types.ohlcv import OHLCV

exchange = ccxt.binance({"enableRateLimit": True})
raw = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=200)

candles = [
    OHLCV(
        timestamp=bar[0] // 1000,  # CCXT returns milliseconds!
        open=bar[1], high=bar[2], low=bar[3], close=bar[4], volume=bar[5],
    )
    for bar in raw
]
```

## SymInfo (Symbol Information)

SymInfo tells PyneCore about the financial instrument — currency, tick size, timezone, market type,
etc. Scripts access this via `syminfo.*` (e.g., `syminfo.mintick`, `syminfo.currency`).

### Option 1: Load from TOML

When you convert a CSV file, a `.toml` file is automatically generated:

```python
from pynecore.core.syminfo import SymInfo

syminfo = SymInfo.load_toml(Path("data/BTCUSD_1h.toml"))
```

### Option 2: Create Manually

For custom data sources, build SymInfo by hand:

```python
from pynecore.core.syminfo import SymInfo

syminfo = SymInfo(
    prefix="BINANCE",             # exchange/provider name
    description="Bitcoin / USD",  # human-readable name
    ticker="BTCUSD",             # symbol ticker
    currency="USD",              # quote currency
    basecurrency="BTC",          # base currency
    period="60",                 # timeframe: "1", "5", "15", "60", "D", "W", "M"
    type="crypto",               # "stock", "forex", "crypto", "futures", "index"
    mintick=0.01,                # smallest price increment
    pricescale=100,              # 1 / mintick
    minmove=1,                   # minimum price movement in pricescale units
    pointvalue=1.0,              # profit per 1 unit price move per 1 contract
    timezone="UTC",              # IANA timezone (e.g., "America/New_York")
    volumetype="base",           # "base", "quote", "tick", "n/a"
    opening_hours=[],            # trading session hours (empty for 24/7 crypto)
    session_starts=[],           # session start times
    session_ends=[],             # session end times
)
```

### Common SymInfo Configurations

**Crypto (24/7 trading):**

```python
SymInfo(
    prefix="BINANCE", description="BTC / USDT", ticker="BTCUSDT",
    currency="USDT", basecurrency="BTC", period="60",
    type="crypto", mintick=0.01, pricescale=100, minmove=1, pointvalue=1.0,
    timezone="UTC", volumetype="base",
    opening_hours=[], session_starts=[], session_ends=[],
)
```

**Forex:**

```python
SymInfo(
    prefix="FX", description="EUR / USD", ticker="EURUSD",
    currency="USD", basecurrency="EUR", period="60",
    type="forex", mintick=0.0001, pricescale=10000, minmove=1, pointvalue=1.0,
    timezone="America/New_York", volumetype="tick",
    opening_hours=[], session_starts=[], session_ends=[],
)
```

**US Stocks:**

```python
SymInfo(
    prefix="NASDAQ", description="Apple Inc.", ticker="AAPL",
    currency="USD", period="D",
    type="stock", mintick=0.01, pricescale=100, minmove=1, pointvalue=1.0,
    timezone="America/New_York", volumetype="base",
    opening_hours=[], session_starts=[], session_ends=[],
)
```

### Effective-Dated Session Schedules

Markets occasionally change their trading hours — a futures contract shortens its
night session, an exchange shifts an open by half an hour, and so on. The fields
above describe **one** static schedule. If a backtest range spans such a change,
that single schedule confirms the bars on the *other* side of the change against
the wrong hours, producing a small, quiet divergence on exactly the dates around
the change.

The optional `session_schedules` field fixes this. It is an **effective-dated
history**: a list of schedule *variants*, each in effect from a given date until
the next variant's date. It is entirely **opt-in** — leave it empty (the default)
and the symbol keeps its single flat schedule, with no per-era refinement. When you
do have the historical hours (from exchange notices,
your data vendor, or your own records), add them by hand and every bar is
confirmed against the schedule that was actually in effect on its trading day.

> **Scope:** in the current release a history affects **session-bounded higher-
> timeframe confirmation** — i.e. how `request.security()` confirms an intraday HTF
> bar that is bounded by its trading session (see
> [request.security() Internals](../advanced/request-security-internals.md)).
> A symbol without a history confirms every such bar against its single flat
> schedule; the history only refines *which* schedule applies on each side of a
> change.

#### Resolution rules

- Each variant has an `effective_from` date (the exchange-local **trading-day**
  date the new hours took effect) and its own `opening_hours` / `session_starts` /
  `session_ends`.
- A bar is matched to the **last** variant whose `effective_from` is on or before
  the bar's trading day. A bar before the earliest variant uses the earliest one.
- The trading-day key matters for overnight sessions: a night bar that opens the
  evening before belongs to the **next** trading day, so it is confirmed against
  that day's variant — not the calendar date of its open.
- The flat `opening_hours` / `session_starts` / `session_ends` always mirror the
  **newest** variant. When a history is present they are regenerated from it on
  save, so **edit the variants, not the flat block** — the flat block is derived.

#### In TOML

A `.toml` written for a symbol *without* a history includes a commented-out
example showing the exact layout. Uncomment and adapt it, or write the blocks
directly. List variants oldest first; `effective_from` must be the **first line**
of each block (an unquoted date), before its nested tables:

```toml
# A futures contract whose night session END moved 23:30 -> 23:00 on 2026-01-12.

[[session_schedules]]
effective_from = 2025-06-01
[[session_schedules.opening_hours]]
day = 0
start = "10:00:00"
end = "18:00:00"
[[session_schedules.opening_hours]]
day = 0
start = "21:00:00"
end = "23:30:00"
[[session_schedules.session_starts]]
day = 0
time = "10:00:00"
[[session_schedules.session_ends]]
day = 0
time = "23:30:00"

[[session_schedules]]
effective_from = 2026-01-12
[[session_schedules.opening_hours]]
day = 0
start = "10:00:00"
end = "18:00:00"
[[session_schedules.opening_hours]]
day = 0
start = "21:00:00"
end = "23:00:00"
[[session_schedules.session_starts]]
day = 0
time = "10:00:00"
[[session_schedules.session_ends]]
day = 0
time = "23:00:00"
```

`day` is the weekday of the session open (`0` = Monday … `6` = Sunday); an
overnight interval is one whose `end` is at or before its `start`. `effective_from`
also accepts a quoted `"YYYY-MM-DD"` string. Duplicate `effective_from` dates are
rejected on load.

#### In Python

Build the same history programmatically with `SymInfoScheduleVariant`:

```python
from datetime import date, time
from pynecore.core.syminfo import (
    SymInfo, SymInfoInterval, SymInfoSession, SymInfoScheduleVariant,
)

def night_variant(effective_from, night_end):
    return SymInfoScheduleVariant(
        effective_from=effective_from,
        opening_hours=[
            SymInfoInterval(day=d, start=time(10, 0), end=time(18, 0)) for d in range(5)
        ] + [
            SymInfoInterval(day=d, start=time(21, 0), end=night_end) for d in range(5)
        ],
        session_starts=[SymInfoSession(day=d, time=time(10, 0)) for d in range(5)],
        session_ends=[SymInfoSession(day=d, time=night_end) for d in range(5)],
    )

variants = [
    night_variant(date(2025, 6, 1), time(23, 30)),   # old hours
    night_variant(date(2026, 1, 12), time(23, 0)),   # new hours
]

syminfo = SymInfo(
    prefix="EXCH", description="Palm Oil Future", ticker="FCPO1!",
    currency="MYR", period="720", type="futures",
    mintick=1.0, pricescale=1, minmove=1, pointvalue=25.0, mincontract=1.0,
    timezone="Asia/Kuala_Lumpur", volumetype="base",
    # The flat fields mirror the newest variant; the history is the source of truth.
    opening_hours=variants[-1].opening_hours,
    session_starts=variants[-1].session_starts,
    session_ends=variants[-1].session_ends,
    session_schedules=variants,
)
```

`SymInfo` exposes `has_schedule_history`, `schedule_index_for(date)` and
`schedule_for(date)` to inspect which variant applies on a given date.

### Period Values

The `period` field uses the same values as TradingView's Pine Script `timeframe.period`:

| Timeframe | Period value |
|-----------|-------------|
| 1 minute  | `"1"`       |
| 5 minutes | `"5"`       |
| 15 minutes | `"15"`     |
| 30 minutes | `"30"`     |
| 1 hour    | `"60"`      |
| 4 hours   | `"240"`     |
| Daily     | `"D"`       |
| Weekly    | `"W"`       |
| Monthly   | `"M"`       |
