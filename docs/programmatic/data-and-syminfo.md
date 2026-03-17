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
