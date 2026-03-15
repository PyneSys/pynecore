<!--
---
weight: 1005
title: "Extra Fields"
description: "Accessing custom CSV columns beyond OHLCV data in Pyne scripts"
icon: "playlist_add"
date: "2025-03-31"
lastmod: "2026-03-15"
draft: false
toc: true
categories: ["Advanced", "Data Handling"]
tags: ["extra-fields", "csv", "custom-data", "series", "data"]
---
-->

# Extra Fields

PyneCore allows you to access additional columns beyond standard OHLCV data from your CSV files inside Pyne scripts. This is useful when your data includes pre-computed indicators, signals, or any other custom data that you want to use alongside price data.

## How It Works

When a CSV file contains columns beyond the standard OHLCV fields (`timestamp`, `open`, `high`, `low`, `close`, `volume`), PyneCore automatically makes them available through `extra_fields` — a dictionary that is updated on each bar with the current row's extra column values.

The data flow depends on how you run your script:

### Binary OHLCV Path (workdir)

When running from a workdir with `pyne run`, PyneCore converts CSV to binary `.ohlcv` format. Since the binary format only stores OHLCV data, extra columns are saved to a **sidecar file** (`.extra.csv`) that is position-aligned with the binary data:

```
workdir/data/my_data/
    EURUSD_1h.csv          # Source: OHLCV + extra columns
    EURUSD_1h.ohlcv        # Binary OHLCV (auto-generated)
    EURUSD_1h.toml         # Symbol metadata (auto-generated)
    EURUSD_1h.extra.csv    # Extra columns only (auto-generated)
```

The sidecar file is generated and regenerated automatically whenever the source CSV is converted. You never need to create or edit it manually.

### Direct CSV Path (CSVReader / standalone)

When reading CSV directly (e.g., via `CSVReader` or standalone execution), extra columns are parsed inline — no sidecar file is needed.

## Usage in Scripts

Access extra fields through `lib.extra_fields`, which is a `dict[str, Any]` updated each bar:

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, ta, close, extra_fields, plot


@script.indicator(title="Extra Fields Example", overlay=True)
def main():
    # Access extra columns as Series by annotating with Series[T]
    rsi: Series[float] = extra_fields["rsi"]
    signal: Series[str] = extra_fields["signal"]

    # Series indexing works — access previous bars
    prev_rsi = rsi[1]       # Previous bar's RSI value
    rsi_2_ago = rsi[2]      # RSI from 2 bars ago

    # Use with built-in functions like any other Series
    rsi_sma = ta.sma(rsi, 14)

    # Use string fields for conditional logic
    if signal[0] == "buy":
        plot(close, "Buy Signal", linewidth=2)
```

### Key Points

- **Type annotation creates the Series**: Writing `rsi: Series[float] = extra_fields["rsi"]` makes `rsi` a proper Series with history. The `extra_fields["rsi"]` part just returns the current bar's value (a plain `float`).
- **Supported types**: `float`, `int`, `str`, and `bool`. The type is detected automatically from the CSV data.
- **Missing values**: Empty cells in the CSV appear as empty string (`''`) when read via CSVReader, or `NaN` when read via the binary OHLCV + sidecar path.
- **No AST magic needed**: The standard Series annotation mechanism handles everything — there is no special treatment for `extra_fields` in the AST transformers.

## CSV Format

Your source CSV simply includes extra columns alongside the standard OHLCV columns:

```csv
timestamp,open,high,low,close,volume,rsi,signal,custom_price
2024-01-01T00:00:00,100.0,105.0,95.0,102.0,1000,45.2,buy,99.5
2024-01-01T01:00:00,102.0,108.0,100.0,106.0,1200,52.1,,101.3
2024-01-01T02:00:00,106.0,110.0,104.0,108.0,800,38.7,sell,
```

The following column names are recognized as standard OHLCV and will **not** appear in `extra_fields`:

| Recognized OHLCV columns                              |
|--------------------------------------------------------|
| `timestamp`, `time`, `date`, `datetime`                |
| `open`, `high`, `low`, `close`, `volume`               |

Any other column name is treated as an extra field.

## Sidecar File Format

The auto-generated `.extra.csv` file contains only the extra columns, with rows aligned 1:1 to the binary `.ohlcv` file (including gap-filled rows):

```csv
rsi,signal,custom_price
45.2,buy,99.5
52.1,,101.3
38.7,sell,
,,
,,
42.0,hold,100.0
```

Empty rows correspond to gap-filled bars in the OHLCV binary (bars with `volume = -1`).

## Limitations

- **Binary format unchanged**: The `.ohlcv` binary format remains fixed at 24 bytes per record. Extra fields are stored separately in the sidecar CSV.
- **JSON source files**: Extra field extraction is currently supported for CSV and TXT source formats, not JSON.
- **Memory**: The sidecar is loaded entirely into memory when opening the OHLCV file. For typical datasets (up to a few hundred thousand bars with a handful of extra columns), this is negligible.
- **Not available from providers**: Data download providers (e.g., CCXT, TradingView) produce standard OHLCV data only. Extra fields are for user-provided CSV data.

## See Also

- [OHLCV Reader/Writer](./ohlcv-reader-writer.md) — Binary OHLCV format details
- [CSV Reader/Writer](./csv-reader-writer.md) — CSV processing internals
