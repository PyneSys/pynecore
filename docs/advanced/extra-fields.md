<!--
---
weight: 1005
title: "Extra Fields"
description: "Accessing extra per-bar columns beyond OHLCV data in Pyne code"
icon: "playlist_add"
date: "2025-03-31"
lastmod: "2026-07-28"
draft: false
toc: true
categories: ["Advanced", "Data Handling"]
tags: ["extra-fields", "csv", "custom-data", "series", "data"]
---
-->

# Extra Fields

PyneCore allows you to access additional columns beyond standard OHLCV data from your CSV files inside Pyne code. This is useful when your data includes pre-computed indicators, signals, or any other custom data that you want to use alongside price data.

## How It Works

When a CSV file contains columns beyond the standard OHLCV fields (`timestamp`, `open`, `high`, `low`, `close`, `volume`), PyneCore automatically makes them available through `extra_fields` — a dictionary that is updated on each bar with the current row's extra column values.

The data flow depends on where the data comes from and how you run your script:

### Binary OHLCV Path (workdir)

When running from a workdir with `pyne run`, PyneCore converts CSV to binary `.ohlcv` format. The binary record holds only the OHLCV columns declared by the file header, so extra columns are saved to a **sidecar file** (`.extra.csv`) that is position-aligned with the binary data:

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

### Downloaded Data (providers)

Some data providers deliver more than plain OHLCV per bar — Capital.com returns the ask-side OHLC alongside the bid bars, cTrader reconstructs it from ask tick history. `pyne data download` fetches those values only when asked for them:

```bash
pyne data download capitalcom:EURUSD@60 --extra-data
```

They are written to the very same `.extra.csv` sidecar as user CSV extras, and are read back into `extra_fields` in exactly the same way. The flag is off by default because extra fields cost extra requests while downloading and slow every later backtest that loads the sidecar; providers that have no extra per-bar fields (CCXT, TradingView, Bybit, Coinbase) ignore it.

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
- **Supported types**: Numbers and text, detected automatically from the data. `CSVReader` converts each cell on its own — `int` when it parses as an integer, `float` when it parses as a decimal, otherwise the raw `str`. The binary + sidecar path classifies each column once, from its first usable value: numeric columns come back as `float`, everything else as `str`.
- **Missing values**: Read via `CSVReader`, an empty cell appears as an empty string (`''`) and a `NaN`/`na`/`nan` cell as `na`. Read through the sidecar, an empty, `nan` or `na` cell in a numeric column becomes `NaN`, while in a text column the raw text is kept (an empty string for an empty cell).
- **No AST magic needed**: The standard Series annotation mechanism handles everything — there is no special treatment for `extra_fields` in the AST transformers.

## CSV Format

Your source CSV simply includes extra columns alongside the standard OHLCV columns:

```csv
timestamp,open,high,low,close,volume,rsi,signal,custom_price
2024-01-01T00:00:00,100.0,105.0,95.0,102.0,1000,45.2,buy,99.5
2024-01-01T01:00:00,102.0,108.0,100.0,106.0,1200,52.1,,101.3
2024-01-01T02:00:00,106.0,110.0,104.0,108.0,800,38.7,sell,
```

The following column names are recognized as standard OHLCV when the source is converted to `.ohlcv`, and will **not** appear in `extra_fields`:

| Recognized OHLCV columns                                       |
|----------------------------------------------------------------|
| `timestamp`, `time`, `date`, `datetime`, `ts_event`, `ts_recv` |
| `open`, `high`, `low`, `close`, `volume`                       |

Matching is case-insensitive and ignores surrounding whitespace; any other column name is treated as an extra field. Reading a CSV directly with `CSVReader` is narrower — there only `time` or `timestamp` is taken as the timestamp column, so a `date` column would show up among the extra fields on that path.

## Sidecar File Format

The auto-generated `.extra.csv` file contains only the extra columns, with exactly one data row per bar stored in the `.ohlcv` file. The binary format stores real bars only — a missing time interval produces no record and therefore no sidecar row — so data row `N` always describes record `N`:

```csv
rsi,signal,custom_price
45.2,buy,99.5
52.1,,101.3
38.7,sell,
42.0,hold,100.0
```

A cell is empty when that bar has no value for the column. A whole row can be empty as well: the sidecar starts at the first bar that carries extra fields, and records committed before it are padded with empty rows to keep row `N` on record `N`. This happens, for example, when a file is first downloaded without extra fields and they are requested only later.

The alignment is enforced, not assumed: opening the OHLCV file fails if the sidecar's data-row count differs from the binary file's record count, or if a row has a different number of columns than the header.

## Limitations

- **Extra fields live outside the binary**: The `.ohlcv` file stores only the columns declared in its header, and everything the writer produces is a standard OHLCV column; extra fields are kept in the sidecar CSV. The record layout comes from that header (a 64-byte fixed header plus a 24-byte descriptor per column), so the record size follows the schema instead of being one fixed number for every file — the standard profile is 36 bytes per record, 48 when high/low/close have to be stored as absolute float64.
- **JSON source files**: Extra field extraction is currently supported for CSV and TXT source formats, not JSON. Converting a JSON source deletes any sidecar left over from an earlier conversion.
- **Memory**: The sidecar is loaded entirely into memory when opening the OHLCV file. For typical datasets (up to a few hundred thousand bars with a handful of extra columns), this is negligible.
- **Provider extras are opt-in**: Providers that have extra per-bar fields deliver them only with `--extra-data`; providers without them (CCXT, TradingView, Bybit, Coinbase) never produce a sidecar.

## See Also

- [OHLCV Reader/Writer](./ohlcv-reader-writer.md) — Binary OHLCV format details
- [CSV Reader/Writer](./csv-reader-writer.md) — CSV processing internals
