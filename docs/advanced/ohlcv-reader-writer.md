<!--
---
weight: 1003
title: "OHLCV Reader/Writer"
description: "The self-describing OHLCV v2 binary format: declared timeframe, delta-encoded records and memory-mapped access"
icon: "data_object"
date: "2025-03-31"
lastmod: "2026-07-28"
draft: false
toc: true
categories: ["Advanced", "Data Handling"]
tags: ["ohlcv", "data", "performance", "memory-mapping", "binary-format", "io"]
---
-->

# OHLCV Reader/Writer

The PyneCore OHLCV Reader/Writer system stores and reads financial market data in a compact, self-describing binary format. It is pure Python, uses memory mapping for near-native read performance, and answers every metadata question — timeframe, record count, first and last timestamp — from the file header instead of scanning the data.

## Overview

Time series data is at the heart of any trading system. The OHLCV (Open, High, Low, Close, Volume) format is the industry standard for representing price action over a time interval, and how it is stored decides how fast — and how correctly — a backtest can read it.

The current format (v2) is built on four principles:

- **Self-describing files**: the header declares the timeframe, the column layout and the committed record count
- **Real bars only**: a missing interval produces no record; nothing synthetic is ever stored
- **Addressing by data, not by arithmetic**: timestamp lookups are binary searches over the stored timestamps
- **Durable appends**: a record becomes visible only once its bytes are on disk

### One Entry Point, One Timestamp Unit

`pynecore.core.ohlcv` is the only module you need — and the only one you should import:

```python
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, record_count
```

Two rules govern the whole API:

1. **Timestamps are Unix milliseconds, everywhere.** `OHLCV.timestamp`, `start_timestamp`, `end_timestamp` and every `read_from()` argument use milliseconds, whatever the file on disk contains.
2. **The on-disk format is never your concern.** `OHLCVReader` reads both v2 files and files written by older PyneCore versions; it decides once, when the file is opened, and callers never learn which one it was.

## Why v2

The v1 format was a headerless array of 24-byte records: a `uint32` timestamp in *seconds* followed by five `float32` values. It had no header at all, and that single design choice produced a chain of concrete failures.

**Positions were computed, not searched.** A time-range query resolved its start index as `(requested_timestamp - first_timestamp) // interval`. That is only correct if every record sits exactly on a grid anchored at the first record. One off-grid bar — a feed that emits a bar a few seconds late, a session that opens at 09:30 in a 60-minute file, a daily bar shifted by a DST change — and the computed index no longer matches the real one. The error does not correct itself: every record after the irregular bar is off by at least one slot, and each further irregularity adds to the offset. Queries silently return the wrong bars.

**Gaps had to be filled with fake bars.** To keep that arithmetic true, v1 had to materialise a record for every empty slot, marking them with a negative volume. A weekend in a 1-minute FX file meant thousands of phantom candles on disk, and every consumer had to remember to filter them out. Data that was never traded was indistinguishable, structurally, from data that was.

**The timeframe was guessed.** v1 inferred its interval from the difference between the first two records. If those two bars happened to straddle a gap, a session boundary or a single irregular bar, the entire file was addressed with the wrong interval.

**Nothing was declared.** With no header there was no record count, so a write interrupted halfway left a file that looked complete. There was no way to state whether a file was gap-free. And the `uint32` seconds field could not represent sub-second data and ran out in 2106.

v2 fixes each of these at the source: the header **declares** the period instead of inferring it, a writer-verified `DENSE` flag states whether the file is actually gap-free, the authoritative `record_count` bounds the committed data, timestamps are signed 64-bit milliseconds, and positions come from a binary search over the stored timestamp column — so an irregular file is queried exactly as correctly as a regular one, and no phantom record is ever written.

## Binary File Format

A v2 file has three parts:

```
[ 8-byte magic ][ 64-byte fixed header ][ N x 24-byte column descriptors ][ packed records ]
```

The magic is `\x89PYN\r\n\x1a\n` and occupies the first eight bytes of the fixed header. Everything is little-endian.

### Fixed Header

| Offset | Field           | Type    | Meaning                                                   |
|--------|-----------------|---------|-----------------------------------------------------------|
| 0      | magic           | 8 bytes | `\x89PYN\r\n\x1a\n`                                       |
| 8      | version_major   | uint16  | `2`                                                       |
| 10     | version_minor   | uint16  | `1`; readers accept any equal or older minor              |
| 12     | header_size     | uint32  | `64 + 24 * column_count` (208 for the standard profile)   |
| 16     | record_size     | uint32  | Packed record size in bytes (36 for the standard profile) |
| 20     | column_count    | uint16  | Number of descriptors that follow                         |
| 22     | flags           | uint16  | Bit 0 (`0x0001`) is `DENSE`                               |
| 24     | record_count    | uint64  | Authoritative number of committed records                 |
| 32     | first_timestamp | int64   | First committed timestamp, Unix milliseconds              |
| 40     | last_timestamp  | int64   | Last committed timestamp, Unix milliseconds               |
| 48     | interval_value  | uint32  | Timeframe multiplier                                      |
| 52     | interval_unit   | uint8   | 1=second, 2=minute, 3=hour, 4=day, 5=week, 6=month        |
| 53     | padding         | 3 bytes | Alignment                                                 |
| 56     | minmove         | uint32  | Tick-grid numerator (minor >= 1); `0` = grid unknown      |
| 60     | pricescale      | uint32  | Tick-grid denominator; zero exactly when `minmove` is     |

Opening a file validates all of it: the magic, the version, the flag bits, that `header_size` matches `column_count`, that `record_count` records actually fit in the file, and that `first_timestamp` and `last_timestamp` equal the timestamps physically stored in the first and last committed record. A header that disagrees with its own data is rejected rather than trusted.

### Column Descriptors

Each column is described by a 24-byte descriptor:

| Offset | Field       | Type     | Meaning                                                             |
|--------|-------------|----------|---------------------------------------------------------------------|
| 0      | role        | uint8    | What the column means (see below)                                   |
| 1      | dtype       | uint8    | `2` = int64, `5` = float32, `6` = float64                           |
| 2      | base        | uint8    | `255` = absolute value, otherwise the role this value is a delta of |
| 3      | padding     | 1 byte   | Alignment                                                           |
| 4      | byte_offset | uint16   | Offset of the column inside the packed record                       |
| 6      | name        | 18 bytes | ASCII column name, NUL-padded                                       |

Role `0` is the timestamp and roles `1`-`5` are open, high, low, close and volume; codes up to `13` are reserved for future well-known columns, and `255` marks a named custom column. The descriptors must tile the record exactly — consecutive `byte_offset` values, no gaps, no overlaps, covering precisely `record_size` bytes — and delta chains must terminate at an absolute column without cycles. All of this is checked when the file is opened.

### Record Layout

The standard OHLCV profile stores 36 bytes per record. The columns sit in the file in the order the format's name promises — the timestamp, then open, high, low, close and volume:

| Offset | Column    | Type    | Base     | Meaning                                 |
|--------|-----------|---------|----------|-----------------------------------------|
| 0      | timestamp | int64   | absolute | Unix timestamp in milliseconds          |
| 8      | open      | float64 | absolute | Opening price                           |
| 16     | high      | float32 | `open`   | Highest price, stored as `high - open`  |
| 20     | low       | float32 | `open`   | Lowest price, stored as `low - open`    |
| 24     | close     | float32 | `open`   | Closing price, stored as `close - open` |
| 28     | volume    | float64 | absolute | Trading volume                          |

Record *N* starts at `header_size + N * record_size`, so random access by position is a single offset computation.

### Why float32 Deltas Are Safe

A float32 has a 24-bit mantissa: roughly seven significant decimal digits of **relative** precision. Its absolute resolution therefore depends entirely on the magnitude of the number it stores — and a bar's price levels are large numbers, while their distance from the bar's own open is a small one.

Take Bitcoin at 95,000 with a tick size of 0.01. Stored as an absolute float32, 95,000 falls in the binade `[65536, 131072)`, where one unit in the last place is `2^16 * 2^-23 = 0.0078` — almost the whole tick, so rounding would visibly corrupt prices. Stored as a delta from the open, the value is the bar's range, say 100. That falls in the binade `[64, 128)`, where one unit in the last place is `2^6 * 2^-23 = 0.0000076` — about a thousandth of a tick. The anchor is kept in a full float64 `open`, so the only quantity that ever passes through float32 is the small offset, where float32 is precise to spare.

This is what makes the compact layout lossless in practice: the encoding puts the precision where the magnitude is small.

### Promotion to float64

The writer does not have to take the previous section on trust. Give it the instrument's tick grid and it checks the claim against every bar it stores:

```python
with OHLCVWriter(file_path, "1", minmove=1, pricescale=100) as writer:
    ...
```

The pair declares the grid as `mintick = minmove / pricescale` — the same two integers TradingView's symbol info uses, so fractional grids like `25/1000` (0.025) or `1/32` are exact. Each float32 delta column is measured on each bar against two criteria: the float32 resolution at that delta value must be smaller than half a tick, and the price itself must be an exact grid multiple. A column that fails either is **promoted** to an absolute float64 for the whole file. If the failure happens on the very first bar, only the header and descriptors are rewritten; later, the file is rebuilt into a temporary file and atomically renamed into place, after which appending continues normally. With all three price columns promoted, the record grows from 36 to 48 bytes.

Without a declared grid there is nothing to measure against, and float32 deltas are kept: no instrument in the measured corpus loses half a tick to them, and promoting on absence alone would inflate every file by a third.

Providers usually build their writer before the symbol info exists, so the grid can also be declared later with `set_tick_info(minmove, pricescale)`; the download flow injects it there before the first bar is appended. When a writer opens an existing file without its own grid, it adopts the one stored in the header.

NaN is preserved exactly. Delta columns store a canonical NaN rather than a difference, so a missing high or close survives a round trip instead of turning into a number.

### Read-Time Grid Snapping

Since minor version 1 the header carries the tick grid itself, and the reader uses it to undo the float32 rounding entirely. A delta-decoded price is off its true value by at most half an ulp of the stored delta — orders of magnitude less than half a tick. If the decoded value lies within that error bound of a grid point, the true value can only be that grid point, and the reader returns it at full float64 precision: `65399.99` comes back as `65399.99`, not `65399.990000000224`.

The writer-side promotion is what makes this safe for every record written under the declared grid: a price that is legitimately off-grid — split-adjusted history, for instance — is never stored as a float32 delta in the first place, so it comes back as the exact float64 it was written as and the snap never touches it. Such a float32 delta can only encode a true grid multiple, and snapping it merely removes the float32 noise.

A grid can also be declared after records were written — the data converter stamps its analyzed grid once the load finishes, and a continued download stamps the symbol info's grid onto a file that started without one. Those earlier records were never grid-checked, but the snap cannot corrupt them either: it only moves a value that already lies within the float32 error bound of a grid point, so any movement is capped at the encoding noise the delta storage introduced in the first place. A read through a late-declared grid is never worse than the gridless read; it just cannot promise the exact-restoration guarantee that write-time checking gives. Files with a zeroed grid (all minor-0 files) read exactly as before.

### The Committed Extent

`record_count` in the header is the single authority on how much data a file holds. Appends are published in two steps: the record bytes are written and fsynced first, and only then is the header's count raised to include them. A crash between the two leaves extra bytes past the committed end, which readers ignore and the next writer truncates away. The header can never promise data that is not on disk.

### No Gap Records

A missing time interval produces **no record at all**. Files are not required to be evenly spaced, so weekends, holidays and trading halts cost nothing, and nothing synthetic can ever be mistaken for a real bar.

Because gaps are allowed rather than papered over, the format states explicitly whether a given file happens to be gap-free. On every append the writer compares the new timestamp with the previous one; if every adjacent pair has matched the declared timeframe exactly, the `DENSE` flag stays set in the header. It is a verified fact about the stored data, not a promise about the data source:

```python
with OHLCVReader(file_path) as reader:
    print(reader.period)  # '1', '60', '1D', '1W' ... (None for a legacy file)
    print(reader.dense)   # True / False (None for a legacy file)
```

Empty and single-record files are never marked `DENSE` — there is no adjacent pair to verify — and neither are monthly files, whose calendar interval has no fixed millisecond length.

### The Declared Period

`period` uses TradingView notation with an explicit multiplier: bare numbers are minutes (`'1'`, `'60'`), and the `S`, `D`, `W` and `M` suffixes carry seconds, days, weeks and months (`'1D'`, not `'D'`). The writer requires it, stores it in the header as a multiplier plus a unit, and refuses to append to a file that declares a different one — mixing two timeframes into one file by accident is not possible.

## Memory Mapping for Maximum Performance

The reader uses memory mapping (`mmap`) to access file data through the operating system's virtual memory:

1. **Zero-copy access**: data is read straight from the file system cache
2. **Lazy loading**: only the pages actually touched are faulted in
3. **OS-level optimization**: read-ahead and caching are handled by the kernel
4. **Shared resources**: several processes can map the same file efficiently

The reader maps exactly the committed extent, so it works on an immutable snapshot: a writer appending in another process cannot shift data under an in-progress iteration.

## Usage Examples

### Basic Reading and Writing

```python
from pathlib import Path

from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV

file_path = Path("example.ohlcv")

# Writing OHLCV data - the timeframe is required and timestamps are milliseconds
with OHLCVWriter(file_path, "1", minmove=1, pricescale=100) as writer:
    writer.write(OHLCV(timestamp=1609459200000, open=100.0, high=110.0,
                       low=90.0, close=105.0, volume=1000.0))
    writer.write(OHLCV(timestamp=1609459260000, open=105.0, high=115.0,
                       low=95.0, close=110.0, volume=1200.0))

# Reading OHLCV data
with OHLCVReader(file_path) as reader:
    for candle in reader:
        print(f"Time: {candle.timestamp}, Close: {candle.close}, Volume: {candle.volume}")

    print(f"Timeframe: {reader.period}")
    print(f"Records: {reader.size}")
    print(f"Gap-free: {reader.dense}")
    print(f"Start time: {reader.start_datetime}")
    print(f"End time: {reader.end_datetime}")
```

Opening an existing file appends to it, and the declared timeframe must match the file's own.

### Reading Specific Time Ranges

Time ranges are resolved by binary search over the stored timestamps, so an irregular file is queried just as efficiently — and just as correctly — as a regular one:

```python
with OHLCVReader(file_path) as reader:
    start_time = 1609459200000  # Unix milliseconds
    end_time = 1609459800000    # Unix milliseconds

    for candle in reader.read_from(start_time, end_time):
        print(f"Time: {candle.timestamp}, Close: {candle.close}")

    # Bounds are inclusive; positions and counts use the same window
    first, end = reader.get_positions(start_time, end_time)
    print(reader.get_size(start_time, end_time), "records in range")
```

### Counting Records Without Opening a Reader

`record_count()` answers "how many records does this file hold?" from a few hundred header bytes. It opens no reader, maps nothing and never raises: it returns `0` for anything unusable — a missing, empty, truncated or invalid file, or a file that is not an OHLCV file at all — which makes it a convenient cache-validity gate. It works on both formats, selecting between them by the same magic bytes the reader uses:

```python
from pynecore.core.ohlcv import record_count

if record_count(file_path) == expected_bars:
    print("cache is intact")
```

This is the supported way to ask for a record count without opening the file. Deriving one from the file size is not: the size of a v2 file depends on its header and its schema, both of which vary.

### Converting From/To Other Formats

Importers live in `pynecore.core.ohlcv_importers` and write into an open writer; exporters are methods on the reader:

```python
from pathlib import Path

from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.core.ohlcv_importers import infer_csv_period, load_from_csv

# Import from CSV - the timeframe is inferred from the source's own cadence
period = infer_csv_period(Path("data.csv"), tz="UTC")
with OHLCVWriter(Path("from_csv.ohlcv"), period, truncate=True) as writer:
    load_from_csv(writer, Path("data.csv"), tz="UTC")

# Export to CSV / JSON (as_datetime=True writes ISO dates instead of milliseconds)
with OHLCVReader(Path("from_csv.ohlcv")) as reader:
    reader.save_to_csv("exported.csv", as_datetime=True)
    reader.save_to_json("exported.json", as_datetime=True)
```

`infer_csv_period` inspects the source's timestamps and returns a canonical period string for the header; equivalents exist for TXT and JSON (`infer_txt_period`, `load_from_txt`, `infer_json_period`, `load_from_json`). Inference happens once, on import, from the source's own timestamps, and the result is then declared in the header — it is not v1's read-time guess from the first two records.

For everyday use, the `pyne data convert-from` CLI command wraps exactly this flow and writes the matching symbol-info TOML as well.

### Extra Fields

Columns a source file carries beyond OHLCV — an indicator value, a signal label, a bid/ask snapshot — are stored in a `.extra.csv` sidecar next to the binary file. The sidecar holds **exactly one data row per committed record**, so row *N* always describes record *N*, and the reader attaches it automatically:

```python
with OHLCVReader(file_path) as reader:
    for candle in reader:
        print(candle.extra_fields)  # None when there is no sidecar
```

Writing follows the same rule: pass `extra_fields` on the `OHLCV` you write and the row is appended just before the record itself is published, so a failed write leaves neither behind and can be retried. Because alignment is positional, truncating a file also deletes its sidecar — a sidecar can never outlive the records it describes.

## Advanced Operations

### Truncating and Replacing

The writer only ever appends, so there is no seek. To start over, truncate:

```python
# Empty an open file, keeping its timeframe and schema
with OHLCVWriter(file_path, "1") as writer:
    writer.truncate()

# Or replace whatever is on disk with a fresh, empty file
with OHLCVWriter(file_path, "1", truncate=True) as writer:
    ...
```

`truncate=True` replaces *any* existing file, including one written by an older PyneCore version.

### Automatic Symbol Analysis

While writing, the writer observes the data and derives symbol properties that most data sources do not provide. `pyne data convert-from` uses these to generate the symbol-info TOML:

```python
with OHLCVWriter(file_path, "1") as writer:
    for candle in candles:
        writer.write(candle)

    print(writer.analyzed_tick_size, writer.tick_analysis_confidence)
    print(writer.analyzed_price_scale, writer.analyzed_min_move)
    print(writer.analyzed_qty_step)        # mincontract candidate from volumes
    print(writer.analyzed_opening_hours)   # session intervals from bar times
```

## Reading Legacy v1 Files

Files written by earlier PyneCore versions remain readable through the same `OHLCVReader`. The first eight bytes are the sole discriminator: the v2 magic selects the native reader, anything else selects the read-only legacy one. The choice is made once, in `open()`, and nothing on the data path checks the version again.

What differs for a legacy file:

- **Timestamps are converted for you.** v1 stored seconds; every value the reader hands out — `OHLCV.timestamp`, `start_timestamp`, `end_timestamp` and the `read_from()` bounds — is in milliseconds, like everywhere else in the API.
- **`period` and `dense` are `None`.** v1 declares neither, and the reader does not invent them by inference.
- **Phantom records still exist on disk.** Range reads skip them; a full iteration yields them, recognisable by their negative volume.
- **Range queries inherit v1's addressing.** Positions in a v1 file can only be computed from its inferred interval, so an irregular legacy file can still return the wrong window. Nothing can repair that after the fact — only re-downloading the data into a v2 file.
- **`extra_fields` is an empty dict, not `None`,** when a legacy file has no sidecar. Test it for truthiness rather than for `None` if your code must handle both formats.

**There is no v1 writer.** v1 files can be read, never written. Opening one for append is refused — old records cannot be extended in the current format — so migrating a file means re-downloading or re-converting it, which writes a fresh v2 file. Legacy reading exists solely as a migration path and will be removed in a future release; treat any v1 file still on disk as something to convert, not something to keep.

## Technical Details

### Memory Efficiency

Only the pages actually touched are loaded, so datasets larger than available RAM can be read without loading them into memory.

### Chronological Validation

The writer enforces data integrity as records arrive:

1. Records must be written in strictly ascending timestamp order
2. Duplicate timestamps are rejected
3. OHLC relations are validated (`high` cannot be below `open`/`close`, `low` cannot be above them)
4. Infinite values are rejected; `na` is preserved as a canonical NaN
5. Timestamps must fit the signed 64-bit millisecond range

### Performance Considerations

1. Direct memory access provides near-native speed
2. Fixed-size records enable O(1) random access by position
3. Binary search over stored timestamps keeps time-range queries fast without assuming regular spacing
4. Header metadata answers count and range questions without touching the data
5. No compression means no CPU overhead for decompression

For typical backtesting scenarios, the system can process millions of candles per second on modern hardware.

## Conclusion

The OHLCV v2 format is what a binary time-series container looks like once it stops assuming its data is regular. The header declares the timeframe instead of guessing it, the `DENSE` flag reports the truth about spacing instead of enforcing it, lookups search the timestamps they actually find, and the file contains only bars that really traded. Delta-encoded records keep it compact without giving up precision, and memory mapping keeps a pure Python implementation competitive with native code.
