<!--
---
weight: 439
title: "Global functions"
description: "Top-level functions of pynecore.lib — time and date, na handling, max_bars_back and the plot family"
icon: "functions"
date: "2026-10-25"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["time", "time_close", "timestamp", "session", "na", "nz", "fixnan", "max_bars_back", "plot", "plotshape", "bgcolor", "fill"]
---
-->

# Global functions

The functions on this page live at the top level of `pynecore.lib`, not in a namespace. You
import them by name, the same way as the bar variables:

```python
from pynecore.lib import time, time_close, timestamp, year, na, nz, fixnan, plotshape
```

They fall into four groups: [time and date](#time-and-date), [na handling](#na-handling),
[max_bars_back()](#max_bars_back) and the [plot family](#plot-family).

## Quick Example

```python
from pynecore.lib import (
    script, input, time, timestamp, hour, na, nz, fixnan, close, ta,
    plot, plotshape, bgcolor, color, shape, location
)


@script.indicator(title="Global Functions", overlay=True)
def main(start: int = input.time(timestamp("2024-01-01"), title="Start")):
    # time() with a session is na on bars outside 09:30-16:00 New York time
    in_session: bool = not na(time("60", "0930-1600", "America/New_York"))
    # The bare name is the current bar's open time
    active: bool = time >= start

    sma: float = ta.sma(close, 20)
    # Hold the last SMA value through the bars outside the session
    held: float = fixnan(sma if in_session else na)

    plot(held if active else na, title="Session SMA")
    plot(nz(close - sma), title="Distance")
    plotshape(active and in_session and hour(time, "America/New_York") == 10,
              title="10 o'clock", style=shape.circle, location=location.belowbar,
              color=color.blue)
    bgcolor(color.new(color.green, 90) if in_session else na)
```

## Bare names and calls

Several names on this page are both a value and a function. Written bare, the name is the value
for the current bar; called with arguments, it is the function:

| Bare name         | Value on the current bar                 | Call form                                  |
|-------------------|------------------------------------------|--------------------------------------------|
| `time`            | Open time of the bar, UNIX milliseconds  | [`time(...)`](#time)                       |
| `time_close`      | Close time of the bar, UNIX milliseconds | [`time_close(...)`](#time_close)           |
| `time_tradingday` | Start of the bar's trading day           | —                                          |
| `timenow`         | Current time                             | —                                          |
| `updatetime`      | Time of the bar's latest update          | —                                          |
| `year`, `month`   | Calendar parts of the bar's open time    | [`year(time, timezone)`](#date-components) |
| `weekofyear`      | ISO week number of the bar's open time   | [`weekofyear(...)`](#date-components)      |
| `dayofmonth`      | Day of the month of the bar's open time  | [`dayofmonth(...)`](#date-components)      |
| `dayofweek`       | Day of the week of the bar's open time   | [`dayofweek(...)`](#date-components)       |
| `hour`, `minute`  | Clock parts of the bar's open time       | [`hour(time, timezone)`](#date-components) |
| `second`          | Seconds part of the bar's open time      | [`second(...)`](#date-components)          |
| `na`              | The typeless na value                    | [`na(x)`](#na)                             |

The AST transformer turns a bare read into a call with no arguments, so `time` and `time()` give
the same number. A bare name works everywhere a value does, as an argument, in an expression, or
with the history operator:

```python
p = chart.point.from_time(time, high)   # bare time: the current bar's open time
prev_open: float = time[1]              # open time of the previous bar
same: bool = time == time()             # always true
```

The bar variables `open`, `high`, `low`, `close`, `volume`, `hl2`, `hlc3`, `ohlc4`, `hlcc4`,
`bar_index`, `last_bar_index` and `last_bar_time` are plain series values with no call form.

All times are UNIX timestamps in milliseconds, stored as a Pine `int` (a Python `float` with an
integral value, see [Types](../types.md#int)). Calendar parts are computed in the exchange
timezone (`syminfo.timezone`) unless a function takes a `timezone` argument and you pass one.

---

## Time and date

### time()

Open time of the bar that a timeframe, session and bar offset name, or `na`.

```python
time(timeframe=None, session=None, timezone=None, bars_back=0, timeframe_bars_back=0) -> int
```

| Parameter             | Type         | Description                                                                       |
|-----------------------|--------------|-----------------------------------------------------------------------------------|
| `timeframe`           | `str`        | Timeframe of the bar to report; `""` or `na` is the chart's timeframe             |
| `session`             | `str \| int` | Session filter, see [Session strings](#session-strings); an `int` is `bars_back`  |
| `timezone`            | `str`        | Timezone of the `session` hours (default: the exchange timezone)                  |
| `bars_back`           | `int`        | Chart bars back to evaluate the call on; negative for a future bar (default: `0`) |
| `timeframe_bars_back` | `int`        | Bars to step on the requested timeframe's grid (default: `0`)                     |

All parameters are optional. `time()` with no arguments is the current chart bar's open time.

**Returns:** `int` — UNIX time in milliseconds, or `na`.

How the arguments combine:

- **No arguments** (`time()`, or bare `time`) returns the current chart bar's open time.
- **A timeframe** returns the open time of the bar of that timeframe that contains the current
  chart bar: on a 60-minute chart `time("D")` is the open of the bar's day and `time("240")` the
  open of its 4-hour bucket. Weekly and monthly bars open at the session open of their first
  trading day. On a daily, weekly or monthly chart an intraday timeframe (`"60"`, `"1"`, ...)
  resolves as `"D"`. An invalid timeframe string returns `na`.
- **A session** turns the result into a session filter: the call returns `na` when the current
  chart bar is outside the session, whatever intraday or daily timeframe is requested (`"W"` and
  `"M"` requests with a session are never `na`). Inside the session an intraday timeframe is
  tiled from the session open, so `time("60", "0930-1600")` on the 10:00 chart bar reports
  09:30, the open of the session's first 1-hour bucket. A `"D"` request reports the session's
  own open for that day. A malformed session specification returns `na`.
- **`bars_back`** names a chart bar by offset: `time("", 1)` equals `time[1]`, and the result is
  `na` while the chart has fewer bars behind it (or the offset exceeds the 5000-bar history
  limit). A negative offset asks for a bar that has not opened yet: `time("60", -1)` is the
  expected open of the next 1-hour bar, walked over the symbol's session schedule (it skips the
  weekend, but knows neither holidays nor bars missing from the feed).
- **`timeframe_bars_back`** moves on the requested timeframe's grid instead of the chart's:
  `time("D", timeframe_bars_back=1)` is the open of the previous trading day. Intraday grids are
  walked one bar at a time over the symbol's sessions. Daily, weekly and monthly walks across the
  turn of the year follow a specific calendar rule, described in
  [time() offsets on D/W/M grids](../../development/tradingview-time-bars-back.md).

A second positional `int` argument is the two-argument `time(timeframe, bars_back)` form:

```python
prev_hour: int = time("60", 1)        # open of the 1-hour bar holding the previous chart bar
next_hour: int = time("60", -1)       # expected open of the next 1-hour bar
```

The most common use is a session test, where only `na` versus "not `na`" matters:

```python
from pynecore.lib import script, time, na, close, plot

@script.indicator(title="London Session", overlay=True)
def main():
    in_london: bool = not na(time("", "0800-1630", "Europe/London"))
    plot(close if in_london else na, title="London close")
```

### time_close()

Close time of the bar that a timeframe, session and bar offset name, or `na`. The parameters are
the same as for [time()](#time).

```python
time_close(timeframe=None, session=None, timezone=None, bars_back=0, timeframe_bars_back=0) -> int
```

| Parameter             | Type         | Description                                                                       |
|-----------------------|--------------|-----------------------------------------------------------------------------------|
| `timeframe`           | `str`        | Timeframe of the bar to report; `""` or `na` is the chart's timeframe             |
| `session`             | `str \| int` | Session filter, see [Session strings](#session-strings); an `int` is `bars_back`  |
| `timezone`            | `str`        | Timezone of the `session` hours (default: the exchange timezone)                  |
| `bars_back`           | `int`        | Chart bars back to evaluate the call on; negative for a future bar (default: `0`) |
| `timeframe_bars_back` | `int`        | Bars to step on the requested timeframe's grid (default: `0`)                     |

All parameters are optional. `time_close()` with no arguments is the current chart bar's close time.

**Returns:** `int` — UNIX time in milliseconds, or `na`.

The close of an intraday bar is its open plus the timeframe length, cut short at the end of the
bar's trading day: the last bar of a session closes when the session ends. Lunch breaks and
continuous overnight sessions do not cut a bar short. A daily bar closes at the end of its trading
day. A longer period (`nD`, `W`, `M`) closes at the end of its last trading day on a daily, weekly
or monthly chart, and when the next period opens on an intraday chart.

```python
bar_end: int = time_close                 # close of the current chart bar
day_end: int = time_close("D")            # end of the current trading day
```

### time_tradingday

Start of the trading day the current bar belongs to: 00:00 UTC of the calendar date, in the
exchange timezone, on which the bar's trading session ends. Read it bare; it takes no arguments.

**Type:** `int` — UNIX time in milliseconds.

For a session that stays within one calendar day (stocks, 24/7 crypto) this is 00:00 UTC of the
bar's own exchange-timezone date. For a session that crosses midnight (forex, futures), a bar that
reaches into the evening session open belongs to the next calendar day's trading day, including
the bar whose window merely contains the open (a 17:00-18:00 bar for a 17:05 open).

```python
new_day: bool = time_tradingday != time_tradingday[1]
```

### timenow

Current time, UNIX milliseconds. Read it bare.

**Type:** `int`

During a historical run started with `pyne run`, `timenow` is pinned to the open time of the data's
last bar, so a script that filters on `time >= timenow - ...` gives the same result on every run.
A live run reads the system clock. When you run a script through
[`ScriptRunner`](../../programmatic/script-runner.md) yourself, nothing pins it, and it reads the
system clock.

### updatetime

Time of the bar's latest update, UNIX milliseconds, as a `float`. Read it bare.

**Type:** `float`

On historical bars it is the bar's open time. On a realtime bar it is the clock reading at the
first read during that update, with sub-millisecond precision, and every later read in the same
update returns the same value. If `timenow` is pinned, the pin is used instead of the clock.

### timestamp()

Build a UNIX timestamp in milliseconds from a date string or from date and time components.

```python
timestamp(dateString) -> int
timestamp(year, month, day, hour=0, minute=0, second=0) -> int
timestamp(timezone, year, month, day, hour=0, minute=0, second=0) -> int
```

**From a date string:**

| Parameter    | Type  | Description       |
|--------------|-------|-------------------|
| `dateString` | `str` | The date to parse |

`date_string` is accepted as the keyword too. A string without a timezone is read as UTC (GMT+0),
whatever the exchange timezone is. A missing time is 00:00. Accepted spellings include:

| Form                    | Example                                       |
|-------------------------|-----------------------------------------------|
| ISO 8601 date           | `"2024-01-05"`                                |
| ISO 8601 date and time  | `"2024-01-05T15:30"`, `"2024-01-05 15:30:00"` |
| ISO 8601 with offset    | `"2024-01-05T15:30:00+02:00"`, `"...-0500"`   |
| RFC 2822 style          | `"20 Feb 2020 15:30:00 GMT+0200"`             |
| Month name              | `"Feb 01 2020 22:10:05"`, `"1 January 2018"`  |
| Month first, numeric    | `"01-02-2023"`, `"03/04/2023 10:20:30"`       |
| Year, or year and month | `"2025"`, `"2025-06"`, `"Jan 2025"`           |

A string that matches none of the formats raises `ValueError` and stops the script.

**From components:**

| Parameter  | Type  | Description                                                     |
|------------|-------|-----------------------------------------------------------------|
| `timezone` | `str` | Optional first argument: timezone of the components (see below) |
| `year`     | `int` | Year                                                            |
| `month`    | `int` | Month, 1-12                                                     |
| `day`      | `int` | Day of the month                                                |
| `hour`     | `int` | Hour (default: `0`)                                             |
| `minute`   | `int` | Minute (default: `0`)                                           |
| `second`   | `int` | Second (default: `0`)                                           |

Without a `timezone` argument the components are read in the exchange timezone. A timezone is an
IANA name (`"America/New_York"`), `"UTC±H"` / `"UTC±HHMM"`, `"GMT±H"` / `"GMT±HHMM"`, or a raw
offset (`"+0530"`, `"-05:00"`).

Out-of-range components roll over: month 13 is January of the next year, hour 26 is 02:00 of the
next day. An `na` component counts as `0`. Dates before 1582-10-15 use the Julian calendar, and
years outside 1-9999 are supported.

**Returns:** `int` — UNIX time in milliseconds.

```python
t1: int = timestamp("2024-01-05")                           # 2024-01-05 00:00 UTC
t2: int = timestamp(2024, 1, 5)                             # 2024-01-05 00:00, exchange timezone
t3: int = timestamp("America/New_York", 2024, 1, 5, 9, 30)  # 09:30 New York time
t4: int = timestamp(2024, 13, 1)                            # 2025-01-01: month 13 rolls over
```

### Date components

`year`, `month`, `weekofyear`, `dayofmonth`, `dayofweek`, `hour`, `minute` and `second` read one
part of a timestamp. Bare, they read the current bar's open time in the exchange timezone. Called,
they read the timestamp you pass.

```python
year(time=None, timezone=None) -> int
```

| Parameter  | Type  | Description                                                      |
|------------|-------|------------------------------------------------------------------|
| `time`     | `int` | UNIX time in milliseconds. Default: the current bar's open time  |
| `timezone` | `str` | Timezone the part is computed in. Default: the exchange timezone |

**Returns:** `int`, or `na` when `time` is `na`.

| Function       | Result                                                                                    |
|----------------|-------------------------------------------------------------------------------------------|
| `year()`       | Year                                                                                      |
| `month()`      | Month, 1-12                                                                               |
| `weekofyear()` | ISO 8601 week number, 1-53                                                                |
| `dayofmonth()` | Day of the month, 1-31                                                                    |
| `dayofweek()`  | Day of the week, 1 = Sunday ... 7 = Saturday (compare with [`dayofweek.*`](dayofweek.md)) |
| `hour()`       | Hour, 0-23                                                                                |
| `minute()`     | Minute, 0-59                                                                              |
| `second()`     | Second, 0-59                                                                              |

```python
is_january: bool = month == 1
ny_hour: int = hour(time, "America/New_York")
close_day: int = dayofmonth(time_close)
```

`dayofweek` is also the namespace of the day constants, so `dayofweek == dayofweek.monday`
compares the current bar's day with a constant. See [dayofweek](dayofweek.md).

### Session strings

The `session` argument of `time()` and `time_close()` uses this format, the same one that
`input.session()` values are written in:

| Form                            | Meaning                                                      |
|---------------------------------|--------------------------------------------------------------|
| `"0930-1600"`                   | One range, every day of the week                             |
| `"0930-1600:23456"`             | One range on the listed days (1 = Sunday ... 7 = Saturday)   |
| `"0400-0700,0900-1300:23456"`   | Several ranges on the listed days                            |
| `"0930-1600\|1000-1300:7"`      | Sections joined by `\|`, each with its own days              |
| `"2200-0200"`                   | A range whose end is earlier than its start crosses midnight |
| `"0930-2500"`                   | Hours past 24 run into the next day (here: until 01:00)      |
| `"24x7"`                        | The whole day, every day                                     |
| `""`, `"regular"`, `"extended"` | The symbol's own session (see below)                         |

Rules:

- A section without a day list, in any form other than a single bare range, runs on the weekdays
  (Monday to Friday) that no other section names. A later section that names a day replaces
  the earlier ones on that day.
- The day digit of a run is the day its last minute falls on, which matters for ranges that cross
  midnight.
- A string that is empty, blank or does not start with a digit selects the symbol's own session
  from its `SymInfo` opening hours. `"extended"` selects the extended (pre- and post-market) hours
  when the symbol has them, `"regular"` the regular ones; see
  [Extended Trading Hours](../../programmatic/data-and-syminfo.md#extended-trading-hours).
- A string that starts with a digit but is malformed (`"0930"`, `"0930-1600:8"`, ...) makes the
  call return `na`.
- An unknown timezone raises an error instead of returning `na`.

The [session](session.md) namespace has flags for the symbol's own session (`session.ismarket`,
`session.isfirstbar`, ...) and [timeframe](timeframe.md) has helpers for timeframe strings.
[Compatibility](../../overview/compatibility.md#session-mask-on-the-day-before-a-fall-dst-change)
documents one session edge case on the day before a fall DST change.

---

## na handling

### na

`na` has three faces:

| Written as  | Meaning                                                                       |
|-------------|-------------------------------------------------------------------------------|
| `na`        | The na value, for assignments and arguments: `x: float = close if ok else na` |
| `na(x)`     | Predicate: `True` when `x` is na                                              |
| `na(float)` | An na of the given type (`na(int)`, `na(bool)`, `na(str)`, ...)               |

```python
na(x) -> bool
```

| Parameter | Type | Description       |
|-----------|------|-------------------|
| `x`       | any  | The value to test |

**Returns:** `bool`

`na(x)` is true for a float `nan`, `inf` or `-inf`, for any na object, and for the empty string
(the string na). Everything else, including `0` and `False`, is not na. The parameter is named
`x`, so `na(x=close)` works too.

A numeric na is a native float `nan`: `na(float)` and `na(int)` both return `nan`, and a `float`
or `int` variable assigned `na` holds `nan`. Because `nan` compares unequal to everything, itself
included, `x == na` is always false. Test with `na(x)`:

```python
sma20: float = ta.sma(close, 20)
warming_up: bool = na(sma20)       # true for the first 19 bars
wrong: bool = sma20 == na          # always false, do not use
```

In arithmetic `na` propagates (`na + 1` is `na`); see [Types](../types.md#na).

### nz()

Replace an na value.

```python
nz(source, replacement=0.0)
```

| Parameter     | Type | Description                                                       |
|---------------|------|-------------------------------------------------------------------|
| `source`      | any  | The value to check                                                |
| `replacement` | any  | Returned when `source` is na (default: zero of the source's type) |

**Returns:** `source` when it is not na, otherwise `replacement`.

Float `nan`, `inf` and `-inf` all count as na, so `nz(math.inf, -5.0)` is `-5.0`. Without a
`replacement` a number becomes `0.0`, an na `bool` becomes `False` and an na color becomes
transparent black.

```python
change: float = nz(close - close[1])          # 0.0 on the first bar
prev_high: float = nz(high[1], high)          # current high on the first bar
```

### fixnan()

Replace na values with the last value that was not na.

```python
fixnan(source)
```

| Parameter | Type | Description        |
|-----------|------|--------------------|
| `source`  | any  | The series to fill |

**Returns:** `source` when it is not na, otherwise the most recent earlier value that was not na.
Before the first such value it returns `na`.

`fixnan()` keeps state per call site, like a `ta.*` function, so call it on every bar. Only `nan`
and na objects count as holes: unlike `na()`, it passes `inf` through as a regular value.

```python
pivot: float = ta.pivothigh(high, 5, 5)
last_pivot: float = fixnan(pivot)   # the latest pivot high, held between pivots
```

---

## max_bars_back()

Set how many past bars a series keeps for the history operator.

```python
max_bars_back(var, num) -> None
```

| Parameter | Type   | Description                          |
|-----------|--------|--------------------------------------|
| `var`     | series | A variable declared as `Series[...]` |
| `num`     | `int`  | Number of past bars to keep, 1-5000  |

Every series keeps 5000 past bars by default, which is also the maximum. Call `max_bars_back()`
as a statement on a `Series` variable to resize its buffer, for example to save memory on a series
that is only read a few bars back. Reading further back than the buffer holds returns `na`. A
value of `0` or less, or above 5000, raises `ValueError`. In any other position, or on something
other than a series variable name, the call does nothing.

```python
from pynecore.types import Series
from pynecore.lib import script, close, max_bars_back, plot

@script.indicator(title="Short History")
def main():
    src: Series[float] = close
    max_bars_back(src, 10)
    plot(src[10], title="Close 10 bars ago")   # src[11] would be na
```

---

## Plot family

`plot`, `plotshape`, `plotchar`, `plotarrow`, `plotcandle`, `plotbar`, `bgcolor`, `barcolor`,
`fill`, `hline` and `alertcondition` declare outputs.

> **Unconditional calls only.** A plot-family call must run on every bar, in the direct body of
> `main()`, before any possible early return. A call inside an `if`, a loop, a conditional
> expression or a helper function is a `SyntaxError` at transform time. To hide a value on some
> bars, keep the call unconditional and pass `na`. The values and colors you pass can be
> conditional. See [plot](plot.md) for the details.

```python
plotshape(signal, title="Signal")                # signal is True/False per bar
bgcolor(color.red if alert_bar else na)          # na leaves the bar unpainted
```

`plot`, `plotshape`, `plotchar`, `plotarrow`, `plotcandle` and `plotbar` write their values to the
plot CSV (`pyne run --plot`, `plot_path`), one column per title; a repeated title gets a numeric
suffix. `bgcolor`, `barcolor`, `fill` and `hline` produce no CSV column. All of them, with their
styles and colors, are recorded in the [visual output](../../programmatic/visual-output.md). A
color that changes from bar to bar is recorded per bar.

Several parameters are shared by most functions:

| Parameter       | Type        | Description                                                                 |
|-----------------|-------------|-----------------------------------------------------------------------------|
| `title`         | `str`       | Output title, the CSV column name and visual output id                      |
| `offset`        | `int`       | Shift the output this many bars (default: `0`)                              |
| `editable`      | `bool`      | Style editable in a settings dialog (default: `True`)                       |
| `show_last`     | `int`       | Draw only the last this many bars (default: `None`, all bars)               |
| `display`       | `display.*` | Where the output is shown, see [display](display.md) (default: `None`, all) |
| `format`        | `format.*`  | Value format, see [format](format.md) (default: `None`)                     |
| `precision`     | `int`       | Decimal places of displayed values (default: `None`)                        |
| `force_overlay` | `bool`      | Draw on the main chart pane (default: `False`)                              |

### plot()

Plot a series. Full reference and style constants: [plot](plot.md).

```python
plot(series, title=None, color=None, linewidth=1, style=None, trackprice=False, histbase=0.0,
     offset=0, join=False, editable=True, show_last=None, display=None, format=None,
     precision=None, force_overlay=False) -> Plot
```

**Returns:** a `Plot` reference for [fill()](#fill). The default title is `"Plot"`.

### plotshape()

Draw a shape marker on bars where `series` is true.

```python
plotshape(series, title=None, style=None, location=None, color=None, offset=0, text=None,
          textcolor=None, editable=True, size=None, show_last=None, display=None, format=None,
          precision=None, force_overlay=False) -> None
```

| Parameter   | Type            | Description                                                                 |
|-------------|-----------------|-----------------------------------------------------------------------------|
| `series`    | `bool \| float` | Marker on bars where it is true; `na` draws nothing                         |
| `style`     | `shape.*`       | Marker shape, see [shape](shape.md) (default: `shape.xcross`)               |
| `location`  | `location.*`    | Marker position, see [location](location.md) (default: `location.abovebar`) |
| `color`     | `Color`         | Marker color                                                                |
| `text`      | `str`           | Text shown with the marker                                                  |
| `textcolor` | `Color`         | Text color                                                                  |
| `size`      | `size.*`        | Marker size, see [size](size.md) (default: `size.auto`)                     |

The CSV column holds `1`/`0` for a `bool` series and the raw value for a numeric one, so
`plotshape(high if cond else na, location=location.absolute)` records the price. The default title
is `"Shapes"`.

### plotchar()

Draw a character marker on bars where `series` is true. Parameters as for
[plotshape()](#plotshape), with `char` in place of `style`:

```python
plotchar(series, title=None, char=None, location=None, color=None, offset=0, text=None,
         textcolor=None, editable=True, size=None, show_last=None, display=None, format=None,
         precision=None, force_overlay=False) -> None
```

| Parameter | Type  | Description                                                      |
|-----------|-------|------------------------------------------------------------------|
| `char`    | `str` | The character to draw; only the first one is used (default: `★`) |

The default title is `"Chars"`.

### plotarrow()

Draw an up arrow for a positive value and a down arrow for a negative one, with a height scaled by
the absolute value.

```python
plotarrow(series, title=None, colorup=None, colordown=None, offset=0, minheight=5, maxheight=100,
          editable=True, show_last=None, display=None, format=None, precision=None,
          force_overlay=False) -> None
```

| Parameter   | Type    | Description                                     |
|-------------|---------|-------------------------------------------------|
| `series`    | `float` | Positive: up arrow, negative: down arrow        |
| `colorup`   | `Color` | Color of up arrows                              |
| `colordown` | `Color` | Color of down arrows                            |
| `minheight` | `int`   | Minimum arrow height in pixels (default: `5`)   |
| `maxheight` | `int`   | Maximum arrow height in pixels (default: `100`) |

The default title is `"Arrows"`.

### plotcandle() and plotbar()

Draw candles or OHLC bars from four series.

```python
plotcandle(open, high, low, close, title=None, color=None, wickcolor=None, editable=True,
           show_last=None, bordercolor=None, display=None, format=None, precision=None,
           force_overlay=False) -> None
plotbar(open, high, low, close, title=None, color=None, editable=True, show_last=None,
        display=None, format=None, precision=None, force_overlay=False) -> None
```

| Parameter     | Type    | Description                       |
|---------------|---------|-----------------------------------|
| `open`        | `float` | Open value                        |
| `high`        | `float` | High value                        |
| `low`         | `float` | Low value                         |
| `close`       | `float` | Close value                       |
| `color`       | `Color` | Body color (candle) or bar color  |
| `wickcolor`   | `Color` | Wick color, `plotcandle()` only   |
| `bordercolor` | `Color` | Border color, `plotcandle()` only |

Each call writes four CSV columns: `"<title> (open)"`, `"<title> (high)"`, `"<title> (low)"` and
`"<title> (close)"`. The default titles are `"Candles"` and `"Bars"`.

### bgcolor()

Paint the background of the current bar.

```python
bgcolor(color=None, offset=0, editable=True, show_last=None, title=None, display=None,
        force_overlay=False) -> None
```

| Parameter | Type    | Description                                     |
|-----------|---------|-------------------------------------------------|
| `color`   | `Color` | Background color; `na` leaves the bar unpainted |

The default title is `"Background color"`.

### barcolor()

Color the price bar of the current bar.

```python
barcolor(color=None, offset=0, editable=True, show_last=None, title=None, display=None) -> None
```

| Parameter | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `color`   | `Color` | Bar color; `na` leaves the bar unchanged |

The default title is `"Bar color"`.

### fill()

Fill the area between two plots or two hlines, with a solid color or a vertical gradient.

```python
fill(plot1, plot2, color=None, title=None, editable=True, show_last=None, fillgaps=False,
     display=None)
fill(hline1, hline2, color=None, title=None, editable=True, fillgaps=False, display=None)
fill(plot1, plot2, top_value, bottom_value, top_color, bottom_color, title=None, display=None,
     fillgaps=False, editable=True)
```

| Parameter          | Type    | Description                                           |
|--------------------|---------|-------------------------------------------------------|
| `plot1`, `plot2`   | `Plot`  | Plot references returned by `plot()`                  |
| `hline1`, `hline2` | `HLine` | Hline references returned by `hline()`                |
| `color`            | `Color` | Solid fill color                                      |
| `top_value`        | `float` | Value mapped to `top_color` (gradient form)           |
| `bottom_value`     | `float` | Value mapped to `bottom_color` (gradient form)        |
| `top_color`        | `Color` | Color at `top_value` (gradient form)                  |
| `bottom_color`     | `Color` | Color at `bottom_value` (gradient form)               |
| `fillgaps`         | `bool`  | Continue the fill across `na` gaps (default: `False`) |

Positional arguments bind to the hline form when the first argument is an hline, to the gradient
form when the third argument is a number, and to the plot form otherwise. The default title is
`"Plots Background"`.

```python
upper = plot(ta.highest(high, 20), title="Upper")
lower = plot(ta.lowest(low, 20), title="Lower")
fill(upper, lower, color.new(color.blue, 85))
```

### hline()

Horizontal line at a fixed price; its reference can be passed to `fill()`. See [hline](hline.md).

### alertcondition()

Declares an alert condition. In PyneCore the call is accepted and does nothing; no alert is
raised. It is still a plot-family call and must be unconditional. To raise alerts while a script
runs, use [alert()](alert.md).

```python
alertcondition(condition, title=None, message=None) -> None
```

| Parameter   | Type   | Description         |
|-------------|--------|---------------------|
| `condition` | `bool` | The alert condition |
| `title`     | `str`  | Alert title         |
| `message`   | `str`  | Alert message       |

---

## Compatibility

- `time()` and `time_close()` support every Pine v6 form: timeframe, session, timezone,
  `bars_back` and `timeframe_bars_back`, and the two-argument `(timeframe, bars_back)` form.
  Future-bar offsets follow the session schedule only, so holidays and bars missing from the feed
  are not known in advance.
- A malformed session specification returns `na` instead of stopping the script.
- `timenow` is pinned to the last bar during a `pyne run` backtest, which makes backtests that
  read it reproducible.
- `timestamp()` also accepts the keyword `date_string` in addition to `dateString`.
- `max_bars_back()` resizes a `Series` buffer when called as a statement on a series variable;
  elsewhere it is a no-op. The history limit is 5000 bars.
- `alertcondition()` is a no-op: PyneCore has no alert dialog to register conditions with.
- Plot-family calls must be unconditional; see [plot](plot.md).
