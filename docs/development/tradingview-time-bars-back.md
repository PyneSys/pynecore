# `time()` / `time_close()` with `timeframe_bars_back` on D, W and M grids

How PyneCore walks a requested daily, weekly or monthly grid when `timeframe_bars_back`
is not zero, and what was measured on TradingView to arrive at it. The intraday walk
(sessions, buckets) is documented in the `time()` docstring and in
`_session_bar_bounds`; this note covers the calendar grids only.

## What was measured

Probes on CAPITALCOM:US500 (daily, 240- and 60-minute charts, 1999-2026, 27 turns of
the year), BTCUSD (daily and 60-minute), EURUSD, AAPL and GOLD (daily), 2026-09-26:

| Grid | Multipliers | Offsets | Values compared | Mismatches |
|------|-------------|---------|-----------------|------------|
| nW   | 1-18, 20, 25, 26, 30, 40, 52 | -2 .. 20 | 1 482 278 | 0 |
| nM   | 5-11 | -1, 1, 2 | 180 202 | 0 |
| D    | 1 | -4 .. 6 | 70 180 | 0 |
| nD   | 2-60 | -2 .. 8, and 5 .. 132 across two and three turns of the year | 1 900 000 | 0 |

Inside a year every grid is regular, and a walk of any length is exact there. The
differences appear only when the walk crosses the turn of the year, where the yearly
reset of the grid leaves a short last bar (the remainder weeks, months or days).

## Weekly grid, backward

TradingView does not step from bar to bar. One step subtracts the bar length (n weeks)
from the chart bar's time and reports the bar holding the result. Inside a year that is
exact. When the result falls before the year's first Monday, D weeks before it, the
result is not "D weeks before the first Monday" counted on the previous year's grid:
TradingView tiles the previous year with D-week tiles from *its* first Monday and lands
on the last tile that starts inside that year.

```
T  = chart bar time - n weeks           (the naive target)
if T >= first_monday(year):             # same year
    land on T
else:
    D  = weeks from monday(T) to first_monday(year)
    Wp = weeks in the previous year     (52 or 53)
    land on first_monday(year) - ((Wp - 1) mod D + 1) weeks
```

The two agree whenever D divides Wp, and differ by up to D - 1 weeks otherwise. Two
visible consequences:

- From the first week of the year (D = n) the walk always reaches the previous year's
  short last bar, so `time("3W", timeframe_bars_back=1)` on the first week of 2020 is
  the lone week of 2019-12-30.
- From the second week D = n - 1; 52 weeks tile into 2-week tiles with none left over,
  so the same call skips the lone week and reports the bar of 2019-12-09. From the
  third week it reaches the lone week again.

Further steps continue from the landing week, so they stay inside the previous year
until they cross again, when the same rule applies with that year's numbers. A single
"W" grid follows the same rule: on the days of January that precede the year's first
Monday (they belong to the previous year's last week), one bar back after a 53-week year
is that same week.

PyneCore: `_tv_week_walk` in `lib/__init__.py`.

## Weekly grid forward, monthly grid both ways

Plain arithmetic on the chart bar's time: n weeks (or n months) per bar, then the bar
holding the result. A move that crosses the turn of the year can skip the year's short
last bar -- from the second week of 2019's last full 3W bar, one bar forward is the
first bar of 2020, and from April 2020 one 5M bar back is the two-month November bar
while from March it is the June bar.

PyneCore: `_dwm_walk_probe_ms` in `lib/__init__.py`.

## Daily and multi-day grid, backward

The grid is the trading days the session template schedules, counted from
January 1, holidays included (Christmas 2019 is a scheduled Wednesday without data on
CAPITALCOM symbols; the bar starting on it opens Tuesday evening); an nD bar groups n
of them, the year's last bar keeps the remainder. Inside a year a walk is plain
arithmetic on the day ordinals. The bar that crosses the turn of the year is placed
from the chart bar's own bar, and the rest of the walk continues from where it lands:

```
remaining = scheduled days from the chart day to the end of its nD bar (n - position)
mirror    = Dec 31 of the previous year - (day[remaining - 1] - Jan 1)
            (day[i] = the i-th scheduled day of the chart year, counted from 0;
             a mirror the template does not schedule rounds up to the next scheduled day)
landing   = mirror - (weeks of remaining beyond the first) x 2 weeks   (in scheduled days)
```

The reported bar is the previous year's bar holding the landing day; a landing that
rounded up into the chart year is the chart year's first bar. Further bars back continue
from the landing a bar length at a time, a landing in the chart year continuing from
the day after the previous year's last one, and a walk that reaches the start of the
previous year crosses again by the same rule.

What this looks like:

- A single day (`remaining` = 1) mirrors the year's first scheduled day: `Dec 31 -
  (F - Jan 1)`. Before a year that starts on Saturday (`F` = January 3) that is
  Wednesday December 29, two scheduled days short of Friday the 31st. Before a year that
  starts on Monday it is Sunday December 31, which rounds up to January 1 itself, so
  `time("D", timeframe_bars_back=1)` and `=2` on Tuesday 2001-01-02 are both Monday
  2001-01-01, and `=3` is Friday 2000-12-29. Every other weekday of January 1 gives the
  previous year's last scheduled day.
- The mirror reproduces the weekends of the chart year, not of the previous one. 2002
  starts on Tuesday: from Monday 2002-01-07 three days remain of the 7D bar, the third
  scheduled day (Thursday January 3) mirrors to Saturday December 29, which rounds up
  to Monday the 31st, so the walk lands in the bar of Friday 2001-12-28 rather than in
  the one holding Wednesday the 26th. The shift is at most two calendar days and
  depends on the weekday of January 1 and of the landing.
- Every whole week of `remaining` beyond the first costs two more weeks. 60D one bar
  back on 2003-01-02 (59 days remain: eleven weeks) lands ten weeks earlier than the
  mirror, in the bar of 2002-03-26, the fourth bar from the end of 2002; on 2003-01-14
  (49 days, nine weeks) in the bar of 2002-06-18. Across the first bar of the year the
  reported bar moves one bar per n/3 chart days, which is what made the crossing look
  non-linear before the rule was found. A seven-day template (BTCUSD) has no weekend
  to mirror and no weeks to pay for: its walk is plain arithmetic.
- The position inside the chart bar decides, not the distance walked: 14D three bars
  back from the second bar of 2010 lands where one bar back from the first bar does,
  two bar lengths further on.

The rule reproduces every measured value (1.9 million, see the table), including
offsets that cross two and three turns of the year.

## Daily and multi-day grid, forward

A bar that has not opened yet holds the trading day the template schedules n bars of
days ahead. Forward from the year's last scheduled day, the walk stops on December 31
when the template does not schedule it (a year ending on Saturday or Sunday) and
reports it as the next year's first scheduled day; the next step is that first day
itself. A year ending on a scheduled day walks straight on. Measured for D..5D,
offsets -1 and -2, no mismatch.

PyneCore: `_tv_day_walk` (backward) and `_scheduled_day_step` (forward) in
`lib/__init__.py`.

## Chart bars versus templates

An intraday chart bar belongs to the D/W/M period its scheduled close falls into, and
that close is the session end when the end cuts the bar short of its nominal span.
Measured on CAPITALCOM:US500 (240-minute chart, 2017-2026): the Monday session ends at
16:00 and the next one opens at once, so the 13:00 bar runs only to 16:00 and reports
Sunday 17:00 as its day, while the 16:00 bar starts Tuesday's trading day; a bar whose
nominal span reaches into a session that opens later than the bar (a 17:05 open on a
17:00 bar) starts the new day. PyneCore: `_chart_span_off_ms` in `lib/__init__.py`.

For symbols whose grid mode is `observed` (exchange-listed types), the nD/nW/nM bar
stamps come from the observed day counter, which counts the days present in the data.
TradingView's CAPITALCOM feeds count scheduled weekdays regardless of data, so AAPL,
GOLD and US500 stamp their 2D/3D bars differently from TradingView after a holiday
(the counter is one behind until the year resets) and their nW/nM bars one session late
when the period's first scheduled day has no data. This is the grid stamp, independent
of `timeframe_bars_back`; the walks above are consistent with whatever stamp the grid
gives.
