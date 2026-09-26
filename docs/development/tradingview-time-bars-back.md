# `time()` / `time_close()` with `timeframe_bars_back` on D, W and M grids

How PyneCore walks a requested daily, weekly or monthly grid when `timeframe_bars_back`
is not zero, and what was measured on TradingView to arrive at it. The intraday walk
(sessions, buckets) is documented in the `time()` docstring and in
`_session_bar_bounds`; this note covers the calendar grids only.

## What was measured

Probes on CAPITALCOM:US500 (daily, 1999-2026, 27 turns of the year), BTCUSD (daily
and 60-minute), EURUSD, AAPL and GOLD (daily), 2026-09-26:

| Grid | Multipliers | Offsets | Values compared | Mismatches |
|------|-------------|---------|-----------------|------------|
| nW   | 1-18, 20, 25, 26, 30, 40, 52 | -2 .. 20 | 1 482 278 | 0 |
| nM   | 5-11 | -1, 1, 2 | 180 202 | 0 |
| nD   | 2-6, 10 | -2 .. 4 | 126 217 (daily symbols) | see below |

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

## Multi-day grid

The grid pairs the trading days the session template schedules, counted from
January 1, holidays included (Christmas 2019 is a scheduled Wednesday without data on
CAPITALCOM symbols; the bar starting on it opens Tuesday evening). A walk moves the
chart bar's trading day by n scheduled days per bar and reports the bar holding the
result. This reproduces every value on a seven-day template (BTCUSD) and every value
inside a year on a Monday to Friday template.

Not reproduced: on a Monday to Friday template, walks that cross the turn of the year
deviate on some of the first days of January. The deviation depends on the weekday of
January 1 and on the position of the target inside the last week of the previous year
(0.2-1% of the daily bars for offset 1 on 2D..10D, more for larger offsets). The
measured pattern by the weekday of January 1: Saturday -- the target lands two calendar
days earlier; Monday and Tuesday -- two calendar days later (Monday-start years then
report the current bar itself once); Wednesday, Thursday and Sunday -- no deviation;
Friday -- two days earlier except from the last day of the year. The probe data lives in
the measurement session; the rule behind it is not understood, so it is not implemented.

## Chart bars versus templates

For symbols whose grid mode is `observed` (exchange-listed types), the nD/nW/nM bar
stamps come from the observed day counter, which counts the days present in the data.
TradingView's CAPITALCOM feeds count scheduled weekdays regardless of data, so AAPL,
GOLD and US500 stamp their 2D/3D bars differently from TradingView after a holiday
(the counter is one behind until the year resets) and their nW/nM bars one session late
when the period's first scheduled day has no data. This is the grid stamp, independent
of `timeframe_bars_back`; the walks above are consistent with whatever stamp the grid
gives.
