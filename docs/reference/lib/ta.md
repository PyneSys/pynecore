<!--
---
weight: 421
title: "ta"
description: "Technical analysis indicators and functions"
icon: "show_chart"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["ta", "library", "reference"]
---
-->

# ta

The `ta` namespace provides Pine Script-compatible technical analysis indicators for calculating moving averages, momentum oscillators, volatility measures, trend indicators, and statistical functions. These functions work with series data from OHLCV bars and are essential for strategy and indicator development.

## Quick Example

```python
from pynecore.lib import (
    close, high, low, open, volume, bar_index, ta, plot, strategy, script
)

@script.indicator(title="RSI and MACD Analysis", overlay=False)
def main():
    # Calculate RSI
    rsi_val: float = ta.rsi(close, 14)
    
    # Calculate MACD
    macd_line: float
    signal_line: float
    histogram: float
    macd_line, signal_line, histogram = ta.macd(close, 12, 26, 9)
    
    # Calculate Bollinger Bands for volatility
    bb_middle: float
    bb_upper: float
    bb_lower: float
    bb_middle, bb_upper, bb_lower = ta.bb(close, 20, 2.0)
    
    # Plot results
    plot(rsi_val, title="RSI(14)", color="blue")
    plot(macd_line, title="MACD", color="red")
    plot(signal_line, title="Signal", color="orange")
    plot(histogram, title="Histogram", color="gray")
```

## Moving Averages

### sma()

Simple moving average. Returns the arithmetic mean of the series over the last `length` bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Simple moving average of source.

```python
avg: float = ta.sma(close, 20)  # 20-bar SMA
```

### ema()

Exponential moving average. Applies decreasing weights to earlier bars using alpha = 2 / (length + 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Exponential moving average of source.

```python
ema_val: float = ta.ema(close, 9)  # 9-bar EMA
```

### rma()

RSI moving average (exponential weighted moving average with alpha = 1 / length). Used internally by RSI calculation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — RMA of source.

```python
rma_val: float = ta.rma(close, 14)  # 14-bar RMA
```

### wma()

Weighted moving average with decreasing weights in arithmetic progression. Most recent bar has highest weight.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Weighted moving average of source.

```python
wma_val: float = ta.wma(close, 10)  # Weighted average
```

### hma()

Hull Moving Average. A fast-responding moving average using multiple WMA calculations at different periods.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Hull moving average of source.

```python
hma_val: float = ta.hma(close, 20)  # Hull MA
```

### vwma()

Volume-weighted moving average. Incorporates volume to weight prices, giving more importance to high-volume bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Volume-weighted average of source.

```python
vwma_val: float = ta.vwma(close, 14)  # Volume-weighted MA
```

### swma()

Symmetrically weighted moving average with fixed length 4 and weights [1/6, 2/6, 2/6, 1/6]. Fast and responsive.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |

**Returns:** float — Symmetrically weighted average of source.

```python
swma_val: float = ta.swma(close)  # Fixed 4-bar symmetric MA
```

### alma()

Arnaud Legoux Moving Average. Uses Gaussian distribution as weights, providing a smooth and responsive moving average.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to average |
| `length` | int | Number of bars (must be > 0) |
| `offset` | float | Offset parameter (0-1), default 0.85 |
| `sigma` | float | Gaussian sigma, default 6.0 |
| `floor` | bool | Floor the offset calculation, default false |

**Returns:** float — Arnaud Legoux Moving Average.

```python
alma_val: float = ta.alma(close, 20, 0.85, 6.0)  # ALMA with default params
```

### linreg()

Linear regression curve. Finds the best-fit line through the source series using least squares method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to fit |
| `length` | int | Number of bars (must be > 0) |
| `offset` | int | Offset from current bar |

**Returns:** float — Linear regression value.

```python
linreg_val: float = ta.linreg(close, 20, 0)  # Linear regression line
```

## Momentum & Rate of Change

### rsi()

Relative Strength Index. Momentum oscillator measuring magnitude of price changes (0-100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — RSI value (typically 0-100).

```python
rsi_val: float = ta.rsi(close, 14)  # 14-period RSI
```

### mom()

Momentum. Simple difference between current source and source N bars ago.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to measure |
| `length` | int | Number of bars back (must be > 0) |

**Returns:** float — Change in source over length bars.

```python
momentum: float = ta.mom(close, 10)  # 10-bar momentum
```

### roc()

Rate of Change. Percentage change from N bars ago.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to measure |
| `length` | int | Number of bars back (must be > 0) |

**Returns:** float — Percentage change as decimal.

```python
roc_val: float = ta.roc(close, 12)  # 12-bar rate of change
```

### macd()

MACD (Moving Average Convergence Divergence). Returns MACD line, signal line, and histogram.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `fastlen` | int | Fast EMA length, typically 12 |
| `slowlen` | int | Slow EMA length, typically 26 |
| `siglen` | int | Signal line EMA length, typically 9 |

**Returns:** tuple[float, float, float] — (MACD line, Signal line, Histogram).

```python
macd_line, signal, histogram = ta.macd(close, 12, 26, 9)
```

### cmo()

Chande Momentum Oscillator. Normalized momentum oscillator (-100 to +100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — CMO value.

```python
cmo_val: float = ta.cmo(close, 14)  # 14-period CMO
```

### tsi()

True Strength Index. Momentum indicator using double exponential smoothing (-1 to +1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `short_length` | int | Short EMA length |
| `long_length` | int | Long EMA length |

**Returns:** float — TSI value (range -1 to 1).

```python
tsi_val: float = ta.tsi(close, 25, 13)  # TSI
```

### cog()

Center of Gravity. Statistics-based indicator using Fibonacci ratio to identify turning points.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — Center of gravity value.

```python
cog_val: float = ta.cog(close, 10)  # Center of gravity
```

### rci()

Rank Correlation Index. Measures directional consistency of price movements (-100 to +100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — RCI value.

```python
rci_val: float = ta.rci(close, 14)  # Rank correlation index
```

## Volatility & Bands

### atr()

Average True Range. Measures volatility as the RMA of true range over N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `length` | int | Period length (must be > 0) |

**Returns:** float — Average true range.

```python
atr_val: float = ta.atr(14)  # 14-period ATR
```

### tr()

True Range. Maximum of (high - low), (abs(high - close[1])), or (abs(low - close[1])). Accounts for gaps.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle_na` | bool | Handle NA values, default true |

**Returns:** float — True range value.

```python
tr_val: float = ta.tr(True)  # True range with NA handling
```

### stdev()

Standard Deviation. Measures dispersion of prices around their mean.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `biased` | bool | Use biased (N) or unbiased (N-1) estimator |

**Returns:** float — Standard deviation.

```python
stdev_val: float = ta.stdev(close, 20, False)  # 20-bar standard deviation
```

### variance()

Variance. Squared standard deviation, measuring dispersion around the mean.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `biased` | bool | Use biased (N) or unbiased (N-1) estimator |

**Returns:** float — Variance.

```python
var_val: float = ta.variance(close, 20, False)  # Variance
```

### dev()

Deviation. Difference between a value and its moving average.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — Deviation from SMA.

```python
dev_val: float = ta.dev(close, 20)  # Deviation from 20-bar MA
```

### bb()

Bollinger Bands. Returns middle band, upper band, and lower band (distance is mult × stdev).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `mult` | float | Standard deviation multiplier (must be > 0) |

**Returns:** tuple[float, float, float] — (Middle band, Upper band, Lower band).

```python
middle, upper, lower = ta.bb(close, 20, 2.0)  # Bollinger Bands with 2 stdev
```

### bbw()

Bollinger Bands Width. Ratio of band width to middle band, expressed as percentage.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `mult` | float | Standard deviation multiplier (must be > 0) |

**Returns:** float — BB width as percentage.

```python
bb_width: float = ta.bbw(close, 20, 2.0)  # Band width %
```

### kc()

Keltner Channels. Returns middle band, upper band, and lower band using ATR for distance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series for middle band |
| `length` | int | Period length (must be > 0) |
| `mult` | float | ATR multiplier (must be > 0) |
| `useTrueRange` | bool | Use true range or high-low |

**Returns:** tuple[float, float, float] — (Middle, Upper, Lower).

```python
kc_mid, kc_up, kc_low = ta.kc(close, 20, 2.0, True)  # Keltner Channels
```

### kcw()

Keltner Channels Width. Ratio of KC width to middle band, expressed as percentage.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series for middle band |
| `length` | int | Period length (must be > 0) |
| `mult` | float | ATR multiplier (must be > 0) |
| `useTrueRange` | bool | Use true range or high-low |

**Returns:** float — KC width as percentage.

```python
kc_width: float = ta.kcw(close, 20, 2.0, True)  # Keltner width %
```

## Trend & Direction

### cci()

Commodity Channel Index. Oscillator measuring deviation from typical price (-100 to +100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — CCI value.

```python
cci_val: float = ta.cci(close, 20)  # 20-period CCI
```

### dmi()

Directional Movement Index. Returns +DI, -DI, and ADX.

| Parameter | Type | Description |
|-----------|------|-------------|
| `diLength` | int | DI period (must be > 0) |
| `adxSmoothing` | int | ADX smoothing period (must be > 0) |

**Returns:** tuple[float, float, float] — (+DI, -DI, ADX).

```python
di_plus, di_minus, adx = ta.dmi(14, 14)  # DMI with 14-period DI and ADX
```

### supertrend()

Supertrend. Trend-following indicator using ATR, returns trend line and direction (1 or -1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `factor` | float | Multiplier for ATR |
| `atrPeriod` | int | ATR period (must be > 0) |

**Returns:** tuple[float, float] — (Supertrend line, Direction: 1 or -1).

```python
st_line, st_dir = ta.supertrend(3.0, 10)  # Supertrend with 3x ATR
```

### sar()

Parabolic SAR. Stop and Reverse indicator designed by J. Welles Wilder.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | float | Initial SAR value, typically 0.02 |
| `inc` | float | SAR increment step, typically 0.02 |
| `max` | float | Maximum acceleration factor, typically 0.2 |

**Returns:** float — SAR value.

```python
sar_val: float = ta.sar(0.02, 0.02, 0.2)  # Parabolic SAR
```

### rising()

Test if a series is rising. Returns true if current value is higher than any prior value in the last N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to test |
| `length` | int | Number of bars to check (must be > 0) |

**Returns:** bool — True if rising over length bars.

```python
is_rising: bool = ta.rising(close, 5)  # Rising for 5 bars
```

### falling()

Test if a series is falling. Returns true if current value is lower than any prior value in the last N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to test |
| `length` | int | Number of bars to check (must be > 0) |

**Returns:** bool — True if falling over length bars.

```python
is_falling: bool = ta.falling(close, 5)  # Falling for 5 bars
```

## Crossovers & Signals

### cross()

Cross. Returns true when two series have crossed (in either direction).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source1` | float | First series |
| `source2` | float | Second series |

**Returns:** bool — True if the two series crossed.

```python
crossed: bool = ta.cross(sma_fast, sma_slow)  # Either direction
```

### crossover()

Crossover. Returns true when source1 crosses above source2.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source1` | float | First series |
| `source2` | float | Second series |

**Returns:** bool — True if source1 > source2 and source1[1] <= source2[1].

```python
cross_up: bool = ta.crossover(close, ma)  # Price crosses above MA
```

### crossunder()

Crossunder. Returns true when source1 crosses below source2.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source1` | float | First series |
| `source2` | float | Second series |

**Returns:** bool — True if source1 < source2 and source1[1] >= source2[1].

```python
cross_down: bool = ta.crossunder(close, ma)  # Price crosses below MA
```

### barssince()

Bars Since. Counts bars since a condition was last true.

| Parameter | Type | Description |
|-----------|------|-------------|
| `condition` | bool | Condition to check |

**Returns:** int — Number of bars since condition was true (or NA).

```python
bars_since_cross: int = ta.barssince(ta.crossover(close, sma))
```

### change()

Change. Difference between current value and value N bars ago.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float/bool | Series to measure |
| `length` | int | Number of bars back, default 1 |

**Returns:** float/bool — Difference (or bool for bool source).

```python
change_val: float = ta.change(close, 1)  # 1-bar change
```

## Highs, Lows & Extremes

### highest()

Highest. Highest value of a series over the last N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Highest value.

```python
highest_20: float = ta.highest(high, 20)  # 20-bar high
```

### highestbars()

Highest Bars. Number of bars since the highest value occurred.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** int — Bars since highest value.

```python
bars_since_high: int = ta.highestbars(high, 20)  # Bars since 20-bar high
```

### lowest()

Lowest. Lowest value of a series over the last N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Lowest value.

```python
lowest_20: float = ta.lowest(low, 20)  # 20-bar low
```

### lowestbars()

Lowest Bars. Number of bars since the lowest value occurred.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** int — Bars since lowest value.

```python
bars_since_low: int = ta.lowestbars(low, 20)  # Bars since 20-bar low
```

### max()

All-Time Max. Highest value from the beginning of the chart to current bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |

**Returns:** float — All-time maximum.

```python
all_time_high: float = ta.max(high)  # Highest high ever
```

### min()

All-Time Min. Lowest value from the beginning of the chart to current bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |

**Returns:** float — All-time minimum.

```python
all_time_low: float = ta.min(low)  # Lowest low ever
```

### range()

Range. Difference between highest and lowest in a series over N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Max minus min over length bars.

```python
price_range: float = ta.range(close, 20)  # 20-bar range
```

## Statistical & Distribution

### correlation()

Correlation Coefficient. Measures how two series move together (-1 to +1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source1` | float | First series |
| `source2` | float | Second series |
| `length` | int | Period length (must be > 0) |

**Returns:** float — Correlation (-1 to 1).

```python
corr: float = ta.correlation(close, ta.ema(close, 9), 20)  # Correlation
```

### percentile_linear_interpolation()

Percentile using linear interpolation between nearest ranks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `percentage` | float | Percentile (0-100) |

**Returns:** float — P-th percentile.

```python
p_90: float = ta.percentile_linear_interpolation(close, 20, 90)  # 90th percentile
```

### percentile_nearest_rank()

Percentile using Nearest Rank method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |
| `percentage` | float | Percentile (0-100) |

**Returns:** float — P-th percentile.

```python
p_50: float = ta.percentile_nearest_rank(close, 20, 50)  # Median
```

### percentrank()

Percent Rank. Percentage of prior values less than or equal to current value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — Percent rank (0-100).

```python
pr: float = ta.percentrank(close, 20)  # Percent rank
```

### median()

Median. Middle value of a series over N bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Median value over length bars.

```python
med: float = ta.median(close, 20)  # 20-bar median
```

### mode()

Mode. Most frequently occurring value in a series. Returns smallest value if tie.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Number of bars (must be > 0) |

**Returns:** float — Most frequent value over length bars.

```python
mode_val: float = ta.mode(close, 20)  # 20-bar mode
```

## Volume Indicators

### mfi()

Money Flow Index. Volume-weighted oscillator similar to RSI (0-100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `length` | int | Period length (must be > 0) |

**Returns:** float — MFI value (0-100).

```python
mfi_val: float = ta.mfi(close, 14)  # 14-period Money Flow Index
```

### obv()

On Balance Volume. Cumulative indicator adding/subtracting volume based on price direction.

**Returns:** float — OBV value.

```python
obv_val: float = ta.obv()  # On Balance Volume
```

### vwap()

Volume Weighted Average Price. Returns VWAP line, and optionally upper and lower bands when `stdev_mult` is specified.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze |
| `anchor` | bool \| None | Anchor to session, default None |
| `stdev_mult` | float \| None | Standard deviation multiplier for bands, default None |

**Returns:** float — VWAP value. When `stdev_mult` is set, returns tuple[float, float, float] — (VWAP, Upper band, Lower band).

```python
vwap_val: float = ta.vwap(close)                        # Basic VWAP
vwap_line, upper, lower = ta.vwap(close, None, 2.0)     # VWAP with bands
```

### pvt()

Price-Volume Trend. Volume-based indicator showing price momentum weighted by volume.

**Returns:** float — PVT value.

```python
pvt_val: float = ta.pvt()  # Price-Volume Trend
```

### accdist()

Accumulation/Distribution Index. Relates closing price to the range and weights by volume.

**Returns:** float — A/D value.

```python
ad_val: float = ta.accdist()  # Accumulation/Distribution
```

### wad()

Williams Accumulation/Distribution. Similar to A/D with different calculation.

**Returns:** float — WAD value.

```python
wad_val: float = ta.wad()  # Williams Accumulation/Distribution
```

### pvi()

Positive Volume Index. Volume-based indicator focusing on days with increasing volume.

**Returns:** float — PVI value.

```python
pvi_val: float = ta.pvi()  # Positive Volume Index
```

### nvi()

Negative Volume Index. Volume-based indicator focusing on days with decreasing volume.

**Returns:** float — NVI value.

```python
nvi_val: float = ta.nvi()  # Negative Volume Index
```

### iii()

Intraday Intensity Index. Relates close position to the daily range and volume.

**Returns:** float — III value.

```python
iii_val: float = ta.iii()  # Intraday Intensity Index
```

### wvad()

Williams Variable Accumulation/Distribution. A variant of WAD with different weighting.

**Returns:** float — WVAD value.

```python
wvad_val: float = ta.wvad()  # Williams Variable Accumulation/Distribution
```

## Stochastic & Price-Position

### stoch()

Stochastic. Price position indicator showing where current price lies within its range (0-100).

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series (typically close) |
| `high` | float | High series |
| `low` | float | Low series |
| `length` | int | Period length (must be > 0) |

**Returns:** float — Stochastic value (0-100).

```python
stoch_val: float = ta.stoch(close, high, low, 14)  # 14-period Stochastic
```

### wpr()

Williams %R. Price position oscillator similar to stochastic (-100 to 0).

| Parameter | Type | Description |
|-----------|------|-------------|
| `length` | int | Period length (must be > 0) |

**Returns:** float — Williams %R value.

```python
wpr_val: float = ta.wpr(14)  # 14-period Williams %R
```

## Pivot Points

### pivot_point_levels()

Pivot Point Levels. Calculates support and resistance levels using specified pivot point method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | str | Pivot point type (e.g., "standard", "fibonacci") |
| `anchor` | str | Anchor type for calculation |
| `developing` | bool | Include developing period |

**Returns:** list[float] — Array of 11 pivot point levels.

```python
pivots: list[float] = ta.pivot_point_levels("standard", "day", False)
```

### pivothigh()

Pivot High. Returns price of a pivot high point (local maximum). Has two overloads: one using built-in `high` series, and one accepting a custom source series.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze (optional, defaults to `high`) |
| `leftbars` | int | Number of bars to the left |
| `rightbars` | int | Number of bars to the right |

**Returns:** float — Pivot high price, or NA if no pivot found.

```python
ph: float = ta.pivothigh(5, 5)              # Using built-in high
ph2: float = ta.pivothigh(close, 10, 10)    # Custom source
```

### pivotlow()

Pivot Low. Returns price of a pivot low point (local minimum). Has two overloads: one using built-in `low` series, and one accepting a custom source series.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to analyze (optional, defaults to `low`) |
| `leftbars` | int | Number of bars to the left |
| `rightbars` | int | Number of bars to the right |

**Returns:** float — Pivot low price, or NA if no pivot found.

```python
pl: float = ta.pivotlow(5, 5)              # Using built-in low
pl2: float = ta.pivotlow(close, 10, 10)    # Custom source
```

## Aggregation & Summary

### cum()

Cumulative. Running total sum of all values from the start of the chart.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | float | Series to sum |

**Returns:** float — Cumulative sum.

```python
cumsum: float = ta.cum(volume)  # Total volume from bar 1
```

### valuewhen()

Value When. Returns the value of source at the nth most recent occurrence of a condition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `condition` | bool | Condition to find |
| `source` | float | Value to retrieve |
| `occurrence` | int | Nth occurrence (1 = most recent) |

**Returns:** float — Source value when condition was true.

```python
val: float = ta.valuewhen(ta.crossover(close, sma), close, 1)
```

---

## Compatibility & Notes

All functions in the `ta` namespace return `float | NA[float]`, `int | NA[int]`, `bool`, or tuple variants when a value is not yet available (such as during the warmup period). All calculations are compatible with Pine Script v6 and produce equivalent results to TradingView when given identical inputs.

Functions returning tuples (like `macd()`, `bb()`, `dmi()`) can be unpacked in Python:
```python
line1, line2, line3 = ta.macd(close, 12, 26, 9)
```

Volume-based indicators (`obv`, `pvt`, `accdist`, etc.) are accessed as parameterless functions due to their use of built-in OHLCV data.