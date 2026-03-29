<!--
---
weight: 422
title: "math"
description: "Mathematical functions and constants"
icon: "calculate"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["math", "library", "reference"]
---
-->

# math

Mathematical functions and constants for numerical operations. The `math` namespace provides basic arithmetic, trigonometry, logarithmic, and utility functions for use in indicators and strategies. All functions handle `NA` (not available) values gracefully, returning `NA` if any input is `NA`.

## Quick Example

```python
from pynecore.lib import math, close, bar_index, script

@script.indicator(title="Math Functions Demo", overlay=True)
def main():
    # Basic rounding and absolute value
    rounded: float = math.round(close, 2)  # Round price to 2 decimals
    magnitude: float = math.abs(close - close[1])  # Price change magnitude
    
    # Find highest and lowest in a range
    high_val: float = math.max(close, close[1], close[2])
    low_val: float = math.min(close, close[1], close[2])
    
    # Trigonometric operation on normalized data
    normalized: float = (close - math.min(close)) / (math.max(close) - math.min(close))
    sine_val: float = math.sin(normalized * math.pi)
    
    # Logarithmic analysis
    if close > 0:
        log_val: float = math.log(close)
```

## Functions

### abs()

Returns the absolute value of a number.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |

**Returns:** `float` or `NA[float]`

```python
magnitude: float = math.abs(-5)  # 5.0
```

### sign()

Returns the sign of a number: `1.0` for positive, `-1.0` for negative, `0.0` for zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |

**Returns:** `float` or `NA[float]`

```python
direction: float = math.sign(close - close[1])  # 1.0, -1.0, or 0.0
```

### ceil()

Rounds a number up to the nearest integer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |

**Returns:** `float` or `NA[float]`

```python
rounded_up: float = math.ceil(4.3)  # 5.0
```

### floor()

Rounds a number down to the nearest integer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |

**Returns:** `float` or `NA[float]`

```python
rounded_down: float = math.floor(4.9)  # 4.0
```

### round()

Rounds a number to the nearest integer, or to a specified number of decimal places.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |
| `precision` | `int` (optional) | Number of decimal places; if omitted, rounds to nearest integer |

**Returns:** `float` or `NA[float]`

```python
rounded: float = math.round(close, 2)  # Round to 2 decimal places
as_int: float = math.round(4.7)  # 5.0
```

### round_to_mintick()

Rounds a number to the symbol's minimum tick value with ties rounding up.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A number |

**Returns:** `float` or `NA[float]`

```python
tick_price: float = math.round_to_mintick(100.5467)  # Rounded to mintick
```

### max()

Returns the largest of multiple values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `numbers` | `float` or `int` (variadic) | Two or more numbers |

**Returns:** `float` or `NA[float]`

```python
highest: float = math.max(close, close[1], close[2])  # 102.5
```

### min()

Returns the smallest of multiple values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `numbers` | `float` or `int` (variadic) | Two or more numbers |

**Returns:** `float` or `NA[float]`

```python
lowest: float = math.min(close, close[1], close[2])  # 98.5
```

### avg()

Returns the average of multiple values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `numbers` | `float` or `int` (variadic) | Two or more numbers |

**Returns:** `float` or `NA[float]`

```python
average: float = math.avg(close, close[1], close[2])  # 100.5
```

### sum()

Returns the sum of the last `length` values of a source.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `float` or series | The source series |
| `length` | `int` | Number of bars to sum |

**Returns:** `float` or `NA[float]`

```python
volume_sum: float = math.sum(volume, 20)  # Sum of volume over last 20 bars
```

### sqrt()

Returns the square root of a number. Returns `NA` if the number is negative.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A non-negative number |

**Returns:** `float` or `NA[float]`

```python
root: float = math.sqrt(16)  # 4.0
```

### pow()

Raises a base number to the power of an exponent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `float` or `int` | The base |
| `exponent` | `float` or `int` | The exponent |

**Returns:** `float` or `NA[float]`

```python
squared: float = math.pow(5, 2)  # 25.0
cubed: float = math.pow(2, 3)  # 8.0
```

### exp()

Returns e (Euler's number) raised to the power of a number.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | The exponent |

**Returns:** `float` or `NA[float]`

```python
result: float = math.exp(1)  # 2.718...
```

### log()

Returns the natural logarithm of a number. The input must be positive.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A positive number |

**Returns:** `float` or `NA[float]`

```python
ln_val: float = math.log(2.718)  # Approximately 1.0
```

### log10()

Returns the base-10 logarithm of a number. The input must be positive.

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | `float` or `int` | A positive number |

**Returns:** `float` or `NA[float]`

```python
log_val: float = math.log10(100)  # 2.0
```

### sin()

Returns the trigonometric sine of an angle (in radians).

| Parameter | Type | Description |
|-----------|------|-------------|
| `angle` | `float` or `int` | An angle in radians |

**Returns:** `float` or `NA[float]`

```python
sine: float = math.sin(math.pi / 2)  # 1.0
```

### cos()

Returns the trigonometric cosine of an angle (in radians).

| Parameter | Type | Description |
|-----------|------|-------------|
| `angle` | `float` or `int` | An angle in radians |

**Returns:** `float` or `NA[float]`

```python
cosine: float = math.cos(0)  # 1.0
```

### tan()

Returns the trigonometric tangent of an angle (in radians).

| Parameter | Type | Description |
|-----------|------|-------------|
| `angle` | `float` or `int` | An angle in radians |

**Returns:** `float` or `NA[float]`

```python
tangent: float = math.tan(math.pi / 4)  # 1.0
```

### asin()

Returns the arcsine (inverse sine) of a value in radians. The input must be in the range [-1, 1].

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `float` or `int` | A value in [-1, 1] |

**Returns:** `float` or `NA[float]` — angle in radians, range [-π/2, π/2]

```python
angle: float = math.asin(1)  # π/2
```

### acos()

Returns the arccosine (inverse cosine) of a value in radians. The input must be in the range [-1, 1].

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `float` or `int` | A value in [-1, 1] |

**Returns:** `float` or `NA[float]` — angle in radians, range [0, π]

```python
angle: float = math.acos(0)  # π/2
```

### atan()

Returns the arctangent (inverse tangent) of a value in radians.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `float` or `int` | Any real number |

**Returns:** `float` or `NA[float]` — angle in radians, range [-π/2, π/2]

```python
angle: float = math.atan(1)  # π/4
```

### toradians()

Converts an angle from degrees to radians.

| Parameter | Type | Description |
|-----------|------|-------------|
| `degrees` | `float` or `int` | An angle in degrees |

**Returns:** `float` or `NA[float]` — angle in radians

```python
radians: float = math.toradians(180)  # π
```

### todegrees()

Converts an angle from radians to degrees.

| Parameter | Type | Description |
|-----------|------|-------------|
| `radians` | `float` or `int` | An angle in radians |

**Returns:** `float` or `NA[float]` — angle in degrees

```python
degrees: float = math.todegrees(math.pi)  # 180.0
```

### random()

Returns a pseudo-random value between a minimum and maximum. The sequence is deterministic if a seed is provided.

| Parameter | Type | Description |
|-----------|------|-------------|
| `min` | `float` or `int` (optional) | Lower bound (default: 0) |
| `max` | `float` or `int` (optional) | Upper bound (default: 1) |
| `seed` | `int` (optional) | Seed for reproducible randomness |

**Returns:** `float` or `NA[float]`

```python
rand_val: float = math.random(0, 100)  # Random value between 0 and 100
```

## Constants

| Name | Value | Description |
|------|-------|-------------|
| `math.e` | 2.718... | Euler's number |
| `math.pi` | 3.141... | Archimedes' constant (π) |
| `math.phi` | 1.618... | Golden ratio |
| `math.rphi` | 0.618... | Golden ratio conjugate (1/φ) |

## Compatibility

All 24 functions and 4 constants are fully implemented in PyneCore. All functions handle `NA` (not available) values correctly, returning `NA` if any input is `NA`.