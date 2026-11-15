<!--
---
weight: 431
title: "Run-Time Errors"
description: "Look up the error codes a PyneCore run stops with: what each R code means, the message it shows and what to change in the script or its data."
icon: "error"
date: "2026-11-15"
lastmod: "2026-11-15"
draft: false
toc: true
categories: ["Reference"]
tags: ["error-codes", "runtime", "debugging"]
---
-->

# Run-Time Errors

<a id="R0001"></a>

## R0001: Division by zero

| Severity | PyneCore raises     |
|----------|---------------------|
| error    | `ZeroDivisionError` |

Message:

```text
Division by zero.
```

A division, `%` or floor division in Pyne code had a divisor of 0. Pine's own
operators answer `na` here; plain Python arithmetic in Pyne code stops the
run, as it does in PyneCore. Guard the divisor, or use the Pine operator.

<a id="R0002"></a>

## R0002: An na value converted to an integer

| Severity | PyneCore raises           |
|----------|---------------------------|
| error    | `ValueError`, `TypeError` |

Message:

```text
An `na` value cannot be converted to an integer.
```

A value that is `na` reached a place that needs a whole number: an index, a
size, a count or Python's `int()` / `round()`. Check the value with `na()` or
replace it with `nz()` first.

<a id="R0003"></a>

## R0003: An infinite value converted to an integer

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `OverflowError` |

Message:

```text
An infinite value cannot be converted to an integer.
```

A value that is positive or negative infinity reached a place that needs a
whole number: an index, a size, a count, a bar coordinate or Python's
`int()` / `round()`. Infinity usually comes from a division by a value close
to 0.

<a id="R0004"></a>

## R0004: Assertion failed

| Severity | PyneCore raises  |
|----------|------------------|
| error    | `AssertionError` |

Message:

```text
Assertion failed. {message}
```

An `assert` statement of the script found its condition false. The text after
it is the assertion's own message.

<a id="R0005"></a>

## R0005: runtime.error()

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
{message}
```

The script called `runtime.error()`. The message is the script's own.

<a id="R0006"></a>

## R0006: Argument must be greater than 0

| Severity | PyneCore raises                |
|----------|--------------------------------|
| error    | `AssertionError`, `ValueError` |

Message:

```text
`{function}`: `{parameter}` must be greater than 0, but it is {value}.
```

A built-in got a length, a period or a multiplier of 0 or less. Pine refuses
these values too.

<a id="R0007"></a>

## R0007: Argument must not be negative

| Severity | PyneCore raises  |
|----------|------------------|
| error    | `AssertionError` |

Message:

```text
`{function}`: `{parameter}` must not be negative, but it is {value}.
```

A built-in got a negative count, size or offset where only 0 or more is
meaningful.

<a id="R0008"></a>

## R0008: Argument must not be 0

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}`: `{parameter}` must not be 0.
```

A loop or range got a step of 0, which would never reach its end.

<a id="R0009"></a>

## R0009: Percentage out of range

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}`: `{parameter}` must be between 0 and 100, but it is {value}.
```

A percentile or rank built-in got a percentage outside 0..100.

<a id="R0010"></a>

## R0010: Index out of range

| Severity | PyneCore raises            |
|----------|----------------------------|
| error    | `IndexError`, `ValueError` |

Message:

```text
`{function}`: index {index} is out of range for a size of {size}.
```

An array, matrix or series view was read or written at an index it does not
have. Indexes start at 0; a negative index counts from the end where the
built-in allows it.

<a id="R0011"></a>

## R0011: Empty collection

| Severity | PyneCore raises                                |
|----------|------------------------------------------------|
| error    | `IndexError`, `RuntimeError`, `AssertionError` |

Message:

```text
`{function}` needs at least one element, but there is none.
```

The built-in takes an element out of an array, a list or a matrix, and there
is no element to take. Check the size first.

<a id="R0012"></a>

## R0012: Sizes do not match

| Severity | PyneCore raises                |
|----------|--------------------------------|
| error    | `ValueError`, `AssertionError` |

Message:

```text
`{function}`: the sizes do not match ({left} and {right}).
```

The built-in combines two arrays or matrices whose sizes or dimensions must
agree, and they do not.

<a id="R0013"></a>

## R0013: Matrix is not square

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}` needs a square matrix, but it has {rows} rows and {columns} columns.
```

The inverse, the determinant, the trace, a power and the eigen decomposition
exist only for a matrix with as many rows as columns.

<a id="R0014"></a>

## R0014: Matrix is singular

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}`: the matrix is singular.
```

A singular matrix has no inverse. Its determinant is 0.

<a id="R0015"></a>

## R0015: Calculation did not converge

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}` did not converge.
```

An iterative matrix calculation reached its step limit without a result. The
matrix usually holds extreme or `na` values.

<a id="R0016"></a>

## R0016: An na object

| Severity | PyneCore raises               |
|----------|-------------------------------|
| error    | `AttributeError`, `TypeError` |

Message:

```text
`{name}` cannot take an `na` object.
```

A field was written, a method called, a length taken or a loop run on an
object, array or matrix that is `na`. Create the object before it is used,
or check it with `na()`.

<a id="R0017"></a>

## R0017: Variable read before it is assigned

| Severity | PyneCore raises     |
|----------|---------------------|
| error    | `UnboundLocalError` |

Message:

```text
`{name}` is read before it is assigned on this path.
```

On the path the script took, the variable was not assigned before this read.
Give it a value before the `if` or loop that assigns it.

<a id="R0018"></a>

## R0018: Resource limit exceeded

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
The run exceeded the limit of {limit} {resource}.
```

The script created more objects, drawings or memory than the runtime allows,
or a regular expression needed too many steps. The limit protects the run
from running out of memory or time. PyneCore takes the limit of
`request.security` contexts from the `PYNESYS_MAX_SECURITY_CONTEXTS`
environment variable.

<a id="R0019"></a>

## R0019: Invalid timeframe

| Severity | PyneCore raises                |
|----------|--------------------------------|
| error    | `AssertionError`, `ValueError` |

Message:

```text
`{timeframe}` is not a valid timeframe.
```

A timeframe is a positive multiplier and a unit: `1`, `60`, `1D`, `1W`, `1M`,
`30S`. An empty string, an unknown unit or a multiplier of 0 is not one.

<a id="R0020"></a>

## R0020: Tick timeframe

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
The tick timeframe `{timeframe}` is not supported here.
```

A tick timeframe (`1T`) has no fixed length in time, so this built-in cannot
use it.

<a id="R0021"></a>

## R0021: Invalid session

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{session}` is not a valid session.
```

A session is one or more `HHMM-HHMM` ranges, optionally followed by `:` and
the days `1` (Sunday) to `7` (Saturday), separated by commas.

<a id="R0022"></a>

## R0022: Invalid date

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{text}` is not a valid date.
```

`timestamp()` takes a date in one of its formats, such as `2024-01-31`,
`2024-01-31T09:30:00+00:00` or `31 Jan 2024 09:30 +0000`. The date must
exist: a month has 1..12, a day as many as its month.

<a id="R0023"></a>

## R0023: Invalid format pattern

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{pattern}` is not a valid format for `{function}`.
```

The pattern of `str.format()`, `str.format_time()` or a number format has a
placeholder or a letter the built-in does not know, or is cut short.

<a id="R0024"></a>

## R0024: Invalid regular expression

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `re.error`      |

Message:

```text
`{pattern}` is not a valid regular expression: {reason}.
```

`str.match()` takes a regular expression; this one does not compile.

<a id="R0025"></a>

## R0025: Unknown time zone

| Severity | PyneCore raises                                |
|----------|------------------------------------------------|
| error    | `pynecore.core.datetime.TimezoneNotFoundError` |

Message:

```text
`{timezone}` is not a known time zone.
```

A time zone is an IANA name, such as `America/New_York`, or a UTC offset,
such as `UTC-5` or `GMT+0530`.

<a id="R0026"></a>

## R0026: No data for request.security

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`request.security` needs the data of {symbol} on the {timeframe} timeframe, and none was given.
```

A `request.security()` call asks for another symbol or timeframe. The run
reads it from the data handed to it; give the data of this symbol and
timeframe too: `pyne run` takes it with `--security`, a `ScriptRunner` with
its `security_data` parameter (`security_data={'EXCHANGE:SYMBOL': 'path/to/data.ohlcv'}`).

<a id="R0027"></a>

## R0027: Lower timeframe request of a higher timeframe

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
`request.security_lower_tf` cannot request `{timeframe}`, which is higher than the script's `{chart}`.
```

`request.security_lower_tf()` reads the bars inside each bar of the script,
so its timeframe must be lower. Pass `ignore_invalid_timeframe = true` to get
`na` instead of this error.

<a id="R0028"></a>

## R0028: request.security that varies in a loop

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
A `request.security` expression changes between the iterations of a loop.
```

A `request.security()` call inside a loop must request the same expression
on every iteration.

<a id="R0029"></a>

## R0029: Strategy value is not finite

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}`: {what} is not a finite number.
```

A strategy order or setting got `na` or an infinite quantity, price or
offset, where the strategy engine needs a number.

<a id="R0030"></a>

## R0030: Strategy read inside request.security

| Severity | PyneCore raises  |
|----------|------------------|
| error    | `AttributeError` |

Message:

```text
`{function}` reads the strategy inside `request.security`, which has no strategy.
```

The expression of a `request.security()` call is evaluated in the requested
context, where there are no orders and no positions.

<a id="R0031"></a>

## R0031: line.get_price on a time-based line

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
`line.get_price` needs a line created with `xloc.bar_index`.
```

A line whose x coordinates are times has no bar index to read a price at.

<a id="R0032"></a>

## R0032: Invalid operand

| Severity | PyneCore raises               |
|----------|-------------------------------|
| error    | `TypeError`, `AssertionError` |

Message:

```text
`{operation}` cannot take {operand}.
```

An operation of Pyne code got a value of a type it does not take, such as a
function where a number is needed.

<a id="R0033"></a>

## R0033: Number out of range

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `OverflowError` |

Message:

```text
`{function}`: a result is too large for a number.
```

An intermediate result of the calculation is beyond the largest number a
float holds.

<a id="R0034"></a>

## R0034: Invalid decimal operation

| Severity | PyneCore raises            |
|----------|----------------------------|
| error    | `decimal.InvalidOperation` |

Message:

```text
`{function}`: the decimal calculation has no result ({why}).
```

A calculation in decimal arithmetic, such as a sum of volumes or a currency
amount, met infinities of both signs or a value too large to round.

<a id="R0035"></a>

## R0035: Not available

| Severity | PyneCore raises       |
|----------|-----------------------|
| error    | `NotImplementedError` |

Message:

```text
`{function}` is not available.
```

The built-in has no data source in this runtime. With
`ignore_invalid_symbol = true` it answers `na` instead.

<a id="R0036"></a>

## R0036: Script timeframe does not fit the data

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
The script timeframe `{timeframe}` does not fit the data on `{chart}`: {why}.
```

A script timeframe folds the bars of the data into longer bars. It must be
equal to or higher than the data's timeframe, a whole multiple of it, and
aligned with the data's bars.

<a id="R0037"></a>

## R0037: Invalid data

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{where}`: the data is invalid: {why}.
```

A data file, an imported row or a symbol description is inconsistent: two
sources under one key, a symbol description of another symbol, a timestamp
that does not follow the previous one, or prices that are not finite.

<a id="R0038"></a>

## R0038: Invalid drawing argument

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `TypeError`     |

Message:

```text
`{function}`: `{parameter}` cannot be {value}.
```

A drawing built-in got a value it does not take, such as a fractional text
formatting flag.

<a id="R0039"></a>

## R0039: No such method

| Severity | PyneCore raises               |
|----------|-------------------------------|
| error    | `AssertionError`, `TypeError` |

Message:

```text
A `{type}` has no method `{method}`.
```

A method is called on an object whose type has no method of this name: the
module of the type, the calling module and the libraries it imports export
none. A method call on an `na` object of such a type answers the `na` itself.

<a id="R0040"></a>

## R0040: Arguments in the wrong order

| Severity | PyneCore raises                |
|----------|--------------------------------|
| error    | `ValueError`, `AssertionError` |

Message:

```text
`{function}`: `{parameter}` ({value}) must not be less than `{other}` ({bound}).
```

The call takes a range whose end comes first: the end argument is less than
the start argument. Swap the two, or check how they are calculated.

<a id="R0041"></a>

## R0041: Matrix with na elements

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}` cannot take a matrix that holds `na`.
```

The calculation is not defined for a matrix with `na` elements. Replace the
`na` elements first, for example with `matrix.fill`, or check the values the
matrix was built from.

<a id="R0042"></a>

## R0042: Invalid run option

| Severity | PyneCore raises                                       |
|----------|-------------------------------------------------------|
| error    | `ValueError`, `NotImplementedError`, `AssertionError` |

Message:

```text
`{option}`: {why}.
```

An option of the run, or of the script's `strategy()` declaration, is not
usable as given.

<a id="R0043"></a>

## R0043: Field of an na object

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
`{name}` is a field of an `na` object.
```

A field was read or written on a user-defined object or a `chart.point` that
is `na`: directly, through a method, nested (`o.inner.v`) or inside `na()`.
Create the object before its fields are used, or check the object itself
with `na()`.

<a id="R0044"></a>

## R0044: Copy of an na object

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
An `na` object cannot be copied.
```

`copy()` was called on a user-defined object or a `chart.point` that is
`na`. A copy reads every field, so the object must exist: create it first,
or check it with `na()` before copying.

<a id="R0045"></a>

## R0045: Symbol detail not given

| Severity | PyneCore raises  |
|----------|------------------|
| error    | `AttributeError` |

Message:

```text
`syminfo.{name}` is not given for this symbol.
```

Some symbol details exist only when the data provider supplies them, such as
`syminfo.avg_spread`. The symbol of this run has none, so reading it stops the
run. Add the value to the symbol's description, or do not read it.

<a id="R0046"></a>

## R0046: Unreadable data file

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{path}` is not a usable OHLCV file: {why}.
```

The file is not an OHLCV file, it is damaged, or it was written in a format
version this PyneCore cannot read. Download or convert the data again.

<a id="R0047"></a>

## R0047: Script cannot run

| Severity | PyneCore raises                 |
|----------|---------------------------------|
| error    | `ImportError`, `AssertionError` |

Message:

```text
`{path}` cannot run: {why}.
```

A Pyne code file starts with a docstring holding `@pyne`, has a `main`
function decorated with `@script.indicator`, `@script.strategy` or
`@script.library`, and its `main` returns a dict when it returns anything.

<a id="R0048"></a>

## R0048: File not found

| Severity | PyneCore raises     |
|----------|---------------------|
| error    | `FileNotFoundError` |

Message:

```text
`{path}` does not exist.
```

A file the run needs, such as the script, its data or the data's symbol
description, is not where the run looks for it.

<a id="R0049"></a>

## R0049: Data provider unavailable

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
`{provider}`: {why}.
```

The data provider or broker of a live run cannot serve it: the connection did
not come up in time, the plugin cannot deliver live data, or the exchange
does not meet what the script requires.

<a id="R0050"></a>

## R0050: Invalid call arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `TypeError`     |

Message:

```text
`{function}`: {why}.
```

A built-in, a constructor or a type's `copy()` was called with arguments it
does not take: more positional arguments than it has parameters, an argument
given both by position and by name, or a name that is not one of its
parameters. Check the call against the built-in's parameter list.

<a id="R0051"></a>

## R0051: Invalid series subscript

| Severity | PyneCore raises                         |
|----------|-----------------------------------------|
| error    | `TypeError`, `IndexError`, `ValueError` |

Message:

```text
`{subscript}` is not a valid series subscript: {why}.
```

A series is read with `x[offset]` or `x[start:stop]`, where the offsets are
whole numbers of bars back from the current bar. The subscript was not a
number, was negative, had a step, or asked for more bars than the series
keeps. Use a non-negative whole number of bars back.

<a id="R0052"></a>

## R0052: Argument out of range

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{function}`: `{parameter}` must be between {low} and {high}, but it is {value}.
```

An argument is outside the range the built-in accepts, such as a
`max_bars_back` above 5000 or a color's alpha outside 0..255. Pass a value
within the range.

<a id="R0053"></a>

## R0053: Export used wrongly

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `ValueError`    |

Message:

```text
`{name}`: {why}.
```

A library export is declared or used in a way the export machinery does not
support: an `Exported()` marker without the function it exports, or an
`@export` function reached without the proxy the transform builds for it.
Declare exports in a `@pyne` library module, as its top-level functions.

<a id="R0054"></a>

## R0054: request.security called outside the transform

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `RuntimeError`  |

Message:

```text
`{function}` runs only where the transform rewrote it; call it directly in a `@pyne` module.
```

`request.security` and its relatives are rewritten by the transform at their
call sites in a `@pyne` module, so the security context can be run as its own
process, one per call site. A call that reaches the function itself, for
example through a variable that holds it or from a module that is not `@pyne`,
has no such rewrite; neither has a function holding the call that is called
through a value (`h = f; h()`), so its one context is written twice on a bar.
Write the calls directly, by name, in the `@pyne` module.

<a id="I0004"></a>

## I0004: Internal error

| Severity |
|----------|
| error    |

Message:

```text
Internal error: {what}.
```

An error of the runtime (PyneCore or PyneAOT), not of the script. Please
report it with the script, the data and the traceback attached.
