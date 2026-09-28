<!--
---
weight: 203
title: "Converting from Pine Script"
description: "Learn how to convert your TradingView Pine Script code to Pyne code"
icon: "swap_horiz"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Getting Started", "Migration"]
tags: ["conversion", "pine-script", "migration", "examples", "syntax", "translation"]
---
-->

# Converting from Pine Script

If you have existing Pine Script that you want to use with PyneCore, you have two options:

1. **Manual conversion**: Convert your Pine Script code to Pyne code by hand
2. **Automatic conversion**: PyneComp, a separate PyneSys service (API key required), converts Pine Script v4, v5 and v6 (v1-v3 on a best-effort basis) to Pyne code. Use `pyne compile script.pine`, or just `pyne run script.pine data`, which compiles it first.

This guide explains both approaches and highlights the key differences you need to be aware of.

## Basic Structure Conversion

Let's start with the basic structure of a script:

**Pine Script**:
```javascript
//@version=6
indicator("My Indicator", overlay=true)

// Calculations
ma = ta.sma(close, 20)

// Plotting
plot(ma, "SMA", color=color.blue)
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore.lib import script, ta, close, plot, color

@script.indicator("My Indicator", overlay=True)
def main():
    # Calculations
    ma = ta.sma(close, 20)

    # Plotting
    plot(ma, "SMA", color=color.blue)
```

Key differences:
1. Pyne code starts with a module docstring beginning with `@pyne` instead of `//@version=6`; a description can follow on the next lines of the docstring (`edge` and `lib` directly after `@pyne` are reserved mode words)
2. You need to import the required components from the `pynecore.lib` module
3. The script declaration is a decorator on the `main()` function
4. The main code goes inside the `main()` function

## Variable Declaration Differences

PyneCore uses Python's type hints for variable declarations:

**Pine Script**:
```javascript
//@version=6
indicator("Variable Example")

// Variable declarations
var float myFloat = 0.0
var int myInt = 0
var bool myBool = true
var string myString = "text"
float[] myArray = array.new_float(0)
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, array

@script.indicator("Variable Example")
def main():
    # Variable declarations
    myFloat: Persistent[float] = 0.0  # Note: you don't need to create a Persistent object, just annotate the type
    myInt: Persistent[int] = 0
    myBool: Persistent[bool] = True
    myString: Persistent[str] = "text"
    myArray = array.new_float(0)
```

Key differences:
1. `var` in Pine Script corresponds to `Persistent` in PyneCore
2. `varip` corresponds to `IBPersistent[T]` (`from pynecore.types import IBPersistent`)
3. PyneCore uses Python's type hints system (`variable: Type = value`)
4. Arrays are handled similarly, but with Python's syntax

## Series Variables

In Pine Script, all variables are series by default. In PyneCore, a variable keeps its history only if you annotate it with `Series[T]`. You need this only when you index its past values (`x[1]`). Built-in series (`close[1]`) and `ta.*` functions keep their own history:

**Pine Script**:
```javascript
//@version=6
indicator("Series Example")

// All these are series
price = close
avgPrice = ta.sma(price, 20)
change = price - price[1]
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta

@script.indicator("Series Example")
def main():
    # Series annotation: needed because the past value of price is read below
    price: Series[float] = close
    # No annotation needed: ta.sma keeps its own history
    avgPrice = ta.sma(price, 20)
    change = price - price[1]
```

## Function Definitions

Function definitions use Python syntax but behave like Pine Script functions:

**Pine Script**:
```javascript
//@version=6
indicator("Function Example")

// Function definition
myMAFunc(src, len) =>
    result = ta.sma(src, len)
    resultSquared = math.pow(result, 2)
    resultSquared

// Function usage
value = myMAFunc(close, 20)
plot(value)
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, plot, math

def myMAFunc(src, len):
    result = ta.sma(src, len)
    resultSquared = math.pow(result, 2)
    return resultSquared  # Explicit return statement

@script.indicator("Function Example")
def main():
    # Function usage
    value = myMAFunc(close, 20)
    plot(value)
```

## Conditional Statements

Conditional statements use Python syntax:

**Pine Script**:
```javascript
//@version=6
indicator("Conditional Example")

// Condition
condition = close > open

// Conditional (ternary) operator
barColor = condition ? color.green : color.red

// If statement
if (condition)
    strategy.entry("Long", strategy.long)
else
    strategy.close("Long")
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore.lib import script, close, open, color, plot, strategy

@script.strategy("Conditional Example")
def main():
    # Condition
    condition = close > open

    # Conditional logic (ternary operator becomes a conditional expression)
    barColor = color.green if condition else color.red

    # If statement
    if condition:
        strategy.entry("Long", strategy.long)
    else:
        strategy.close("Long")
```

## Loops

Loops use Python syntax:

**Pine Script**:
```javascript
//@version=6
indicator("Loop Example")

// For loop
sum = 0.0
for i = 0 to 9
    sum := sum + close[i]
plot(sum / 10)
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, plot

@script.indicator("Loop Example")
def main():
    # For loop
    sum = 0.0
    for i in range(10):
        sum += close[i]
    plot(sum / 10)
```

## NA Value Handling

As in Pine Script, operations with `na` values propagate `na` in PyneCore. Use the `na()` function to check for NA values and `nz()` to replace them:

**Pine Script**:
```javascript
//@version=6
indicator("NA Example")

// NA handling
closeValue = na(close) ? open : close
value = nz(close, 0)
```

**PyneCore**:
```python
"""
@pyne
"""
from pynecore.lib import script, close, open, plot, na, nz

@script.indicator("NA Example")
def main():
    # NA handling
    if na(close):
        close_value = open
    else:
        close_value = close

    value = nz(close, 0)
    plot(value)
```

## Automatic Conversion with PyneComp

For complex scripts, you can use the PyneComp compiler, a separate, closed source PyneSys service that needs an API key. It automatically converts Pine Script v4, v5 and v6 code to Pyne code; v1-v3 sources are converted on a best-effort basis.

```bash
# Compile a Pine Script file to Pyne code (writes script.py next to it)
pyne compile script.pine

# Or run the .pine file directly: it is compiled first
pyne run script.pine data
```

The API key goes in `workdir/config/api.toml`, or can be given with `--api-key` or the `PYNESYS_API_KEY` environment variable. PyneComp output is marked with `"""@pyne edge"""`, a strict, Pine-equivalent subset of Pyne code that PyneCore runs like any other Pyne code. See [Compiling Pine Scripts](../cli/compile.md) for details.

Benefits of using PyneComp:
- Saves time on manual conversion
- Handles complex syntax and edge cases
- Generates clean, readable Python code
- Offers a "strict mode" for better variable scoping
- Compiled scripts can be run directly with `python script.py data.csv` — no CLI or workdir needed

## Example: Complete Strategy Conversion

Here's a more complex strategy conversion example:

**Pine Script (RSI Strategy)**:
```javascript
//@version=6
strategy("RSI Strategy", overlay=true)

// Input parameters
length = input.int(14, "RSI Length")
overbought = input.int(70, "Overbought")
oversold = input.int(30, "Oversold")

// Calculate RSI
rsiValue = ta.rsi(close, length)

// Entry/exit conditions
enterLong = ta.crossover(rsiValue, oversold)
exitLong = ta.crossover(rsiValue, overbought)

// Strategy execution
if (enterLong)
    strategy.entry("Long", strategy.long)
else if (exitLong)
    strategy.close("Long")

// Display RSI
hline(overbought, "Overbought", color.red)
hline(oversold, "Oversold", color.green)
```

**PyneCore (RSI Strategy)**:
```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, strategy, input, hline, color

@script.strategy("RSI Strategy", overlay=True)
def main(
        # Input parameters are declared as main() parameters
        length: int = input.int(14, "RSI Length"),
        overbought: int = input.int(70, "Overbought"),
        oversold: int = input.int(30, "Oversold"),
):
    # Calculate RSI
    rsiValue: Series[float] = ta.rsi(close, length)

    # Entry/exit conditions
    enterLong = ta.crossover(rsiValue, oversold)
    exitLong = ta.crossover(rsiValue, overbought)

    # Strategy execution
    if enterLong:
        strategy.entry("Long", strategy.long)
    elif exitLong:
        strategy.close("Long")

    # Display RSI
    hline(overbought, "Overbought", color=color.red)
    hline(oversold, "Oversold", color=color.green)
```

## Common Conversion Pitfalls

When converting from Pine Script to PyneCore, be aware of these common issues:

### 1. Variable Scope and Lifecycle Differences

Both Pine Script and Python use lexical scoping, but with important differences:

**Scope differences:**
- **Pine Script**: Block-level scoping - every code block (if, for, functions) has its own scope
- **Python**: Function-level scoping - only functions create new scopes, blocks (if, for) do not

**Variable lifecycle differences:**
- **Pine Script**: By default, variables reinitialize on each bar; use the `var` keyword to make them persist between bars
- **PyneCore**: By default, variables follow normal Python behavior; use `Persistent[T]` type annotation to indicate persistence between bars

**Series behavior:**
- **Pine Script**: Every variable is a Series by default
- **PyneCore**: A variable keeps its history only when marked with the `Series[T]` type annotation, which is needed only if you index its past values

**Pine Script example:**
```javascript
//@version=6
indicator("Scope Example")

// Reinitializes on each bar
counter = 0

// Persists between bars
var persistentCounter = 0

myFunction() =>
    // Only accessible within the function
    localCounter = 0
    
    if (true)
        // Only accessible within the if block
        blockCounter = 42
        localCounter := blockCounter
    
    // blockCounter is NOT accessible outside the if block!
    // blockCounter := 10  // This would cause an error
    
    localCounter  // Return value
```

**Equivalent PyneCore code:**
```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script

@script.indicator("Scope Example")
def main():
    # Reinitializes on each bar, explicitly marked as Series
    counter: Series[int] = 0
    
    # Persists between bars
    persistentCounter: Persistent[int] = 0
    
    def my_function():
        # Only accessible within the function
        local_counter = 0
        
        if True:
            # DIFFERENCE: block_counter is accessible OUTSIDE the if block in Python!
            block_counter = 42
            local_counter = block_counter
        
        # In Python, block_counter would be usable here
        # block_counter = 10  # This would work in Python (but not in Pine)
        
        return local_counter
```

The most important difference is that compared to Pine Script's block-level scoping, Python is less strict, with only functions creating new scopes. Therefore, block-level variables that would be safely encapsulated in Pine Script might have broader scope in Python/PyneCore implementations.

### 2. Type Handling

PyneCore requires an explicit `Persistent[T]` annotation for variables that persist between bars, and a `Series[T]` annotation for variables whose past values you index.

### 3. Return Values

In Pine Script, the last expression is automatically the return value. In PyneCore (Python), you need an explicit `return` statement.

### 4. Ternary Operators

Pine Script's ternary operator (`condition ? value1 : value2`) becomes Python's conditional expression `value1 if condition else value2`.

### 5. Reassignment Operator

Pine Script's `:=` reassignment operator becomes `=` in Python.

## Best Practices for Conversion

1. **Start small**: Begin with simple scripts and work your way up
2. **Test thoroughly**: Compare results with the original Pine Script
3. **Use type annotations**: Be explicit about Series and Persistent variables
4. **Leverage Python features**: Take advantage of Python's richer features during conversion
5. **Keep organized**: Break complex scripts into functions or modules

