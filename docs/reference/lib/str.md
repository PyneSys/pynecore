<!--
---
weight: 423
title: "str"
description: "String manipulation functions"
icon: "text_fields"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["str", "library", "reference"]
---
-->

# str

The `str` namespace provides string manipulation functions for working with text values in PyneCore scripts. These functions allow you to search, format, transform, and analyze strings in trading indicators and strategies.

## Quick Example

```python
from pynecore.lib import str as pstr, label, bar_index, high, close, script

@script.indicator(title="String Operations Demo", overlay=True)
def main():
    # Get price as formatted string
    price_text: str = pstr.tostring(close)
    
    # Format with custom precision
    formatted_price: str = pstr.format("{0}", close)
    
    # String transformations
    upper_text: str = pstr.upper(price_text)
    lower_text: str = pstr.lower(upper_text)
    
    # String analysis
    length_val: int = pstr.length(price_text)
    contains_val: bool = pstr.contains(price_text, ".")
    
    # Display result
    if bar_index % 10 == 0:
        label.new(bar_index, high, upper_text)
```

## Functions

### contains()

Returns true if the source string contains the substring, false otherwise.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to search in |
| str_ | str | The substring to search for |

**Returns:** `bool`

**Example:**
```python
found: bool = str.contains("hello world", "world")  # True
```

### endswith()

Returns true if the source string ends with the specified substring, false otherwise.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to check |
| str_ | str | The suffix to search for |

**Returns:** `bool`

**Example:**
```python
is_suffix: bool = str.endswith("example.txt", ".txt")  # True
```

### format()

Creates a formatted string using a format string and arguments, supporting positional placeholders.

| Parameter | Type | Description |
|-----------|------|-------------|
| formatString | str | Format string with placeholders like `{0}`, `{1}` |
| *args | any | Positional arguments to insert into placeholders |

**Returns:** `str`

**Example:**
```python
result: str = str.format("Price is {0}, Change is {1}%", 123.45, 2.5)  # "Price is 123.45, Change is 2.5%"
```

### format_time()

Converts a timestamp into a string formatted according to the specified format string and timezone.

| Parameter | Type | Description |
|-----------|------|-------------|
| time | int | Unix timestamp in milliseconds |
| fmt | str \| None | Format string (e.g., "yyyy-MM-dd HH:mm:ss"), default None |
| tz | str \| None | Timezone identifier (e.g., "America/New_York"), default None |

**Returns:** `str`

**Example:**
```python
formatted: str = str.format_time(1640995200000, "yyyy-MM-dd", "UTC")  # "2022-01-01"
```

### length()

Returns the number of characters in the string.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to measure |

**Returns:** `int`

**Example:**
```python
len_val: int = str.length("hello")  # 5
```

### lower()

Returns a new string with all letters converted to lowercase.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to convert |

**Returns:** `str`

**Example:**
```python
lower_str: str = str.lower("HELLO World")  # "hello world"
```

### match()

Returns the substring that matches the specified regex pattern, or an empty string if no match is found.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to search in |
| regex | str | Regular expression pattern to match |

**Returns:** `str`

**Example:**
```python
matched: str = str.match("Price: 123.45", "[0-9]+")  # "123"
```

### pos()

Returns the position of the first occurrence of the substring, or `na` if not found.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to search in |
| str_ | str | The substring to find |

**Returns:** `int` (or `na` if not found)

**Example:**
```python
position: int = str.pos("hello world", "world")  # 6
```

### repeat()

Constructs a new string by repeating the source string multiple times with a separator between each repetition.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to repeat |
| repeat | int | Number of times to repeat |
| separator | str | String to insert between repetitions, default `""` |

**Returns:** `str`

**Example:**
```python
repeated: str = str.repeat("ab", 3, "-")  # "ab-ab-ab"
```

### replace()

Returns a new string with the Nth occurrence of the target substring replaced by the replacement string.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to modify |
| target | str | The substring to replace |
| replacement | str | The replacement string |
| occurence | int | Which occurrence to replace (0-based), default 0 |

**Returns:** `str`

**Example:**
```python
replaced: str = str.replace("hello hello", "hello", "hi", 1)  # "hi hello"
```

### replace_all()

Replaces every occurrence of the target substring with the replacement string.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to modify |
| target | str | The substring to replace |
| replacement | str | The replacement string |

**Returns:** `str`

**Example:**
```python
replaced: str = str.replace_all("hello hello", "hello", "hi")  # "hi hi"
```

### split()

Divides a string into an array of substrings using the specified separator.

| Parameter | Type | Description |
|-----------|------|-------------|
| string | str | The string to split |
| separator | str | The delimiter to split on |

**Returns:** `array<string>` (array ID)

**Example:**
```python
parts: list[str] = str.split("a,b,c", ",")  # ["a", "b", "c"]
```

### startswith()

Returns true if the source string starts with the specified substring, false otherwise.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to check |
| str_ | str | The prefix to search for |

**Returns:** `bool`

**Example:**
```python
is_prefix: bool = str.startswith("example.txt", "example")  # True
```

### substring()

Returns a new string extracted from the source, starting at the specified position and extending for the specified length.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to extract from |
| begin_pos | int | Starting position (0-based) |
| end_pos | int \| None | Ending position (0-based, exclusive), default None (end of string) |

**Returns:** `str`

**Example:**
```python
sub: str = str.substring("hello", 1, 4)  # "ell"
```

### tonumber()

Converts a string representation of a number to its float equivalent.

| Parameter | Type | Description |
|-----------|------|-------------|
| string | str | The string to convert |

**Returns:** `float`

**Example:**
```python
num: float = str.tonumber("123.45")  # 123.45
```

### tostring()

Converts a value to its string representation.

| Parameter | Type | Description |
|-----------|------|-------------|
| value | any | The value to convert (float, int, bool, or string) |
| fmt | str \| Format | Number format string, default `"#.##########"` |

**Returns:** `str`

**Example:**
```python
str_val: str = str.tostring(123.45)              # "123.45"
str_fmt: str = str.tostring(123.45, "#.##")      # "123.45"
bool_str: str = str.tostring(True)               # "true"
```

### trim()

Constructs a new string with all consecutive whitespace and control characters (such as `\n`, `\t`) removed from both ends.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to trim |

**Returns:** `str`

**Example:**
```python
trimmed: str = str.trim("  hello world  ")  # "hello world"
```

### upper()

Returns a new string with all letters converted to uppercase.

| Parameter | Type | Description |
|-----------|------|-------------|
| source | str | The string to convert |

**Returns:** `str`

**Example:**
```python
upper_str: str = str.upper("hello World")  # "HELLO WORLD"
```

## Compatibility

All 18 functions in the `str` namespace are fully implemented and compatible with TradingView's Pine Script v6 string functions. PyneCore handles string formatting with Pine-compatible rules, including support for timezone-aware timestamp formatting via `format_time()`.