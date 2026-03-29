<!--
---
weight: 427
title: "map"
description: "Key-value map operations"
icon: "map"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["map", "library", "reference"]
---
-->

# map

Maps are collections of key-value pairs, similar to Python dictionaries. Maps in PyneCore store any type of key and value, and provide operations for managing these pairs efficiently. They're useful for organizing related data without needing to create a custom data structure.

## Quick Example

```python
from pynecore.lib import close, strategy, script, ta, map

@script.strategy(title="Map Usage Example", overlay=False)
def main():
    # Create a new map to store moving averages
    ma_values: dict = map.new()
    
    # Add moving averages to the map
    map.put(ma_values, "sma20", ta.sma(close, 20))
    map.put(ma_values, "sma50", ta.sma(close, 50))
    
    # Retrieve a value
    sma20: float = map.get(ma_values, "sma20")
    
    # Check if a key exists
    has_sma20: bool = map.contains(ma_values, "sma20")
    
    # Iterate over keys and values
    all_keys: list = map.keys(ma_values)
    all_values: list = map.values(ma_values)
    
    # Get the number of pairs
    pair_count: int = map.size(ma_values)
    
    # Clear the map
    map.clear(ma_values)
```

## Functions

### map.new()

Creates an empty map for storing key-value pairs.

**Returns:** `dict`

**Example:**
```python
prices: dict = map.new()  # {}
```

### map.put()

Adds or updates a key-value pair in the map. If the key already exists, the previous value is returned.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to add the pair to |
| `key` | Any | The key (any hashable type) |
| `value` | Any | The value to store |

**Returns:** The previous value if the key existed, or `na` otherwise

**Example:**
```python
old_value: float | NA = map.put(data, "price", 100.5)  # Previous value if it existed
```

### map.get()

Retrieves the value associated with a key in the map.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to retrieve from |
| `key` | Any | The key to look up |

**Returns:** The value associated with the key

**Example:**
```python
price: float = map.get(data, "price")  # 100.5
```

### map.contains()

Checks whether a key exists in the map.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to check |
| `key` | Any | The key to search for |

**Returns:** `bool`

**Example:**
```python
has_price: bool = map.contains(data, "price")  # True
```

### map.remove()

Removes a key-value pair from the map and returns the removed value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to remove from |
| `key` | Any | The key to remove |

**Returns:** The value that was removed, or `na` if the key didn't exist

**Example:**
```python
removed: float | NA = map.remove(data, "price")  # Previous value
```

### map.clear()

Removes all key-value pairs from the map, leaving it empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to clear |

**Returns:** None

**Example:**
```python
map.clear(data)  # Map is now empty
```

### map.size()

Returns the number of key-value pairs currently in the map.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to measure |

**Returns:** `int`

**Example:**
```python
count: int = map.size(data)  # 3
```

### map.keys()

Returns a list of all keys in the map. The returned list is a copy; modifications to it don't affect the original map.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to extract keys from |

**Returns:** `list`

**Example:**
```python
key_list: list = map.keys(data)  # ["price", "volume", "time"]
```

### map.values()

Returns a list of all values in the map. The returned list is a copy; modifications to it don't affect the original map.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to extract values from |

**Returns:** `list`

**Example:**
```python
value_list: list = map.values(data)  # [100.5, 1000, 1234567890]
```

### map.copy()

Creates a shallow copy of the map with the same key-value pairs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to copy |

**Returns:** `dict` - A new map with identical contents

**Example:**
```python
data_copy: dict = map.copy(data)  # New dict, same contents
```

### map.put_all()

Adds all key-value pairs from another map into this map. If a key already exists in the target map, its value is overwritten.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `dict` | The map to add pairs into |
| `other` | `dict` | The map to copy pairs from |

**Returns:** None

**Example:**
```python
map.put_all(data, new_data)  # All pairs from new_data now in data
```

## Compatibility

All map functions are fully implemented in PyneCore. Unlike Pine Script's strongly-typed maps (`map.new<string, float>()`), PyneCore maps are standard Python dictionaries that can hold any type of key or value without type syntax.