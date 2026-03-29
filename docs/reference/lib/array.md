<!--
---
weight: 425
title: "array"
description: "Dynamic array operations"
icon: "view_list"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["array", "library", "reference"]
---
-->

# array

The `array` namespace provides dynamic array manipulation and statistical operations. Arrays are Python lists that can store values of a specific type. Use array functions to create, modify, search, and analyze collections of data.

## Quick Example

```python
from pynecore.lib import (
    close, array, script
)

@script.indicator(title="Array Statistics", overlay=False)
def main():
    # Create and populate an array of closes
    closes: list[float] = array.new_float()
    array.push(closes, close)
    
    if array.size(closes) > 20:
        avg: float = array.avg(closes)
        max_val: float = array.max(closes)
        min_val: float = array.min(closes)
```

## Functions

### Creation

#### `new_bool(size, initial_value)`
Creates a new boolean array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | bool | Initial value for all elements, default na |
| **Returns** | list[bool] | New boolean array |

#### `new_int(size, initial_value)`
Creates a new integer array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | int | Initial value for all elements, default na |
| **Returns** | list[int] | New integer array |

#### `new_float(size, initial_value)`
Creates a new float array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | float | Initial value for all elements, default na |
| **Returns** | list[float] | New float array |

#### `new_string(size, initial_value)`
Creates a new string array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | str | Initial value for all elements, default na |
| **Returns** | list[str] | New string array |

#### `new_color(size, initial_value)`
Creates a new color array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | color | Initial value for all elements, default na |
| **Returns** | list[color] | New color array |

#### `new_label(size, initial_value)`
Creates a new label array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | label | Initial value for all elements, default na |
| **Returns** | list[label] | New label array |

#### `new_line(size, initial_value)`
Creates a new line array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | line | Initial value for all elements, default na |
| **Returns** | list[line] | New line array |

#### `new_box(size, initial_value)`
Creates a new box array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | box | Initial value for all elements, default na |
| **Returns** | list[box] | New box array |

#### `new_linefill(size, initial_value)`
Creates a new linefill array.

| Parameter | Type | Description |
|-----------|------|-------------|
| size | int | Initial size, default 0 |
| initial_value | linefill | Initial value for all elements, default na |
| **Returns** | list[linefill] | New linefill array |

#### `from_items(*items)`
Creates an array from the specified elements. (In Pine Script this is `array.from()`, but `from` is a reserved keyword in Python.)

| Parameter | Type | Description |
|-----------|------|-------------|
| *items | any | Elements to include |
| **Returns** | list | Array containing the elements |

Example: `arr: list[int] = array.from_items(1, 2, 3)  # [1, 2, 3]`

### Access

#### `get(id, index)`
Returns the element at the specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| index | int | Index of element |
| **Returns** | any | Element at the index |

Example: `val: float = array.get(my_array, 0)  # first element`

#### `first(id)`
Returns the first element. Throws an error if the array is empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | any | First element |

#### `last(id)`
Returns the last element. Throws an error if the array is empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | any | Last element |

#### `size(id)`
Returns the number of elements in the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | int | Array size |

Example: `len: int = array.size(my_array)  # 10`

### Modification

#### `set(id, index, value)`
Sets the element at the specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| index | int | Index to set |
| value | any | New value |

#### `push(id, value)`
Appends an element to the end of the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Element to append |

#### `pop(id)`
Removes and returns the last element.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | any | Removed element |

Example: `last: float = array.pop(my_array)  # removed from array`

#### `shift(id)`
Removes and returns the first element.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | any | Removed element |

Example: `first: float = array.shift(my_array)  # removed from array`

#### `unshift(id, value)`
Inserts an element at the beginning of the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Element to insert |

#### `insert(id, index, value)`
Inserts an element at the specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| index | int | Index to insert at |
| value | any | Element to insert |

#### `remove(id, index)`
Removes and returns the element at the specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| index | int | Index to remove |
| **Returns** | any | Removed element |

Example: `removed: float = array.remove(my_array, 2)  # element at index 2`

#### `clear(id)`
Removes all elements from the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |

#### `fill(id, value, index_from, index_to)`
Fills a range of elements with a specified value.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Value to fill with |
| index_from | int | Start index (default 0) |
| index_to | int | End index (optional) |

Example: `array.fill(my_array, 0.0, 5, 10)  # fill indices 5-9 with 0.0`

### Search

#### `includes(id, value)`
Returns true if the value exists in the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Value to search for |
| **Returns** | bool | True if found, false otherwise |

Example: `found: bool = array.includes(my_array, 42)  # True or False`

#### `indexof(id, value)`
Returns the index of the first occurrence, or -1 if not found.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Value to search for |
| **Returns** | int | Index or -1 |

#### `lastindexof(id, value)`
Returns the index of the last occurrence, or -1 if not found.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| value | any | Value to search for |
| **Returns** | int | Index or -1 |

#### `binary_search(id, val)`
Binary search in a sorted array. Returns the index or -1 if not found.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Sorted array (ascending) |
| val | any | Value to search for |
| **Returns** | int | Index or -1 |

#### `binary_search_leftmost(id, val)`
Binary search in a sorted array. Returns the index of the value or the leftmost element greater than it.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Sorted array (ascending) |
| val | any | Value to search for |
| **Returns** | int | Index |

#### `binary_search_rightmost(id, val)`
Binary search in a sorted array. Returns the index of the value or the rightmost element less than it.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Sorted array (ascending) |
| val | any | Value to search for |
| **Returns** | int | Index |

### Transformation

#### `copy(id)`
Creates a shallow copy of the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| **Returns** | array | Copy of the array |

Example: `arr_copy: list[float] = array.copy(my_array)  # independent copy`

#### `concat(id1, id2)`
Merges the second array into the first and returns the first.

| Parameter | Type | Description |
|-----------|------|-------------|
| id1 | array | First array (modified) |
| id2 | array | Second array |
| **Returns** | array | First array with merged elements |

#### `slice(id, index_from, index_to)`
Creates a shallow copy of a slice of the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| index_from | int | Start index |
| index_to | int | End index |
| **Returns** | array | Sliced array |

#### `reverse(id)`
Reverses the order of elements in the array.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |

#### `sort(id, order)`
Sorts the array in ascending or descending order.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| order | order | Ascending or descending |

#### `sort_indices(id, order)`
Returns indices that would sort the array without modifying the original.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| order | order | Ascending or descending |
| **Returns** | list[int] | Indices in sorted order |

#### `join(id, separator)`
Concatenates all elements into a single string with separator.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array |
| separator | str | Separator string |
| **Returns** | str | Joined string |

Example: `result: str = array.join(my_array, ",")  # "1,2,3"`

### Statistics

#### `sum(id)`
Returns the sum of all elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | float | Sum |

Example: `total: float = array.sum(my_array)  # 15.0`

#### `avg(id)`
Returns the average (mean) of all elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | float | Average |

Example: `average: float = array.avg(my_array)  # 5.0`

#### `min(id)`
Returns the smallest element.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | number | Minimum value |

Example: `smallest: float = array.min(my_array)  # 1.0`

#### `max(id)`
Returns the largest element.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | number | Maximum value |

Example: `largest: float = array.max(my_array)  # 10.0`

#### `median(id)`
Returns the median of all elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | float | Median value |

#### `mode(id)`
Returns the most frequently occurring value (or smallest if tied).

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | number | Most frequent value |

#### `stdev(id)`
Returns the standard deviation of all elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | float | Standard deviation |

#### `variance(id)`
Returns the variance of all elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | float | Variance |

#### `covariance(id1, id2, biased)`
Returns the covariance between two arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| id1 | array | First array of numbers |
| id2 | array | Second array of numbers |
| biased | bool | If true, use biased covariance (default) |
| **Returns** | float | Covariance |

#### `range(id)`
Returns the difference between the maximum and minimum values.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | number | Range |

Example: `r: float = array.range(my_array)  # max - min`

#### `abs(id)`
Returns an array of absolute values of each element.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | array | Array of absolute values |

Example: `positives: list[float] = array.abs(my_array)  # all positive`

#### `standardize(id)`
Returns an array of standardized (z-score) values.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| **Returns** | list[float] | Standardized array |

#### `percentile_linear_interpolation(id, percentile)`
Returns the value at the specified percentile using linear interpolation.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| percentile | float | Percentile (0-100) |
| **Returns** | float | Value at percentile |

#### `percentile_nearest_rank(id, percentile)`
Returns the value at the specified percentile using the nearest-rank method.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| percentile | float | Percentile (0-100) |
| **Returns** | float | Value at percentile |

#### `percentrank(id, index)`
Returns the percentile rank of the element at the specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of numbers |
| index | int | Element index |
| **Returns** | float | Percentile rank |

### Logic

#### `every(id)`
Returns true if all elements are truthy, false otherwise.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of booleans |
| **Returns** | bool | True if all elements are true |

#### `some(id)`
Returns true if at least one element is truthy, false otherwise.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | array | Input array of booleans |
| **Returns** | bool | True if any element is true |

## Compatibility Notes

The following Pine Script array functions are not available in PyneCore:

- `array.from` — Use `array.from_items()` instead (in PyneCore, `from` is a reserved keyword)
- `array.new<type>` — Use specific type constructors like `array.new_float()`
- `array.new_table` — Not supported

All other array functions are fully implemented.