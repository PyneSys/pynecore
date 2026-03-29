<!--
---
weight: 426
title: "matrix"
description: "Two-dimensional matrix operations"
icon: "grid_on"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["matrix", "library", "reference"]
---
-->

# matrix

The `matrix` namespace provides operations for creating and manipulating two-dimensional matrices. Matrices store elements of the same type in rows and columns, supporting operations ranging from basic element access to advanced linear algebra transformations like eigenvalue decomposition and matrix inversion.

## Quick Example

```python
from pynecore.lib import matrix, script

@script.indicator(title="Matrix Analysis", overlay=False)
def main():
    # Create a 3x3 matrix
    m = matrix.new(3, 3, 0.0)
    
    # Set diagonal values
    matrix.set(m, 0, 0, 10.0)
    matrix.set(m, 1, 1, 20.0)
    matrix.set(m, 2, 2, 30.0)
    
    # Analyze matrix properties
    is_diag: bool = matrix.is_diagonal(m)
    trace_val: float = matrix.trace(m)  # Sum of diagonal
    avg_val: float = matrix.avg(m)
```

## Matrix Creation and Management

### new()

Create a new matrix object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `int` | Initial number of rows (default: 0) |
| `columns` | `int` | Initial number of columns (default: 0) |
| `initial_value` | `float \| int \| str \| bool` | Initial value for all elements (default: NA) |

**Returns:** New matrix object

```python
m = matrix.new(3, 4, 0.0)  # 3×4 matrix filled with 0.0
```

### copy()

Create an independent copy of an existing matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix to copy |

**Returns:** New matrix object

```python
m2 = matrix.copy(m1)  # m2 is independent from m1
```

## Dimensions and Properties

### rows()

Get the number of rows in a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `int`

```python
row_count: int = matrix.rows(m)  # 3
```

### columns()

Get the number of columns in a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `int`

```python
col_count: int = matrix.columns(m)  # 4
```

### elements_count()

Get the total number of elements in a matrix (rows × columns).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `int`

```python
total: int = matrix.elements_count(m)  # 12
```

## Element Access and Modification

### get()

Retrieve an element at a specific row and column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row` | `int` | Row index (zero-based) |
| `column` | `int` | Column index (zero-based) |

**Returns:** Element value

```python
val: float = matrix.get(m, 0, 1)  # 15.5
```

### set()

Set an element at a specific row and column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row` | `int` | Row index |
| `column` | `int` | Column index |
| `value` | element type | Value to assign |

**Returns:** `None`

```python
matrix.set(m, 2, 3, 42.0)
```

### fill()

Fill a rectangular area of a matrix with a single value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `value` | element type | Value to fill with |
| `from_row` | `int` | Starting row, inclusive (default: 0) |
| `to_row` | `int` | Ending row, exclusive (default: last row) |
| `from_column` | `int` | Starting column, inclusive (default: 0) |
| `to_column` | `int` | Ending column, exclusive (default: last column) |

**Returns:** `None`

```python
matrix.fill(m, 0.0, 1, 3, 0, 2)  # Fill rows 1–2, columns 0–1
```

## Row and Column Operations

### row()

Extract a row as a one-dimensional array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row` | `int` | Row index |

**Returns:** `list` — Array of row values

```python
row_vals: list = matrix.row(m, 1)
```

### col()

Extract a column as a one-dimensional array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `column` | `int` | Column index |

**Returns:** `list` — Array of column values

```python
col_vals: list = matrix.col(m, 2)
```

### add_row()

Insert a new row at a specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row` | `int \| None` | Row index (None appends to end) |
| `array_id` | `list \| None` | Array of values for the new row |

**Returns:** `None`

```python
matrix.add_row(m, 1, [1.0, 2.0, 3.0, 4.0])
```

### add_col()

Insert a new column at a specified index.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `column` | `int \| None` | Column index (None appends to end) |
| `array_id` | `list \| None` | Array of values for the new column |

**Returns:** `None`

```python
matrix.add_col(m, 0, [5.0, 6.0, 7.0])
```

### remove_row()

Remove a row and return its values as an array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row` | `int` | Row index to remove |

**Returns:** `list` — Removed row values

```python
removed: list = matrix.remove_row(m, 1)
```

### remove_col()

Remove a column and return its values as an array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `column` | `int` | Column index to remove |

**Returns:** `list` — Removed column values

```python
removed: list = matrix.remove_col(m, 2)
```

### swap_rows()

Exchange two rows in a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `row1` | `int` | First row index |
| `row2` | `int` | Second row index |

**Returns:** `None`

```python
matrix.swap_rows(m, 0, 2)
```

### swap_columns()

Exchange two columns in a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `column1` | `int` | First column index |
| `column2` | `int` | Second column index |

**Returns:** `None`

```python
matrix.swap_columns(m, 1, 3)
```

## Matrix Property Tests

### is_square()

Check if a matrix has equal rows and columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_sq: bool = matrix.is_square(m)  # True
```

### is_diagonal()

Check if all elements outside the main diagonal are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_diag: bool = matrix.is_diagonal(m)  # True
```

### is_antidiagonal()

Check if all elements outside the secondary diagonal are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_antidiag: bool = matrix.is_antidiagonal(m)  # False
```

### is_identity()

Check if the matrix is an identity matrix (ones on main diagonal, zeros elsewhere).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_id: bool = matrix.is_identity(m)  # False
```

### is_symmetric()

Check if a matrix equals its own transpose.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_sym: bool = matrix.is_symmetric(m)  # True
```

### is_antisymmetric()

Check if a matrix equals the negative of its transpose.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_antisym: bool = matrix.is_antisymmetric(m)  # False
```

### is_binary()

Check if all elements are either 0 or 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_bin: bool = matrix.is_binary(m)  # False
```

### is_triangular()

Check if all elements above or below the main diagonal are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_tri: bool = matrix.is_triangular(m)  # True
```

### is_stochastic()

Check if the matrix is stochastic (rows or columns sum to 1).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_stoch: bool = matrix.is_stochastic(m)  # False
```

### is_zero()

Check if all elements are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `bool`

```python
is_z: bool = matrix.is_zero(m)  # False
```

## Arithmetic Operations

### sum()

Add two matrices or a matrix and a scalar.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id1` | matrix | First matrix |
| `id2` | matrix \| `float` \| `int` | Second matrix or scalar |

**Returns:** New matrix

```python
m3 = matrix.sum(m1, m2)    # Element-wise addition
m4 = matrix.sum(m1, 5.0)   # Add 5.0 to all elements
```

### diff()

Subtract two matrices or a scalar from a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id1` | matrix | Matrix to subtract from |
| `id2` | matrix \| `float` \| `int` | Matrix or scalar to subtract |

**Returns:** New matrix

```python
m3 = matrix.diff(m1, m2)   # Element-wise subtraction
m4 = matrix.diff(m1, 3.0)  # Subtract 3.0 from all elements
```

### mult()

Multiply two matrices or a matrix by a scalar.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id1` | matrix | First matrix |
| `id2` | matrix \| `float` \| `int` | Second matrix or scalar |

**Returns:** New matrix

```python
m3 = matrix.mult(m1, m2)   # Matrix multiplication
m4 = matrix.mult(m1, 2.5)  # Scalar multiplication
```

### pow()

Raise a square matrix to a power (multiply by itself n times).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |
| `power` | `int` | Exponent |

**Returns:** New matrix

```python
m_squared = matrix.pow(m, 2)  # m × m
```

## Statistical Functions

### avg()

Calculate the average of all matrix elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `float \| int`

```python
average: float = matrix.avg(m)  # 42.5
```

### min()

Find the smallest element in the matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `float \| int`

```python
min_val: float = matrix.min(m)  # 0.0
```

### max()

Find the largest element in the matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `float \| int`

```python
max_val: float = matrix.max(m)  # 100.0
```

### median()

Calculate the median of all matrix elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `float \| int`

```python
med: float = matrix.median(m)  # 50.0
```

### mode()

Find the most frequently occurring element in the matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** Most frequent value (or smallest if no unique mode)

```python
most_common: float = matrix.mode(m)  # 25.0
```

## Linear Algebra

### transpose()

Create a new matrix with rows and columns swapped.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** New matrix

```python
m_t = matrix.transpose(m)
```

### det()

Calculate the determinant of a square matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |

**Returns:** `float \| int`

```python
determinant: float = matrix.det(m)  # -2.0
```

### inv()

Calculate the inverse of a square matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |

**Returns:** New matrix

```python
m_inv = matrix.inv(m)
```

### pinv()

Calculate the pseudo-inverse (Moore-Penrose inverse) of a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** New matrix

```python
m_pinv = matrix.pinv(m)
```

### rank()

Calculate the rank of a matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `int`

```python
r: int = matrix.rank(m)  # 2
```

### trace()

Calculate the trace (sum of main diagonal elements).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |

**Returns:** `float \| int`

```python
tr: float = matrix.trace(m)  # 30.0
```

### eigenvalues()

Calculate the eigenvalues of a square matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |

**Returns:** `list` — Array of eigenvalues

```python
evals: list = matrix.eigenvalues(m)
```

### eigenvectors()

Calculate the eigenvectors of a square matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Square matrix |

**Returns:** New matrix — Each column is an eigenvector

```python
evecs = matrix.eigenvectors(m)
```

## Matrix Transformations

### submatrix()

Extract a rectangular submatrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `from_row` | `int` | Starting row (inclusive) |
| `to_row` | `int` | Ending row (exclusive) |
| `from_column` | `int` | Starting column (inclusive) |
| `to_column` | `int` | Ending column (exclusive) |

**Returns:** New matrix

```python
sub = matrix.submatrix(m, 1, 3, 0, 2)  # Rows 1–2, columns 0–1
```

### reshape()

Reorganize the matrix to new dimensions (preserves all elements).

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `rows` | `int` | New number of rows |
| `columns` | `int` | New number of columns |

**Returns:** `None` — Modifies in place

```python
matrix.reshape(m, 4, 3)  # Reorganize to 4×3
```

### reverse()

Reverse the order of rows and columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |

**Returns:** `None` — Modifies in place

```python
matrix.reverse(m)
```

### concat()

Append one matrix to another.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id1` | matrix | Matrix to concatenate into |
| `id2` | matrix | Matrix to append |

**Returns:** Modified first matrix

```python
m3 = matrix.concat(m1, m2)
```

### sort()

Sort rows based on values in a specific column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | matrix | Matrix object |
| `column` | `int` | Column index to sort by |
| `order` | `str` | `"ascending"` or `"descending"` |

**Returns:** `None` — Modifies in place

```python
matrix.sort(m, 2, "ascending")
```

### kron()

Calculate the Kronecker product of two matrices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id1` | matrix | First matrix |
| `id2` | matrix | Second matrix |

**Returns:** New matrix

```python
m_kron = matrix.kron(m1, m2)
```

## Compatibility Notes

- All matrix elements must be the same type
- Matrix indices are zero-based
- NA (not available) values are handled automatically—operations on NA return NA
- Linear algebra functions (inv, eigenvalues, etc.) require square or compatible matrices
- For very large matrices, performance depends on matrix size and operation complexity