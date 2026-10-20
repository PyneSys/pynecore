from __future__ import annotations

import math
from typing import TypeVar, Generic, Any, Iterator, cast
from collections import Counter
from .na import NA, na_float

T = TypeVar('T')


class Matrix(Generic[T]):
    """
    A matrix implementation in pure Python
    """
    data: list[list[Any]]

    def __init__(self, rows: int = 0, cols: int = 0, initial_value: T = NA(T)):
        self.rows = rows
        self.cols = cols
        self.data = [[initial_value for _ in range(cols)] for _ in range(rows)]
        #: The na of the element type: the cells of a row or column added without values. An
        #: na initial value already is that na; ``type()`` of it would be ``NA`` itself.
        self.na_element: Any = (initial_value if not (initial_value == initial_value)
                                else NA(type(initial_value)))

    def __iter__(self) -> Iterator[list[T]]:
        """
        Iterate over the rows of the matrix.

        :return: An iterator over the row lists themselves, not over copies of them.
        """
        # Pine's ``for row in m`` hands back the row arrays of the matrix, so a
        # write through the loop variable is visible in the matrix -- unlike
        # ``matrix.row()``, which copies. The indexed ``for [i, row] in m`` form
        # is an ``enumerate()`` over the same iterator.
        return iter(self.data)

    @property
    def rows(self) -> int:
        """Return the number of rows in the matrix."""
        return self._rows

    @rows.setter
    def rows(self, value: int) -> None:
        self._rows = value

    def add_row(self, row: int | None = None, array_id: list[Any] | None = None) -> None:
        """
        Add a row at the specified index of the matrix.

        The row can consist of NA values, or an array can be used to provide values.

        :param row: The index where the new row will be inserted. If None, appends to the end.
        :param array_id: Array whose size must match the column count. A 0x0 matrix takes its
                        column count from the array size.
        :raises IndexError: If row index is out of bounds.
        :raises ValueError: If the array size does not match the column count.
        """
        row_idx: int = self.rows if row is None else row

        if row_idx < 0 or row_idx > self.rows:
            raise IndexError(f"Row index {row_idx} out of bounds for matrix with {self.rows} rows")

        if array_id is not None:
            if (self.rows != 0 or self.cols != 0) and len(array_id) != self.cols:
                raise ValueError(
                    f"Row array size {len(array_id)} does not match the matrix column count "
                    f"{self.cols}"
                )
            self.cols = len(array_id)
            new_row = array_id.copy()
        else:
            new_row = [self.na_element] * self.cols

        self.data.insert(row_idx, new_row)
        self.rows += 1

    def avg(self) -> float | int | NA[float]:
        """
        Calculate the average of all elements in the matrix.

        :return: The average value of all non-NA elements in the matrix.
        :rtype: Union[float, int]
        """
        total = 0
        count = 0
        for row in self.data:
            for val in row:
                if val == val:
                    total += val
                    count += 1
        return total / count if count > 0 else na_float

    def col(self, column: int) -> list[T]:
        """
        Create a one-dimensional array from the elements of a matrix column.

        :param column: Index of the required column.
        :return: An array containing the column values.
        :raises IndexError: If column index is out of bounds.
        """
        if column < 0 or column >= self.cols:
            raise IndexError(f"Column index {column} out of bounds")
        return [self.data[row][column] for row in range(self.rows)]

    def columns(self) -> int:
        """
        Return the number of columns in the matrix.

        :return: The number of columns.
        """
        return self.cols

    def concat(self, other: Matrix) -> Matrix:
        """
        Append another matrix to this matrix.

        :param other: Matrix object whose elements will be appended.
        :return: This matrix after concatenation.
        :raises ValueError: If matrices don't have the same number of columns.
        """
        if self.cols != other.cols:
            raise ValueError("Matrices must have the same number of columns")

        # Append all rows from other matrix
        for row in other.data:
            self.data.append(row[:])
        self.rows += other.rows
        return self

    def copy(self) -> Matrix:
        """
        Create a new matrix which is a copy of the original.

        :return: A new matrix object whose storage is independent while its elements
                 stay the very same objects.
        """
        # Pine's matrix.copy() is shallow, like array.copy(): only the row storage is
        # new, so reference-type elements (drawings, chart points, UDT instances) stay
        # shared and a mutation through the copy is visible on the original.
        new_matrix = Matrix(self.rows, self.cols, self.na_element)
        new_matrix.data = [row.copy() for row in self.data]
        return new_matrix

    def det(self) -> float | int:
        """
        Return the determinant of a square matrix.

        :return: The determinant value of the matrix.
        :raises ValueError: If matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices")

        if self.rows == 0:
            return 1
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        # For larger matrices, use LU decomposition
        return self._lu_determinant()

    def diff(self, other: Matrix | int | float) -> Matrix:
        """
        Return a new matrix resulting from subtraction.

        :param other: Matrix object or scalar value to be subtracted.
        :return: A new matrix containing the difference.
        :raises ValueError: If matrix dimensions don't match.
        """
        result = Matrix(self.rows, self.cols, self.na_element)

        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have same dimensions")
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
        else:
            # Scalar subtraction
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other

        return result

    def elements_count(self) -> int:
        """
        Return the total number of all matrix elements.

        :return: The total number of elements (rows * columns).
        """
        return self.rows * self.cols

    def fill(self, value: T, from_row: int = 0, to_row: int | None = None,
             from_column: int = 0, to_column: int | None = None) -> None:
        """
        Fill a rectangular area of the matrix with the specified value.

        :param value: The value to fill with.
        :param from_row: Row index from which the fill will begin (inclusive).
        :param to_row: Row index where the fill will end (exclusive). If None, fills to end.
        :param from_column: Column index from which the fill will begin (inclusive).
        :param to_column: Column index where the fill will end (exclusive). If None, fills to end.
        """
        _to_row: int = self.rows if to_row is None else to_row
        _to_col: int = self.cols if to_column is None else to_column

        for i in range(from_row, _to_row):
            for j in range(from_column, _to_col):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    self.data[i][j] = value

    def get(self, row: int, column: int) -> T:
        """
        Return the element with the specified index of the matrix.

        :param row: Index of the required row.
        :param column: Index of the required column.
        :return: The value at the specified position.
        :raises IndexError: If indices are out of bounds.
        """
        if row < 0 or row >= self.rows or column < 0 or column >= self.cols:
            raise IndexError(f"Index ({row}, {column}) out of bounds")
        return self.data[row][column]

    def set(self, row: int, column: int, value: T) -> None:
        """
        Assign value to the element at the specified row and column.

        :param row: The row index of the element to be modified.
        :param column: The column index of the element to be modified.
        :param value: The new value to be set.
        :raises IndexError: If indices are out of bounds.
        """
        if row < 0 or row >= self.rows or column < 0 or column >= self.cols:
            raise IndexError(f"Index ({row}, {column}) out of bounds")
        self.data[row][column] = value

    def max(self) -> float | int | NA[float]:
        """
        Return the largest value from the matrix elements.

        :return: The maximum value from the matrix.
        """
        max_val: float | int | None = None
        for row in self.data:
            for val in row:
                if val == val:
                    if max_val is None or val > max_val:
                        max_val = val
        return max_val if max_val is not None else na_float

    def min(self) -> float | int | NA[float]:
        """
        Return the smallest value from the matrix elements.

        :return: The minimum value from the matrix.
        """
        min_val: float | int | None = None
        for row in self.data:
            for val in row:
                if val == val:
                    if min_val is None or val < min_val:
                        min_val = val
        return min_val if min_val is not None else na_float

    def median(self) -> float | int | NA[float]:
        """
        Calculate the median ("middle" value) of matrix elements.

        :return: The median value of all non-NA elements.
        """
        values = []
        for row in self.data:
            for val in row:
                if val == val:
                    values.append(val)

        if not values:
            return na_float

        values.sort()
        n = len(values)
        if n % 2 == 0:
            return (values[n // 2 - 1] + values[n // 2]) / 2
        else:
            return values[n // 2]

    def mode(self) -> float | int | NA[float]:
        """
        Calculate the mode of the matrix.

        Returns the most frequently occurring value. When there are multiple
        values occurring equally frequently, returns the smallest.

        :return: The most frequently occurring value from the matrix.
        """
        values = []
        for row in self.data:
            for val in row:
                if val == val:
                    values.append(val)

        if not values:
            return na_float

        counter = Counter(values)
        if not counter:
            return na_float

        # Get all values with max count
        max_count = max(counter.values())
        modes = [val for val, count in counter.items() if count == max_count]

        # Return smallest if multiple modes
        return min(modes)

    def mult(self, other: Matrix | list[T] | int | float) -> Matrix | list[T]:
        """
        Return the product of matrices, matrix and vector, or matrix and scalar.

        :param other: Second matrix object, array, or scalar value.
        :return: A new matrix (matrix multiplication) or array (vector multiplication).
        :raises ValueError: If dimensions are incompatible for multiplication.
        """
        if isinstance(other, Matrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError("Invalid dimensions for matrix multiplication")

            result = Matrix(self.rows, other.cols, self.na_element)
            for i in range(self.rows):
                for j in range(other.cols):
                    sum_val = 0
                    for k in range(self.cols):
                        sum_val += self.data[i][k] * other.data[k][j]
                    result.data[i][j] = sum_val
            return result

        elif isinstance(other, list):
            # Matrix-vector multiplication
            if self.cols != len(other):
                raise ValueError("Invalid dimensions for matrix-vector multiplication")

            result = []
            for i in range(self.rows):
                sum_val = 0
                for j in range(self.cols):
                    sum_val += self.data[i][j] * other[j]
                result.append(sum_val)
            return result

        else:
            # Scalar multiplication
            result = Matrix(self.rows, self.cols, self.na_element)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result

    def remove_col(self, column: int | None = None) -> list[T]:
        """
        Remove the column at the specified index and return its values.

        :param column: The index of the column to be removed. If None, removes last column.
        :return: An array containing the removed column's values.
        :raises IndexError: If column index is out of bounds.
        """
        col_idx: int = (self.cols - 1) if column is None else column

        if col_idx < 0 or col_idx >= self.cols:
            raise IndexError(f"Column index {col_idx} out of bounds")

        removed = []
        for row in self.data:
            removed.append(row.pop(col_idx))
        self.cols -= 1
        return removed

    def remove_row(self, row: int | None = None) -> list[T]:
        """
        Remove the row at the specified index and return its values.

        :param row: The index of the row to be removed. If None, removes last row.
        :return: An array containing the removed row's values.
        :raises IndexError: If row index is out of bounds.
        """
        row_idx: int = (self.rows - 1) if row is None else row

        if row_idx < 0 or row_idx >= self.rows:
            raise IndexError(f"Row index {row_idx} out of bounds")

        removed = self.data.pop(row_idx)
        self.rows -= 1
        return removed

    def reshape(self, rows: int, columns: int) -> None:
        """
        Rebuild the matrix to the specified dimensions.

        :param rows: The number of rows of the reshaped matrix.
        :param columns: The number of columns of the reshaped matrix.
        :raises ValueError: If new shape doesn't have same number of elements.
        """
        if rows * columns != self.rows * self.cols:
            raise ValueError("New shape must have same number of elements")

        # Flatten to 1D
        flat = []
        for row in self.data:
            flat.extend(row)

        # Reshape to new dimensions
        self.data = []
        for i in range(rows):
            self.data.append(flat[i * columns:(i + 1) * columns])

        self.rows = rows
        self.cols = columns

    def reverse(self) -> None:
        """
        Reverse the order of rows and columns in the matrix.

        The first row and first column become the last, and the last become the first.
        """
        # Reverse rows
        self.data.reverse()
        # Reverse each row
        for row in self.data:
            row.reverse()

    def row(self, row: int) -> list[T]:
        """
        Create a one-dimensional array from the elements of a matrix row.

        :param row: Index of the required row.
        :return: An array containing the row values.
        :raises IndexError: If row index is out of bounds.
        """
        if row < 0 or row >= self.rows:
            raise IndexError(f"Row index {row} out of bounds")
        return self.data[row][:]

    def sort(self, column: int = 0, order: str = 'ascending') -> None:
        """
        Rearrange rows following the sorted order of values in the specified column.

        :param column: Index of the column whose sorted values determine the new order of rows.
        :param order: The sort order. 'ascending' (default) or 'descending'.
        :raises IndexError: If column index is out of bounds.
        """
        if column < 0 or column >= self.cols:
            raise IndexError(f"Column index {column} out of bounds")

        reverse = order == 'descending'
        self.data.sort(key=lambda row: row[column], reverse=reverse)

    def submatrix(self, from_row: int = 0, to_row: int | None = None,
                  from_column: int = 0, to_column: int | None = None) -> Matrix:
        """
        Extract a submatrix within the specified indices.

        :param from_row: Row index from which extraction begins (inclusive).
        :param to_row: Row index where extraction ends (exclusive). If None, extracts to end.
        :param from_column: Column index from which extraction begins (inclusive).
        :param to_column: Column index where extraction ends (exclusive). If None, extracts to end.
        :return: A new matrix containing the submatrix.
        """
        _to_row: int = self.rows if to_row is None else to_row
        _to_col: int = self.cols if to_column is None else to_column

        result = Matrix(_to_row - from_row, _to_col - from_column, self.na_element)
        for i in range(from_row, _to_row):
            for j in range(from_column, _to_col):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    result.data[i - from_row][j - from_column] = self.data[i][j]

        return result

    def sum(self, other: Matrix | int | float) -> Matrix:
        """
        Return a new matrix resulting from addition.

        :param other: Second matrix object or scalar value.
        :return: A new matrix containing the sum.
        :raises ValueError: If matrix dimensions don't match.
        """
        result = Matrix(self.rows, self.cols, self.na_element)

        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have same dimensions")
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
        else:
            # Scalar addition
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other

        return result

    def swap_columns(self, column1: int, column2: int) -> None:
        """
        Swap the columns at the specified indices.

        :param column1: Index of the first column to be swapped.
        :param column2: Index of the second column to be swapped.
        :raises IndexError: If column indices are out of bounds.
        """
        if column1 < 0 or column1 >= self.cols or column2 < 0 or column2 >= self.cols:
            raise IndexError("Column index out of bounds")

        for row in self.data:
            row[column1], row[column2] = row[column2], row[column1]

    def swap_rows(self, row1: int, row2: int) -> None:
        """
        Swap the rows at the specified indices.

        :param row1: Index of the first row to be swapped.
        :param row2: Index of the second row to be swapped.
        :raises IndexError: If row indices are out of bounds.
        """
        if row1 < 0 or row1 >= self.rows or row2 < 0 or row2 >= self.rows:
            raise IndexError("Row index out of bounds")

        self.data[row1], self.data[row2] = self.data[row2], self.data[row1]

    def trace(self) -> float | int:
        """
        Calculate the trace of a matrix (sum of the main diagonal's elements).

        :return: The trace value of the matrix.
        :raises ValueError: If matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Trace can only be calculated for square matrices")

        trace_sum = 0
        for i in range(self.rows):
            trace_sum += self.data[i][i]
        return trace_sum

    def transpose(self) -> Matrix:
        """
        Create a new transposed version of the matrix.

        This interchanges the row and column index of each element.

        :return: A new matrix containing the transposed version.
        """
        result = Matrix(self.cols, self.rows, self.na_element)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    # Helper methods
    def _lu_determinant(self) -> float | int:
        """
        Calculate determinant using LU decomposition.

        :return: The determinant value.
        """
        # Simplified LU for determinant calculation
        n = self.rows
        lu = [row[:] for row in self.data]  # Copy matrix

        det = 1
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(lu[k][i]) > abs(lu[max_row][i]):
                    max_row = k

            # Swap rows if needed
            if max_row != i:
                lu[i], lu[max_row] = lu[max_row], lu[i]
                det *= -1

            # Check for zero pivot
            if lu[i][i] == 0:
                return 0

            det *= lu[i][i]

            # Eliminate column
            for k in range(i + 1, n):
                factor = lu[k][i] / lu[i][i]
                for j in range(i + 1, n):
                    lu[k][j] -= factor * lu[i][j]

        return det

    # is_* check methods
    def is_antidiagonal(self) -> bool:
        """
        Determine if the matrix is anti-diagonal.

        All elements outside the secondary diagonal are zero.

        :return: True if matrix is anti-diagonal, False otherwise.
        """
        if self.rows != self.cols:
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i + j != self.rows - 1 and self.data[i][j] != 0:
                    return False
        return True

    def is_antisymmetric(self) -> bool:
        """
        Determine if a matrix is antisymmetric.

        A matrix is antisymmetric if its transpose equals its negative.

        :return: True if matrix is antisymmetric, False otherwise.
        """
        if self.rows != self.cols:
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != -self.data[j][i]:
                    return False
        return True

    def is_binary(self) -> bool:
        """
        Determine if the matrix is binary.

        A matrix is binary when all elements are 0 or 1.

        :return: True if matrix is binary, False otherwise.
        """
        for row in self.data:
            for val in row:
                if val == val and val != 0 and val != 1:
                    return False
        return True

    def is_diagonal(self) -> bool:
        """
        Determine if the matrix is diagonal.

        All elements outside the main diagonal are zero.

        :return: True if matrix is diagonal, False otherwise.
        """
        if self.rows != self.cols:
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False
        return True

    def is_identity(self) -> bool:
        """
        Determine if a matrix is an identity matrix.

        Elements on the main diagonal are ones and zeros elsewhere.

        :return: True if matrix is identity, False otherwise.
        """
        if self.rows != self.cols:
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                expected = 1 if i == j else 0
                if self.data[i][j] != expected:
                    return False
        return True

    def is_square(self) -> bool:
        """
        Determine if the matrix is square.

        A matrix is square if it has the same number of rows and columns.

        :return: True if matrix is square, False otherwise.
        """
        return self.rows == self.cols

    def is_stochastic(self) -> bool:
        """
        Determine if the matrix is stochastic.

        A matrix is stochastic if all row sums equal 1.

        :return: True if matrix is stochastic, False otherwise.
        """
        for row in self.data:
            row_sum = sum(val for val in row if val == val)
            if abs(row_sum - 1.0) > 1e-10:
                return False
        return True

    def is_symmetric(self) -> bool:
        """
        Determine if a square matrix is symmetric.

        Elements are symmetric with respect to the main diagonal.

        :return: True if matrix is symmetric, False otherwise.
        """
        if self.rows != self.cols:
            return False

        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if self.data[i][j] != self.data[j][i]:
                    return False
        return True

    def is_triangular(self) -> bool:
        """
        Determine if the matrix is triangular.

        A matrix is triangular if all elements above or below the main diagonal are zero.

        :return: True if matrix is triangular, False otherwise.
        """
        if self.rows != self.cols:
            return False

        # Check if upper triangular
        upper = True
        for i in range(1, self.rows):
            for j in range(i):
                if self.data[i][j] != 0:
                    upper = False
                    break
            if not upper:
                break

        # Check if lower triangular
        lower = True
        for i in range(self.rows - 1):
            for j in range(i + 1, self.cols):
                if self.data[i][j] != 0:
                    lower = False
                    break
            if not lower:
                break

        return upper or lower

    def is_zero(self) -> bool:
        """
        Determine if all elements of the matrix are zero.

        :return: True if all elements are zero, False otherwise.
        """
        for row in self.data:
            for val in row:
                if val == val and val != 0:
                    return False
        return True

    # Additional methods
    def kron(self, other: Matrix) -> Matrix:
        """
        Return the Kronecker product of two matrices.

        :param other: Second matrix object.
        :return: A new matrix containing the Kronecker product.
        """
        result_rows = self.rows * other.rows
        result_cols = self.cols * other.cols
        result = Matrix(result_rows, result_cols, self.na_element)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    for l in range(other.cols):
                        result.data[i * other.rows + k][j * other.cols + l] = \
                            self.data[i][j] * other.data[k][l]

        return result

    def pow(self, power: int) -> Matrix:
        """
        Calculate the product of the matrix by itself power times.

        :param power: The number of times the matrix will be multiplied by itself.
        :return: The matrix raised to the specified power.
        :raises ValueError: If matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Power can only be calculated for square matrices")

        if power == 0:
            # Return identity matrix
            result = Matrix(self.rows, self.cols, 0)
            for i in range(self.rows):
                result.data[i][i] = 1
            return result

        result = self.copy()
        for _ in range(power - 1):
            result = cast(Matrix, result.mult(self))

        return result

    def rank(self) -> int:
        """
        Calculate the rank of the matrix.

        :return: The rank of the matrix.
        """
        # Simple rank calculation using row echelon form
        m = [row[:] for row in self.data]  # Copy matrix
        rows, cols = self.rows, self.cols
        rank = 0

        for col in range(cols):
            # Find pivot
            pivot_row = None
            for row in range(rank, rows):
                if m[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap rows
            m[rank], m[pivot_row] = m[pivot_row], m[rank]

            # Eliminate column
            for row in range(rank + 1, rows):
                if m[row][col] != 0:
                    factor = m[row][col] / m[rank][col]
                    for j in range(col, cols):
                        m[row][j] -= factor * m[rank][j]

            rank += 1

        return rank

    def inv(self) -> Matrix:
        """
        Return the inverse of a square matrix.

        The inverse is solved from the LU decomposition with partial pivoting. A pivot smaller
        than ``1e-11`` in magnitude makes the matrix singular.

        :return: A new matrix which is the inverse of this matrix.
        :raises ValueError: If matrix is not square or is singular.
        """
        if self.rows != self.cols:
            raise ValueError("Inverse can only be calculated for square matrices")
        inverse = _lu_inverse(self._float_rows())
        if inverse is None:
            raise ValueError("Matrix is singular")
        result = Matrix(self.rows, self.cols, self.na_element)
        result.data = inverse
        return result

    def pinv(self) -> Matrix:
        """
        Return the Moore-Penrose pseudoinverse of a matrix.

        A square matrix that :meth:`inv` can invert returns its inverse. A matrix with fewer rows
        than columns returns ``A^T (A A^T)^-1`` while ``A A^T`` can be inverted. Every other
        matrix, a rank-deficient one included, returns the pseudoinverse of its singular value
        decomposition, where a singular value up to ``max(rows, cols) * s_max * 2^-52`` counts
        as zero.

        :return: A new matrix of ``columns x rows`` cells containing the pseudoinverse.
        """
        result = Matrix(self.cols, self.rows, self.na_element)
        if self.rows == 0 or self.cols == 0:
            return result
        a = self._float_rows()
        pseudo: list[list[float]] | None = None
        if self.rows == self.cols:
            pseudo = _lu_inverse(a)
        elif self.rows < self.cols:
            at = _transpose(a)
            aat_inverse = _lu_inverse(_multiply(a, at))
            if aat_inverse is not None:
                pseudo = _multiply(at, aat_inverse)
        result.data = pseudo if pseudo is not None else _svd_pseudo_inverse(a)
        return result

    def _float_rows(self) -> list[list[float]]:
        """
        Copy the cells as floats for a numerical decomposition.

        :return: The cells as a new list of float rows.
        """
        return [[float(val) for val in row] for row in self.data]

    def eigenvalues(self) -> list[float]:
        """
        Return an array containing the eigenvalues of a square matrix.

        A symmetric matrix returns its eigenvalues in descending order. Any other matrix returns
        them in the order they take on the diagonal of its real Schur form; a complex conjugate
        pair contributes its real part twice.

        :return: An array containing the eigenvalues. It has the element type of the matrix, so
                 an int matrix gives an int array that still holds fractional values.
        :raises ValueError: If the matrix is empty, contains na or is not square.
        """
        return _eigen_decomposition(self._eigen_input())[0]

    def eigenvectors(self) -> Matrix:
        """
        Return a matrix of eigenvectors.

        Column ``i`` is the eigenvector of the ``i``-th element of :meth:`eigenvalues`. The
        eigenvectors of a symmetric matrix are orthonormal; those of any other matrix are not
        normalized. The two columns of a complex conjugate pair hold the real and the imaginary
        part of the complex eigenvector of the eigenvalue with the positive imaginary part.

        :return: A new matrix of the type of this one containing the eigenvectors, so an int
                 matrix gives an int matrix that still holds fractional values.
        :raises ValueError: If the matrix is empty, contains na or is not square.
        """
        result = Matrix(self.rows, self.cols, self.na_element)
        result.data = _eigen_decomposition(self._eigen_input())[1]
        return result

    def _eigen_input(self) -> list[list[float]]:
        """
        Validate the matrix for an eigen decomposition and copy it as floats.

        :return: The cells as a new list of float rows.
        :raises ValueError: If the matrix is empty, contains na or is not square.
        """
        if self.rows == 0 or self.cols == 0:
            raise ValueError("Eigen decomposition of an empty matrix")
        for row in self.data:
            for val in row:
                if isinstance(val, NA) or val != val:
                    raise ValueError("Eigen decomposition of a matrix that contains na values")
        if self.rows != self.cols:
            raise ValueError(f"Eigen decomposition of a non-square ({self.rows}x{self.cols}) "
                             f"matrix")
        return self._float_rows()

    def add_col(self, col: int | None = None, array_id: list[Any] | None = None) -> None:
        """
        Add a column at the specified index of the matrix.

        The column can consist of NA values, or an array can be used to provide values.

        :param col: The index where the new column will be inserted. If None, appends to the end.
        :param array_id: Array whose size must match the row count. A 0x0 matrix takes its row
                        count from the array size.
        :raises IndexError: If column index is out of bounds.
        :raises ValueError: If the array size does not match the row count.
        """
        col_idx: int = self.cols if col is None else col

        if col_idx < 0 or col_idx > self.cols:
            raise IndexError(f"Column index {col_idx} out of bounds")

        if array_id is not None:
            if (self.rows != 0 or self.cols != 0) and len(array_id) != self.rows:
                raise ValueError(
                    f"Column array size {len(array_id)} does not match the matrix row count "
                    f"{self.rows}"
                )
            if self.rows == 0:
                self.rows = len(array_id)
                self.data = [[] for _ in range(self.rows)]
            for i in range(self.rows):
                self.data[i].insert(col_idx, array_id[i])
        else:
            for i in range(self.rows):
                self.data[i].insert(col_idx, self.na_element)

        self.cols += 1


# Eigen decomposition
#
# Measured on TradingView (CAPITALCOM:EURUSD 60, 28 matrices up to 4x4): matrix.eigenvalues()
# and matrix.eigenvectors() return, to the last bit, what the EISPACK-derived routines of Apache
# Commons Math 3 (Apache License 2.0) compute: EigenDecomposition with TriDiagonalTransformer,
# HessenbergTransformer and SchurTransformer. A symmetric matrix goes through Householder
# tridiagonalization and the implicit QL algorithm, any other one through Hessenberg reduction,
# the Francis double-shift QR algorithm and back-substitution. The functions below port those
# routines operation by operation, so their rounding is TradingView's too: keep the order of
# every arithmetic operation when changing them.

#: Machine epsilon of the decomposition: 2^-53, half of ``sys.float_info.epsilon``
_EIGEN_EPS = 2.0 ** -53
#: Absolute tolerance of the real Schur form: a sub-diagonal entry at most this small splits it
_EIGEN_SPLIT_TOLERANCE = 1e-12
#: The smallest positive double: a value at most this far from zero counts as zero
_EIGEN_MIN_DOUBLE = 5e-324


def _eigen_decomposition(a: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Compute the real eigenvalues and the eigenvectors of a square float matrix.

    :param a: The rows of the matrix; they are not modified.
    :return: The eigenvalues and the rows of the matrix whose columns are the eigenvectors.
    :raises ValueError: If the iteration does not converge or a non-symmetric matrix has a
                        zero norm.
    """
    n = len(a)
    tolerance = 10 * n * n * _EIGEN_EPS
    for i in range(n):
        for j in range(i + 1, n):
            aij = a[i][j]
            aji = a[j][i]
            if abs(aij - aji) > max(abs(aij), abs(aji)) * tolerance:
                return _eigen_general(a)
    return _eigen_symmetric(a)


def _tridiagonalize(a: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    """
    Reduce a symmetric matrix to tridiagonal form with Householder reflections.

    :param a: The rows of the matrix; they are not modified.
    :return: The main diagonal, the secondary diagonal and the orthogonal matrix Q.
    """
    hv = [row[:] for row in a]
    m = len(hv)
    main = [0.0] * m
    secondary = [0.0] * (m - 1)
    z = [0.0] * m
    for k in range(m - 1):
        # Zero out a row and a column simultaneously
        hk = hv[k]
        main[k] = hk[k]
        x_norm_sqr = 0.0
        for j in range(k + 1, m):
            c = hk[j]
            x_norm_sqr += c * c
        alpha = -math.sqrt(x_norm_sqr) if hk[k + 1] > 0 else math.sqrt(x_norm_sqr)
        secondary[k] = alpha
        if alpha != 0.0:
            # Apply the Householder transform from the left and the right simultaneously,
            # reading and updating the upper triangle only
            hk[k + 1] -= alpha
            beta = -1 / (alpha * hk[k + 1])
            for i in range(k + 1, m):
                z[i] = 0.0
            for i in range(k + 1, m):
                hi = hv[i]
                hki = hk[i]
                zi = hi[i] * hki
                for j in range(i + 1, m):
                    hij = hi[j]
                    zi += hij * hk[j]
                    z[j] += hij * hki
                z[i] = beta * (z[i] + zi)
            gamma = 0.0
            for i in range(k + 1, m):
                gamma += z[i] * hk[i]
            gamma *= beta / 2
            for i in range(k + 1, m):
                z[i] -= gamma * hk[i]
            for i in range(k + 1, m):
                hi = hv[i]
                for j in range(i, m):
                    hi[j] -= hk[i] * z[j] + z[i] * hk[j]
    main[m - 1] = hv[m - 1][m - 1]

    # Build Q^T from the Householder vectors, then transpose it
    qt = [[0.0] * m for _ in range(m)]
    for k in range(m - 1, 0, -1):
        hk = hv[k - 1]
        qt[k][k] = 1.0
        if hk[k] != 0.0:
            inv = 1.0 / (secondary[k - 1] * hk[k])
            beta = 1.0 / secondary[k - 1]
            qt[k][k] = 1 + beta * hk[k]
            for i in range(k + 1, m):
                qt[k][i] = beta * hk[i]
            for j in range(k + 1, m):
                beta = 0.0
                for i in range(k + 1, m):
                    beta += qt[j][i] * hk[i]
                beta *= inv
                qt[j][k] = beta * hk[k]
                for i in range(k + 1, m):
                    qt[j][i] += beta * hk[i]
    qt[0][0] = 1.0
    return main, secondary, [[qt[j][i] for j in range(m)] for i in range(m)]


def _eigen_symmetric(a: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Eigen decomposition of a symmetric matrix with the implicit QL algorithm.

    :param a: The rows of the matrix; they are not modified.
    :return: The eigenvalues in descending order and the orthonormal eigenvector columns.
    :raises ValueError: If the QL iteration does not converge.
    """
    main, secondary, z = _tridiagonalize(a)
    n = len(main)
    d = main
    e = secondary + [0.0]

    # Make null any main and secondary value too small to be significant
    max_abs = 0.0
    for i in range(n):
        if abs(d[i]) > max_abs:
            max_abs = abs(d[i])
        if abs(e[i]) > max_abs:
            max_abs = abs(e[i])
    if max_abs != 0:
        for i in range(n):
            if abs(d[i]) <= _EIGEN_EPS * max_abs:
                d[i] = 0.0
            if abs(e[i]) <= _EIGEN_EPS * max_abs:
                e[i] = 0.0

    for j in range(n):
        iterations = 0
        while True:
            m = j
            while m < n - 1:
                delta = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) + delta == delta:
                    break
                m += 1
            if m == j:
                break
            if iterations == 30:
                raise ValueError("Eigen decomposition did not converge")
            iterations += 1
            q = (d[j + 1] - d[j]) / (2 * e[j])
            t = math.sqrt(1 + q * q)
            if q < 0.0:
                q = d[m] - d[j] + e[j] / (q - t)
            else:
                q = d[m] - d[j] + e[j] / (q + t)
            u = 0.0
            s = 1.0
            c = 1.0
            i = m - 1
            while i >= j:
                p = s * e[i]
                h = c * e[i]
                if abs(p) >= abs(q):
                    c = q / p
                    t = math.sqrt(c * c + 1.0)
                    e[i + 1] = p * t
                    s = 1.0 / t
                    c *= s
                else:
                    s = p / q
                    t = math.sqrt(s * s + 1.0)
                    e[i + 1] = q * t
                    c = 1.0 / t
                    s *= c
                if e[i + 1] == 0.0:
                    d[i + 1] -= u
                    e[m] = 0.0
                    break
                q = d[i + 1] - u
                t = (d[i] - q) * s + 2.0 * c * h
                u = s * t
                d[i + 1] = q + u
                q = c * t - h
                for row in z:
                    p = row[i + 1]
                    row[i + 1] = s * row[i] + c * p
                    row[i] = c * row[i] - s * p
                i -= 1
            if t == 0.0 and i >= j:
                continue
            d[j] -= u
            e[j] = q
            e[m] = 0.0

    # Sort the eigenvalues, with their vectors, in descending order
    for i in range(n):
        k = i
        p = d[i]
        for j in range(i + 1, n):
            if d[j] > p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            for row in z:
                row[i], row[k] = row[k], row[i]

    # Make null any eigenvalue too small to be significant
    max_abs = 0.0
    for i in range(n):
        if abs(d[i]) > max_abs:
            max_abs = abs(d[i])
    if max_abs != 0.0:
        for i in range(n):
            if abs(d[i]) < _EIGEN_EPS * max_abs:
                d[i] = 0.0
    return d, z


def _hessenberg(a: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Reduce a square matrix to upper Hessenberg form with Householder reflections.

    :param a: The rows of the matrix; they are not modified.
    :return: The Hessenberg matrix H and the orthogonal matrix P of ``A = P H P^T``.
    """
    hv = [row[:] for row in a]
    n = len(hv)
    high = n - 1
    ort = [0.0] * n
    for m in range(1, high):
        scale = 0.0
        for i in range(m, high + 1):
            scale += abs(hv[i][m - 1])
        if scale > _EIGEN_MIN_DOUBLE:
            h = 0.0
            for i in range(high, m - 1, -1):
                ort[i] = hv[i][m - 1] / scale
                h += ort[i] * ort[i]
            g = -math.sqrt(h) if ort[m] > 0 else math.sqrt(h)
            h -= ort[m] * g
            ort[m] -= g
            # H = (I - u u^T / h) H (I - u u^T / h)
            for j in range(m, n):
                f = 0.0
                for i in range(high, m - 1, -1):
                    f += ort[i] * hv[i][j]
                f /= h
                for i in range(m, high + 1):
                    hv[i][j] -= f * ort[i]
            for i in range(high + 1):
                f = 0.0
                for j in range(high, m - 1, -1):
                    f += ort[j] * hv[i][j]
                f /= h
                for j in range(m, high + 1):
                    hv[i][j] -= f * ort[j]
            ort[m] = scale * ort[m]
            hv[m][m - 1] = scale * g

    p = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for m in range(high - 1, 0, -1):
        if hv[m][m - 1] != 0.0:
            for i in range(m + 1, high + 1):
                ort[i] = hv[i][m - 1]
            for j in range(m, high + 1):
                g = 0.0
                for i in range(m, high + 1):
                    g += ort[i] * p[i][j]
                # Double division avoids possible underflow
                g = (g / ort[m]) / hv[m][m - 1]
                for i in range(m, high + 1):
                    p[i][j] += g * ort[i]

    h = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if i > 0:
            h[i][i - 1] = hv[i][i - 1]
        for j in range(i, n):
            h[i][j] = hv[i][j]
    return h, p


def _hessenberg_norm(t: list[list[float]]) -> float:
    """
    Sum the absolute values of the upper Hessenberg part of a matrix.

    :param t: The rows of the matrix.
    :return: The sum.
    """
    n = len(t)
    norm = 0.0
    for i in range(n):
        for j in range(max(i - 1, 0), n):
            norm += abs(t[i][j])
    return norm


def _real_schur(a: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Compute the real Schur form of a square matrix with the Francis double-shift QR algorithm.

    :param a: The rows of the matrix; they are not modified.
    :return: The quasi upper triangular T and the orthogonal P of ``A = P T P^T``.
    :raises ValueError: If the QR iteration does not converge.
    """
    t, p = _hessenberg(a)
    n = len(t)
    norm = _hessenberg_norm(t)
    exshift = 0.0
    iteration = 0
    iu = n - 1
    while iu >= 0:
        # Look for a single small sub-diagonal element
        il = iu
        while il > 0:
            s = abs(t[il - 1][il - 1]) + abs(t[il][il])
            if s == 0.0:
                s = norm
            if abs(t[il][il - 1]) < _EIGEN_EPS * s:
                break
            il -= 1

        if il == iu:
            # One root found
            t[iu][iu] += exshift
            iu -= 1
            iteration = 0
        elif il == iu - 1:
            # Two roots found: split a real pair into an upper triangular block
            pp = (t[iu - 1][iu - 1] - t[iu][iu]) / 2.0
            q = pp * pp + t[iu][iu - 1] * t[iu - 1][iu]
            t[iu][iu] += exshift
            t[iu - 1][iu - 1] += exshift
            if q >= 0:
                z = math.sqrt(abs(q))
                if pp >= 0:
                    z = pp + z
                else:
                    z = pp - z
                x = t[iu][iu - 1]
                s = abs(x) + abs(z)
                pp = x / s
                q = z / s
                r = math.sqrt(pp * pp + q * q)
                pp /= r
                q /= r
                for j in range(iu - 1, n):
                    z = t[iu - 1][j]
                    t[iu - 1][j] = q * z + pp * t[iu][j]
                    t[iu][j] = q * t[iu][j] - pp * z
                for i in range(iu + 1):
                    z = t[i][iu - 1]
                    t[i][iu - 1] = q * z + pp * t[i][iu]
                    t[i][iu] = q * t[i][iu] - pp * z
                for row in p:
                    z = row[iu - 1]
                    row[iu - 1] = q * z + pp * row[iu]
                    row[iu] = q * row[iu] - pp * z
            iu -= 2
            iteration = 0
        else:
            # No convergence yet: form the shift
            x = t[iu][iu]
            y = 0.0
            w = 0.0
            if il < iu:
                y = t[iu - 1][iu - 1]
                w = t[iu][iu - 1] * t[iu - 1][iu]
            if iteration == 10:
                # Wilkinson's original ad hoc shift
                exshift += x
                for i in range(iu + 1):
                    t[i][i] -= x
                s = abs(t[iu][iu - 1]) + abs(t[iu - 1][iu - 2])
                x = 0.75 * s
                y = 0.75 * s
                w = -0.4375 * s * s
            if iteration == 30:
                # MATLAB's new ad hoc shift
                s = (y - x) / 2.0
                s = s * s + w
                if s > 0.0:
                    s = math.sqrt(s)
                    if y < x:
                        s = -s
                    s = x - w / ((y - x) / 2.0 + s)
                    for i in range(iu + 1):
                        t[i][i] -= s
                    exshift += s
                    x = y = w = 0.964
            iteration += 1
            if iteration > 100:
                raise ValueError("Eigen decomposition did not converge")

            # Look for two consecutive small sub-diagonal elements
            im = iu - 2
            h0 = h1 = h2 = 0.0
            while im >= il:
                z = t[im][im]
                r = x - z
                s = y - z
                h0 = (r * s - w) / t[im + 1][im] + t[im][im + 1]
                h1 = t[im + 1][im + 1] - z - r - s
                h2 = t[im + 2][im + 1]
                if im == il:
                    break
                lhs = abs(t[im][im - 1]) * (abs(h1) + abs(h2))
                rhs = abs(h0) * (abs(t[im - 1][im - 1]) + abs(z) + abs(t[im + 1][im + 1]))
                if lhs < _EIGEN_EPS * rhs:
                    break
                im -= 1

            _double_qr_step(t, p, il, im, iu, h0, h1, h2)
    return t, p


def _double_qr_step(t: list[list[float]], p: list[list[float]], il: int, im: int, iu: int,
                    h0: float, h1: float, h2: float) -> None:
    """
    Perform a double-shift QR step on rows ``il..iu`` and columns ``im..iu`` in place.

    :param t: The rows of the Hessenberg matrix being reduced.
    :param p: The rows of the accumulated orthogonal transformation.
    :param il: The first row of the active block.
    :param im: The row the step starts at.
    :param iu: The last row of the active block.
    :param h0: First component of the initial Householder vector.
    :param h1: Second component of the initial Householder vector.
    :param h2: Third component of the initial Householder vector.
    """
    n = len(t)
    pp = h0
    q = h1
    r = h2
    x = 0.0
    for k in range(im, iu):
        notlast = k != iu - 1
        if k != im:
            pp = t[k][k - 1]
            q = t[k + 1][k - 1]
            r = t[k + 2][k - 1] if notlast else 0.0
            x = abs(pp) + abs(q) + abs(r)
            if x <= _EIGEN_EPS:
                continue
            pp /= x
            q /= x
            r /= x
        s = math.sqrt(pp * pp + q * q + r * r)
        if pp < 0.0:
            s = -s
        if s != 0.0:
            if k != im:
                t[k][k - 1] = -s * x
            elif il != im:
                t[k][k - 1] = -t[k][k - 1]
            pp += s
            x = pp / s
            y = q / s
            z = r / s
            q /= pp
            r /= pp

            # Row modification
            for j in range(k, n):
                pp = t[k][j] + q * t[k + 1][j]
                if notlast:
                    pp += r * t[k + 2][j]
                    t[k + 2][j] -= pp * z
                t[k][j] -= pp * x
                t[k + 1][j] -= pp * y

            # Column modification
            for i in range(min(iu, k + 3) + 1):
                pp = x * t[i][k] + y * t[i][k + 1]
                if notlast:
                    pp += z * t[i][k + 2]
                    t[i][k + 2] -= pp * r
                t[i][k] -= pp
                t[i][k + 1] -= pp * q

            # Accumulate transformations
            for row in p:
                pp = x * row[k] + y * row[k + 1]
                if notlast:
                    pp += z * row[k + 2]
                    row[k + 2] -= pp * r
                row[k] -= pp
                row[k + 1] -= pp * q

    # Clean up the pivot elements
    for i in range(im + 2, iu + 1):
        t[i][i - 2] = 0.0
        if i > im + 2:
            t[i][i - 3] = 0.0


def _complex_div(xr: float, xi: float, yr: float, yi: float) -> tuple[float, float]:
    """
    Divide two complex numbers with Smith's algorithm.

    :param xr: Real part of the dividend.
    :param xi: Imaginary part of the dividend.
    :param yr: Real part of the divisor.
    :param yi: Imaginary part of the divisor.
    :return: The real and the imaginary part of the quotient.
    """
    if abs(yr) < abs(yi):
        q = yr / yi
        denominator = yr * q + yi
        return (xr * q + xi) / denominator, (xi * q - xr) / denominator
    q = yi / yr
    denominator = yi * q + yr
    return (xi * q + xr) / denominator, (xi - xr * q) / denominator


def _eigen_general(a: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Eigen decomposition of a non-symmetric matrix through its real Schur form.

    :param a: The rows of the matrix; they are not modified.
    :return: The real parts of the eigenvalues and the eigenvector columns.
    :raises ValueError: If the QR iteration does not converge or the matrix has a zero norm.
    """
    t, v = _real_schur(a)
    n = len(t)

    # Read the eigenvalues off the 1x1 and 2x2 diagonal blocks
    re = [0.0] * n
    im = [0.0] * n
    i = 0
    while i < n:
        if i == n - 1 or abs(t[i + 1][i]) <= _EIGEN_SPLIT_TOLERANCE:
            re[i] = t[i][i]
        else:
            x = t[i + 1][i + 1]
            pp = 0.5 * (t[i][i] - x)
            z = math.sqrt(abs(pp * pp + t[i + 1][i] * t[i][i + 1]))
            re[i] = x + pp
            im[i] = z
            re[i + 1] = x + pp
            im[i + 1] = -z
            i += 1
        i += 1

    norm = _hessenberg_norm(t)
    if norm <= _EIGEN_SPLIT_TOLERANCE:
        raise ValueError("Eigen decomposition of a matrix with zero norm")

    # Back-substitute to find the vectors of the upper triangular form
    r = s = z = 0.0
    for idx in range(n - 1, -1, -1):
        pp = re[idx]
        q = im[idx]
        if abs(q) <= _EIGEN_MIN_DOUBLE:
            # Real vector
            ll = idx
            t[idx][idx] = 1.0
            for i in range(idx - 1, -1, -1):
                w = t[i][i] - pp
                r = 0.0
                for j in range(ll, idx + 1):
                    r += t[i][j] * t[j][idx]
                if im[i] < -_EIGEN_SPLIT_TOLERANCE:
                    z = w
                    s = r
                else:
                    ll = i
                    if abs(im[i]) <= _EIGEN_MIN_DOUBLE:
                        if w != 0.0:
                            t[i][idx] = -r / w
                        else:
                            t[i][idx] = -r / (_EIGEN_EPS * norm)
                    else:
                        # Solve the real equations
                        x = t[i][i + 1]
                        y = t[i + 1][i]
                        q = (re[i] - pp) * (re[i] - pp) + im[i] * im[i]
                        tt = (x * s - z * r) / q
                        t[i][idx] = tt
                        if abs(x) > abs(z):
                            t[i + 1][idx] = (-r - w * tt) / x
                        else:
                            t[i + 1][idx] = (-s - y * tt) / z
                    # Overflow control
                    tt = abs(t[i][idx])
                    if (_EIGEN_EPS * tt) * tt > 1:
                        for j in range(i, idx + 1):
                            t[j][idx] /= tt
        elif q < 0.0:
            # Complex vector: the last vector component is imaginary, so the matrix is triangular
            ll = idx - 1
            if abs(t[idx][idx - 1]) > abs(t[idx - 1][idx]):
                t[idx - 1][idx - 1] = q / t[idx][idx - 1]
                t[idx - 1][idx] = -(t[idx][idx] - pp) / t[idx][idx - 1]
            else:
                t[idx - 1][idx - 1], t[idx - 1][idx] = _complex_div(
                    0.0, -t[idx - 1][idx], t[idx - 1][idx - 1] - pp, q)
            t[idx][idx - 1] = 0.0
            t[idx][idx] = 1.0
            for i in range(idx - 2, -1, -1):
                ra = 0.0
                sa = 0.0
                for j in range(ll, idx + 1):
                    ra += t[i][j] * t[j][idx - 1]
                    sa += t[i][j] * t[j][idx]
                w = t[i][i] - pp
                if im[i] < -_EIGEN_SPLIT_TOLERANCE:
                    z = w
                    r = ra
                    s = sa
                else:
                    ll = i
                    if abs(im[i]) <= _EIGEN_MIN_DOUBLE:
                        t[i][idx - 1], t[i][idx] = _complex_div(-ra, -sa, w, q)
                    else:
                        # Solve the complex equations
                        x = t[i][i + 1]
                        y = t[i + 1][i]
                        vr = (re[i] - pp) * (re[i] - pp) + im[i] * im[i] - q * q
                        vi = (re[i] - pp) * 2.0 * q
                        if abs(vr) <= _EIGEN_MIN_DOUBLE and abs(vi) <= _EIGEN_MIN_DOUBLE:
                            vr = (_EIGEN_EPS * norm
                                  * (abs(w) + abs(q) + abs(x) + abs(y) + abs(z)))
                        t[i][idx - 1], t[i][idx] = _complex_div(
                            x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi)
                        if abs(x) > (abs(z) + abs(q)):
                            t[i + 1][idx - 1] = (-ra - w * t[i][idx - 1] + q * t[i][idx]) / x
                            t[i + 1][idx] = (-sa - w * t[i][idx] - q * t[i][idx - 1]) / x
                        else:
                            t[i + 1][idx - 1], t[i + 1][idx] = _complex_div(
                                -r - y * t[i][idx - 1], -s - y * t[i][idx], z, q)
                    # Overflow control
                    tt = max(abs(t[i][idx - 1]), abs(t[i][idx]))
                    if (_EIGEN_EPS * tt) * tt > 1:
                        for j in range(i, idx + 1):
                            t[j][idx - 1] /= tt
                            t[j][idx] /= tt

    # Back transformation to the eigenvectors of the original matrix
    for j in range(n - 1, -1, -1):
        for i in range(n):
            z = 0.0
            for k in range(j + 1):
                z += v[i][k] * t[k][j]
            v[i][j] = z
    return re, v


# Inverse and pseudoinverse
#
# Measured on TradingView (CAPITALCOM:EURUSD 60, 29 matrices up to 5x3 and 3x5, singular,
# near-singular and zero ones included): matrix.inv() and matrix.pinv() return, to the last bit,
# what Apache Commons Math 3 (Apache License 2.0) computes with LUDecomposition (default
# singularity threshold 1e-11) and SingularValueDecomposition. matrix.inv() of a matrix with a
# pivot under the threshold is a runtime error. matrix.pinv() takes the LU inverse of a square
# matrix, ``A^T (A A^T)^-1`` with the LU inverse for fewer rows than columns, and falls back to
# the SVD solver's pseudoinverse whenever that inverse does not exist; a matrix with more rows
# than columns always goes through the SVD. The functions below port those routines operation by
# operation, so their rounding is TradingView's too: keep the order of every arithmetic
# operation when changing them.

#: A pivot of the LU decomposition smaller than this in magnitude makes the matrix singular
_LU_SINGULARITY_THRESHOLD = 1e-11
#: Relative threshold of a negligible singular value: 2^-52, ``sys.float_info.epsilon``
_SVD_EPS = 2.0 ** -52
#: Absolute threshold of a negligible singular value
_SVD_TINY = 2.0 ** -966
#: The smallest tolerance of a nonzero singular value: the square root of 2^-1022
_SVD_MIN_TOLERANCE = 2.0 ** -511


def _transpose(a: list[list[float]]) -> list[list[float]]:
    """
    Transpose a non-empty matrix.

    :param a: The rows of the matrix.
    :return: The rows of the transposed matrix.
    """
    return [[row[j] for row in a] for j in range(len(a[0]))]


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """
    Multiply two compatible non-empty matrices.

    :param a: The rows of the left matrix.
    :param b: The rows of the right matrix.
    :return: The rows of the product.
    """
    n_rows = len(a)
    n_cols = len(b[0])
    n_sum = len(b)
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for col in range(n_cols):
        b_col = [b[i][col] for i in range(n_sum)]
        for row in range(n_rows):
            a_row = a[row]
            total = 0.0
            for i in range(n_sum):
                total += a_row[i] * b_col[i]
            out[row][col] = total
    return out


def _lu_inverse(a: list[list[float]]) -> list[list[float]] | None:
    """
    Invert a square matrix through its LU decomposition with partial pivoting.

    :param a: The rows of the matrix; they are not modified.
    :return: The rows of the inverse, or None if a pivot is under the singularity threshold.
    """
    m = len(a)
    lu = [row[:] for row in a]
    pivot = list(range(m))
    for col in range(m):
        # Upper
        for row in range(col):
            lu_row = lu[row]
            total = lu_row[col]
            for i in range(row):
                total -= lu_row[i] * lu[i][col]
            lu_row[col] = total
        # Lower, keeping the best pivot
        max_row = col
        largest = -math.inf
        for row in range(col, m):
            lu_row = lu[row]
            total = lu_row[col]
            for i in range(col):
                total -= lu_row[i] * lu[i][col]
            lu_row[col] = total
            if abs(total) > largest:
                largest = abs(total)
                max_row = row
        if abs(lu[max_row][col]) < _LU_SINGULARITY_THRESHOLD:
            return None
        if max_row != col:
            lu[max_row], lu[col] = lu[col], lu[max_row]
            pivot[max_row], pivot[col] = pivot[col], pivot[max_row]
        lu_diag = lu[col][col]
        for row in range(col + 1, m):
            lu[row][col] /= lu_diag

    # Solve L U X = P I: the identity with its rows permuted
    bp = [[1.0 if pivot[row] == col else 0.0 for col in range(m)] for row in range(m)]
    for col in range(m):
        bp_col = bp[col]
        for i in range(col + 1, m):
            bp_i = bp[i]
            lu_i_col = lu[i][col]
            for j in range(m):
                bp_i[j] -= bp_col[j] * lu_i_col
    for col in range(m - 1, -1, -1):
        bp_col = bp[col]
        lu_diag = lu[col][col]
        for j in range(m):
            bp_col[j] /= lu_diag
        for i in range(col):
            bp_i = bp[i]
            lu_i_col = lu[i][col]
            for j in range(m):
                bp_i[j] -= bp_col[j] * lu_i_col
    return bp


def _exponent(x: float) -> int:
    """
    The unbiased binary exponent of a finite double, -1023 for zero and subnormals.

    :param x: The value.
    :return: The exponent.
    """
    if x == 0.0:
        return -1023
    exponent = math.frexp(x)[1] - 1
    return exponent if exponent >= -1022 else -1023


def _hypot(x: float, y: float) -> float:
    """
    ``sqrt(x^2 + y^2)`` without overflow or underflow, scaled through an intermediate exponent.

    :param x: The first leg.
    :param y: The second leg.
    :return: The hypotenuse.
    """
    if math.isinf(x) or math.isinf(y):
        return math.inf
    if math.isnan(x) or math.isnan(y):
        return math.nan
    exp_x = _exponent(x)
    exp_y = _exponent(y)
    if exp_x > exp_y + 27:
        # y is negligible with respect to x
        return abs(x)
    if exp_y > exp_x + 27:
        # x is negligible with respect to y
        return abs(y)
    # The exponent between the two, truncated toward zero
    middle_exp = int((exp_x + exp_y) / 2)
    scaled_x = math.ldexp(x, -middle_exp)
    scaled_y = math.ldexp(y, -middle_exp)
    return math.ldexp(math.sqrt(scaled_x * scaled_x + scaled_y * scaled_y), middle_exp)


def _svd(matrix: list[list[float]]) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """
    Compute the singular value decomposition ``A = U S V^T`` of a non-empty matrix.

    A matrix with fewer rows than columns is decomposed through its transpose.

    :param matrix: The rows of the matrix; they are not modified.
    :return: The singular values in descending order, U and V.
    """
    if len(matrix) < len(matrix[0]):
        transposed = True
        a = _transpose(matrix)
    else:
        transposed = False
        a = [row[:] for row in matrix]
    # m is always the larger dimension
    m = len(a)
    n = len(a[0])

    s = [0.0] * n
    u = [[0.0] * n for _ in range(m)]
    v = [[0.0] * n for _ in range(n)]
    e = [0.0] * n
    work = [0.0] * m
    # Reduce A to bidiagonal form, storing the diagonal elements in s and the super-diagonal
    # elements in e
    nct = min(m - 1, n)
    nrt = max(0, n - 2)
    for k in range(max(nct, nrt)):
        if k < nct:
            # Compute the transformation for the k-th column and place the k-th diagonal in
            # s[k], the 2-norm of the column computed without under/overflow
            s[k] = 0.0
            for i in range(k, m):
                s[k] = _hypot(s[k], a[i][k])
            if s[k] != 0.0:
                if a[k][k] < 0.0:
                    s[k] = -s[k]
                for i in range(k, m):
                    a[i][k] /= s[k]
                a[k][k] += 1.0
            s[k] = -s[k]
        for j in range(k + 1, n):
            if k < nct and s[k] != 0.0:
                # Apply the transformation
                t = 0.0
                for i in range(k, m):
                    t += a[i][k] * a[i][j]
                t = -t / a[k][k]
                for i in range(k, m):
                    a[i][j] += t * a[i][k]
            # Place the k-th row of A into e for the row transformation
            e[j] = a[k][j]
        if k < nct:
            # Place the transformation in U for the back multiplication
            for i in range(k, m):
                u[i][k] = a[i][k]
        if k < nrt:
            # Compute the k-th row transformation and place the k-th super-diagonal in e[k]
            e[k] = 0.0
            for i in range(k + 1, n):
                e[k] = _hypot(e[k], e[i])
            if e[k] != 0.0:
                if e[k + 1] < 0.0:
                    e[k] = -e[k]
                for i in range(k + 1, n):
                    e[i] /= e[k]
                e[k + 1] += 1.0
            e[k] = -e[k]
            if k + 1 < m and e[k] != 0.0:
                # Apply the transformation
                for i in range(k + 1, m):
                    work[i] = 0.0
                for j in range(k + 1, n):
                    for i in range(k + 1, m):
                        work[i] += e[j] * a[i][j]
                for j in range(k + 1, n):
                    t = -e[j] / e[k + 1]
                    for i in range(k + 1, m):
                        a[i][j] += t * work[i]
            # Place the transformation in V for the back multiplication
            for i in range(k + 1, n):
                v[i][k] = e[i]

    # Set up the final bidiagonal matrix of order p
    p = n
    if nct < n:
        s[nct] = a[nct][nct]
    if m < p:
        s[p - 1] = 0.0
    if nrt + 1 < p:
        e[nrt] = a[nrt][p - 1]
    e[p - 1] = 0.0

    # Generate U
    for j in range(nct, n):
        for i in range(m):
            u[i][j] = 0.0
        u[j][j] = 1.0
    for k in range(nct - 1, -1, -1):
        if s[k] != 0.0:
            for j in range(k + 1, n):
                t = 0.0
                for i in range(k, m):
                    t += u[i][k] * u[i][j]
                t = -t / u[k][k]
                for i in range(k, m):
                    u[i][j] += t * u[i][k]
            for i in range(k, m):
                u[i][k] = -u[i][k]
            u[k][k] = 1.0 + u[k][k]
            # Clears the rows above k - 1 only, as the reference routine does
            for i in range(k - 1):
                u[i][k] = 0.0
        else:
            for i in range(m):
                u[i][k] = 0.0
            u[k][k] = 1.0

    # Generate V
    for k in range(n - 1, -1, -1):
        if k < nrt and e[k] != 0.0:
            for j in range(k + 1, n):
                t = 0.0
                for i in range(k + 1, n):
                    t += v[i][k] * v[i][j]
                t = -t / v[k + 1][k]
                for i in range(k + 1, n):
                    v[i][j] += t * v[i][k]
        for i in range(n):
            v[i][k] = 0.0
        v[k][k] = 1.0

    # Main iteration loop for the singular values
    pp = p - 1
    while p > 0:
        # Inspect for negligible elements in s and e. On completion kase and k are set:
        # kase 1: s[p] and e[k - 1] are negligible and k < p
        # kase 2: s[k] is negligible and k < p
        # kase 3: e[k - 1] is negligible, k < p, and s[k], ..., s[p] are not (QR step)
        # kase 4: e[p - 1] is negligible (convergence)
        k = p - 2
        while k >= 0:
            threshold = _SVD_TINY + _SVD_EPS * (abs(s[k]) + abs(s[k + 1]))
            # Written as a negation so that a NaN breaks out of the loop too
            if not (abs(e[k]) > threshold):
                e[k] = 0.0
                break
            k -= 1
        if k == p - 2:
            kase = 4
        else:
            ks = p - 1
            while ks >= k:
                if ks == k:
                    break
                t = ((abs(e[ks]) if ks != p else 0.0)
                     + (abs(e[ks - 1]) if ks != k + 1 else 0.0))
                if abs(s[ks]) <= _SVD_TINY + _SVD_EPS * t:
                    s[ks] = 0.0
                    break
                ks -= 1
            if ks == k:
                kase = 3
            elif ks == p - 1:
                kase = 1
            else:
                kase = 2
                k = ks
        k += 1

        if kase == 1:
            # Deflate negligible s[p]
            f = e[p - 2]
            e[p - 2] = 0.0
            for j in range(p - 2, k - 1, -1):
                t = _hypot(s[j], f)
                cs = s[j] / t
                sn = f / t
                s[j] = t
                if j != k:
                    f = -sn * e[j - 1]
                    e[j - 1] = cs * e[j - 1]
                for row in v:
                    t = cs * row[j] + sn * row[p - 1]
                    row[p - 1] = -sn * row[j] + cs * row[p - 1]
                    row[j] = t
        elif kase == 2:
            # Split at negligible s[k]
            f = e[k - 1]
            e[k - 1] = 0.0
            for j in range(k, p):
                t = _hypot(s[j], f)
                cs = s[j] / t
                sn = f / t
                s[j] = t
                f = -sn * e[j]
                e[j] = cs * e[j]
                for row in u:
                    t = cs * row[j] + sn * row[k - 1]
                    row[k - 1] = -sn * row[j] + cs * row[k - 1]
                    row[j] = t
        elif kase == 3:
            # Calculate the shift
            max_pm1_pm2 = max(abs(s[p - 1]), abs(s[p - 2]))
            scale = max(max(max(max_pm1_pm2, abs(e[p - 2])), abs(s[k])), abs(e[k]))
            sp = s[p - 1] / scale
            spm1 = s[p - 2] / scale
            epm1 = e[p - 2] / scale
            sk = s[k] / scale
            ek = e[k] / scale
            b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0
            c = (sp * epm1) * (sp * epm1)
            shift = 0.0
            if b != 0.0 or c != 0.0:
                shift = math.sqrt(b * b + c)
                if b < 0.0:
                    shift = -shift
                shift = c / (b + shift)
            f = (sk + sp) * (sk - sp) + shift
            g = sk * ek
            # Chase zeros
            for j in range(k, p - 1):
                t = _hypot(f, g)
                cs = f / t
                sn = g / t
                if j != k:
                    e[j - 1] = t
                f = cs * s[j] + sn * e[j]
                e[j] = cs * e[j] - sn * s[j]
                g = sn * s[j + 1]
                s[j + 1] = cs * s[j + 1]
                for row in v:
                    t = cs * row[j] + sn * row[j + 1]
                    row[j + 1] = -sn * row[j] + cs * row[j + 1]
                    row[j] = t
                t = _hypot(f, g)
                cs = f / t
                sn = g / t
                s[j] = t
                f = cs * e[j] + sn * s[j + 1]
                s[j + 1] = -sn * e[j] + cs * s[j + 1]
                g = sn * e[j + 1]
                e[j + 1] = cs * e[j + 1]
                if j < m - 1:
                    for row in u:
                        t = cs * row[j] + sn * row[j + 1]
                        row[j + 1] = -sn * row[j] + cs * row[j + 1]
                        row[j] = t
            e[p - 2] = f
        else:
            # Convergence: make the singular value positive
            if s[k] <= 0.0:
                s[k] = -s[k] if s[k] < 0.0 else 0.0
                for i in range(pp + 1):
                    v[i][k] = -v[i][k]
            # Order the singular values
            while k < pp:
                if s[k] >= s[k + 1]:
                    break
                s[k], s[k + 1] = s[k + 1], s[k]
                if k < n - 1:
                    for row in v:
                        row[k], row[k + 1] = row[k + 1], row[k]
                if k < m - 1:
                    for row in u:
                        row[k], row[k + 1] = row[k + 1], row[k]
                k += 1
            p -= 1

    if transposed:
        return s, v, u
    return s, u, v


def _svd_pseudo_inverse(a: list[list[float]]) -> list[list[float]]:
    """
    Compute the pseudoinverse ``V S^+ U^T`` of a non-empty matrix from its singular value
    decomposition.

    :param a: The rows of the matrix; they are not modified.
    :return: The rows of the pseudoinverse.
    """
    s, u, v = _svd(a)
    # A singular value up to the tolerance counts as zero: its reciprocal is replaced by 0
    tolerance = max(max(len(a), len(a[0])) * s[0] * _SVD_EPS, _SVD_MIN_TOLERANCE)
    s_ut = _transpose(u)
    for i, value in enumerate(s):
        reciprocal = 1 / value if value > tolerance else 0.0
        row = s_ut[i]
        for j in range(len(row)):
            row[j] *= reciprocal
    return _multiply(v, s_ut)
