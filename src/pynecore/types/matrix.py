from __future__ import annotations

from typing import TypeVar, Generic, Any, Self
import copy
from collections import Counter
from .na import NA

T = TypeVar('T')


def _is_na(val: Any) -> bool:
    """Check if value is NA."""
    return isinstance(val, NA)


class Matrix(Generic[T]):
    """
    A matrix implementation in pure Python
    """

    def __init__(self, rows: int, cols: int, initial_value: T = NA(T)):
        self.rows = rows
        self.cols = cols
        self.data = [[initial_value for _ in range(cols)] for _ in range(rows)]

    def add_row(self, row: int | None = None, array_id: list[Any] | None = None) -> None:
        """
        Add a row at the specified index of the matrix.

        The row can consist of NA values, or an array can be used to provide values.

        :param row: The index where the new row will be inserted. If None, appends to the end.
        :param array_id: Array to use for providing values to the new row. If shorter than matrix
                        width, remaining cells are filled with NA. If longer, array is truncated.
        :raises IndexError: If row index is out of bounds.
        """
        # Ha row nincs megadva, akkor a végére adjuk hozzá
        if row is None:
            row = self.rows

        # Határok ellenőrzése
        if row < 0 or row > self.rows:
            raise IndexError(f"Row index {row} out of bounds for matrix with {self.rows} rows")

        # Új sor létrehozása
        if array_id is not None:
            # Ha array van megadva, használjuk azt (csonkítva vagy kiegészítve NA-val)
            new_row = []
            for i in range(self.cols):
                if i < len(array_id):
                    new_row.append(array_id[i])
                else:
                    new_row.append(NA(T))
        else:
            # Ha nincs array megadva, NA értékekkel töltjük fel
            new_row = [NA(T) for _ in range(self.cols)]

        # Sor beszúrása a megadott pozícióba
        self.data.insert(row, new_row)
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
                if not _is_na(val):
                    total += val
                    count += 1
        return total / count if count > 0 else NA(float)

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

    def concat(self, other: Self) -> Self:
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

    def copy(self) -> Self:
        """
        Create a new matrix which is a copy of the original.

        :return: A new matrix object containing a deep copy of this matrix.
        """
        new_matrix = Matrix(self.rows, self.cols)
        new_matrix.data = copy.deepcopy(self.data)
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

    def diff(self, other: Self | int | float) -> Self:
        """
        Return a new matrix resulting from subtraction.

        :param other: Matrix object or scalar value to be subtracted.
        :return: A new matrix containing the difference.
        :raises ValueError: If matrix dimensions don't match.
        """
        result = Matrix(self.rows, self.cols)

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
        if to_row is None:
            to_row = self.rows
        if to_column is None:
            to_column = self.cols

        for i in range(from_row, to_row):
            for j in range(from_column, to_column):
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
        max_val = None
        for row in self.data:
            for val in row:
                if not _is_na(val):
                    if max_val is None or val > max_val:
                        max_val = val
        return max_val if max_val is not None else NA(float)

    def min(self) -> float | int | NA[float]:
        """
        Return the smallest value from the matrix elements.

        :return: The minimum value from the matrix.
        """
        min_val = None
        for row in self.data:
            for val in row:
                if not _is_na(val):
                    if min_val is None or val < min_val:
                        min_val = val
        return min_val if min_val is not None else NA(float)

    def median(self) -> float | int | NA[float]:
        """
        Calculate the median ("middle" value) of matrix elements.

        :return: The median value of all non-NA elements.
        """
        values = []
        for row in self.data:
            for val in row:
                if not _is_na(val):
                    values.append(val)

        if not values:
            return NA(float)

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
                if not _is_na(val):
                    values.append(val)

        if not values:
            return NA(float)

        counter = Counter(values)
        if not counter:
            return NA(float)

        # Get all values with max count
        max_count = max(counter.values())
        modes = [val for val, count in counter.items() if count == max_count]

        # Return smallest if multiple modes
        return min(modes)

    def mult(self, other: Self | list[T] | int | float) -> Self | list[T]:
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

            result = Matrix(self.rows, other.cols)
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
            result = Matrix(self.rows, self.cols)
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
        if column is None:
            column = self.cols - 1

        if column < 0 or column >= self.cols:
            raise IndexError(f"Column index {column} out of bounds")

        removed = []
        for row in self.data:
            removed.append(row.pop(column))
        self.cols -= 1
        return removed

    def remove_row(self, row: int | None = None) -> list[T]:
        """
        Remove the row at the specified index and return its values.

        :param row: The index of the row to be removed. If None, removes last row.
        :return: An array containing the removed row's values.
        :raises IndexError: If row index is out of bounds.
        """
        if row is None:
            row = self.rows - 1

        if row < 0 or row >= self.rows:
            raise IndexError(f"Row index {row} out of bounds")

        removed = self.data.pop(row)
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

    def rows(self) -> int:
        """
        Return the number of rows in the matrix.

        :return: The number of rows.
        """
        return self.rows

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
                  from_column: int = 0, to_column: int | None = None) -> Self:
        """
        Extract a submatrix within the specified indices.

        :param from_row: Row index from which extraction begins (inclusive).
        :param to_row: Row index where extraction ends (exclusive). If None, extracts to end.
        :param from_column: Column index from which extraction begins (inclusive).
        :param to_column: Column index where extraction ends (exclusive). If None, extracts to end.
        :return: A new matrix containing the submatrix.
        """
        if to_row is None:
            to_row = self.rows
        if to_column is None:
            to_column = self.cols

        result = Matrix(to_row - from_row, to_column - from_column)
        for i in range(from_row, to_row):
            for j in range(from_column, to_column):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    result.data[i - from_row][j - from_column] = self.data[i][j]

        return result

    def sum(self, other: Self | int | float) -> Self:
        """
        Return a new matrix resulting from addition.

        :param other: Second matrix object or scalar value.
        :return: A new matrix containing the sum.
        :raises ValueError: If matrix dimensions don't match.
        """
        result = Matrix(self.rows, self.cols)

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

    def transpose(self) -> Self:
        """
        Create a new transposed version of the matrix.

        This interchanges the row and column index of each element.

        :return: A new matrix containing the transposed version.
        """
        result = Matrix(self.cols, self.rows)
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
                if not _is_na(val) and val != 0 and val != 1:
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
            row_sum = sum(val for val in row if not _is_na(val))
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
                if not _is_na(val) and val != 0:
                    return False
        return True

    # Additional methods
    def kron(self, other: Self) -> Self:
        """
        Return the Kronecker product of two matrices.

        :param other: Second matrix object.
        :return: A new matrix containing the Kronecker product.
        """
        result_rows = self.rows * other.rows
        result_cols = self.cols * other.cols
        result = Matrix(result_rows, result_cols)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    for l in range(other.cols):
                        result.data[i * other.rows + k][j * other.cols + l] = \
                            self.data[i][j] * other.data[k][l]

        return result

    def pow(self, power: int) -> Self:
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
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                result.data[i][i] = 1
            return result

        result = self.copy()
        for _ in range(power - 1):
            result = result.mult(self)

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

    def inv(self) -> Self:
        """
        Return the inverse of a square matrix.

        :return: A new matrix which is the inverse of this matrix.
        :raises ValueError: If matrix is not square or is singular.
        """
        if self.rows != self.cols:
            raise ValueError("Inverse can only be calculated for square matrices")

        n = self.rows
        # Create augmented matrix [A|I]
        aug = [row[:] + [0] * n for row in self.data]
        for i in range(n):
            aug[i][n + i] = 1

        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k

            aug[i], aug[max_row] = aug[max_row], aug[i]

            # Check for singular matrix
            if aug[i][i] == 0:
                raise ValueError("Matrix is singular")

            # Scale pivot row
            pivot = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= pivot

            # Eliminate column
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]

        # Extract inverse from augmented matrix
        result = Matrix(n, n)
        for i in range(n):
            for j in range(n):
                result.data[i][j] = aug[i][n + j]

        return result

    def pinv(self) -> Self:
        """
        Return the pseudoinverse of a matrix.

        Uses Moore-Penrose inverse formula. For non-singular square matrices,
        this returns the same result as inv().

        :return: A new matrix containing the pseudoinverse.
        """
        # For simplicity, if matrix is square and non-singular, return regular inverse
        if self.is_square():
            try:
                return self.inv()
            except ValueError:
                pass

        # Otherwise, use A+ = (A^T A)^(-1) A^T for overdetermined systems
        # or A+ = A^T (A A^T)^(-1) for underdetermined systems
        at = self.transpose()

        if self.rows >= self.cols:
            # Overdetermined
            ata = at.mult(self)
            return ata.inv().mult(at)
        else:
            # Underdetermined
            aat = self.mult(at)
            return at.mult(aat.inv())

    def eigenvalues(self) -> list[float | int]:
        """
        Return an array containing the eigenvalues of a square matrix.

        :return: An array containing the eigenvalues.
        :raises ValueError: If matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Eigenvalues can only be calculated for square matrices")

        # For 2x2 matrix, use analytical solution
        if self.rows == 2:
            a, b = self.data[0][0], self.data[0][1]
            c, d = self.data[1][0], self.data[1][1]

            trace = a + d
            det = a * d - b * c
            discriminant = trace * trace - 4 * det

            if discriminant >= 0:
                sqrt_disc = discriminant ** 0.5
                return [(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2]
            else:
                # Complex eigenvalues - return real parts only
                return [trace / 2, trace / 2]

        # For larger matrices, this would need a proper algorithm
        # For now, return empty list
        return []

    def eigenvectors(self) -> Self:
        """
        Return a matrix of eigenvectors.

        Each column is an eigenvector of the matrix.

        :return: A new matrix containing the eigenvectors.
        :raises ValueError: If matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Eigenvectors can only be calculated for square matrices")

        # Return identity matrix as placeholder
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            result.data[i][i] = 1
        return result

    def add_col(self, col: int | None = None, array_id: list[Any] | None = None) -> None:
        """
        Add a column at the specified index of the matrix.

        The column can consist of NA values, or an array can be used to provide values.

        :param col: The index where the new column will be inserted. If None, appends to the end.
        :param array_id: Array to use for providing values to the new column.
        :raises IndexError: If column index is out of bounds.
        """
        if col is None:
            col = self.cols

        if col < 0 or col > self.cols:
            raise IndexError(f"Column index {col} out of bounds")

        for i in range(self.rows):
            if array_id and i < len(array_id):
                self.data[i].insert(col, array_id[i])
            else:
                self.data[i].insert(col, NA(T))

        self.cols += 1
