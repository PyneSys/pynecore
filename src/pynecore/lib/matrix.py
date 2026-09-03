from __future__ import annotations
from typing import Any

from ..core.module_property import module_property
from ..types.matrix import Matrix
from ..types.na import NA, na_float, na_int
from ..types.pine_types import PyneInt
from . import array as _array

_registry: list[Matrix] = []


# noinspection PyShadowingBuiltins,PyProtectedMember
def _na_element(id: Matrix) -> Any:
    """
    Return an na value matching the matrix's element type.

    :param id: A matrix object
    :return: na of the matrix's element type, typeless na for an empty matrix
    """
    return _array._na_element(id.data[0] if id.data else [])


# noinspection PyShadowingNames
def new(rows: int = 0, columns: int = 0, initial_value: Any = None) -> Matrix:
    """
    Create a new matrix object.

    A matrix is a two-dimensional data structure containing rows and columns.
    All elements in the matrix must be of the same type.

    :param rows: Initial row count of the matrix. Optional, defaults to 0.
    :param columns: Initial column count of the matrix. Optional, defaults to 0.
    :param initial_value: Initial value of all matrix elements. If None, uses NA.
    :return: The ID of the new matrix object.
    """
    if initial_value is None:
        initial_value = na_float

    matrix_obj = Matrix(int(rows), int(columns), initial_value)
    _registry.append(matrix_obj)
    return matrix_obj


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Matrix]:
    """
    Return all matrix objects.

    :return: List of all created matrix objects.
    """
    return _registry


# noinspection PyShadowingBuiltins
def copy(id: Matrix | NA) -> Matrix | NA:
    """
    Create a new matrix which is a copy of the original.

    :param id: A matrix object to copy.
    :return: A new matrix object of the copied matrix.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.copy()


# noinspection PyShadowingBuiltins
def delete(id: Matrix | NA) -> None:
    """
    Delete matrix object.

    :param id: Matrix object to delete.
    """
    if isinstance(id, NA):
        return
    try:
        _registry.remove(id)
    except ValueError:
        pass


# noinspection PyShadowingBuiltins,PyShadowingNames
def add_row(id: Matrix | NA, row: int | None = None, array_id: list[Any] | None = None) -> None:
    """
    Add a row at the specified index of the matrix.

    :param id: A matrix object.
    :param row: The index where the new row will be inserted. If None, appends to the end.
    :param array_id: Array to use for providing values to the new row.
    """
    if isinstance(id, NA):
        return
    id.add_row(row if row is None else int(row), array_id)


# noinspection PyShadowingBuiltins
def add_col(id: Matrix | NA, column: int | None = None, array_id: list[Any] | None = None) -> None:
    """
    Add a column at the specified index of the matrix.

    :param id: A matrix object.
    :param column: The index where the new column will be inserted. If None, appends to the end.
    :param array_id: Array to use for providing values to the new column.
    """
    if isinstance(id, NA):
        return
    id.add_col(column if column is None else int(column), array_id)


# noinspection PyShadowingBuiltins
def avg(id: Matrix | NA) -> float | int | NA:
    """
    Calculate the average of all elements in the matrix.

    :param id: A matrix object.
    :return: The average value from the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.avg()


# noinspection PyShadowingBuiltins
def col(id: Matrix | NA, column: int) -> list[Any] | NA:
    """
    Create a one-dimensional array from the elements of a matrix column.

    An na index yields an na array, exactly like an na matrix does.

    :param id: A matrix object.
    :param column: Index of the required column.
    :return: An array containing the column values, or na if the index is na.
    """
    if isinstance(id, NA):
        return NA(list)
    # Measured on TradingView: an na index hands back an na array; see ``row``.
    if not (column == column):  # is_na_arg
        return NA(list)
    return id.col(int(column))


# noinspection PyShadowingBuiltins
def columns(id: Matrix | NA) -> PyneInt:
    """
    Return the number of columns in the matrix.

    :param id: A matrix object.
    :return: The number of columns.
    """
    if isinstance(id, NA):
        return na_int
    # A Pine int is a double at runtime
    return float(id.cols)


# noinspection PyShadowingBuiltins
def concat(id1: Matrix | NA, id2: Matrix | NA) -> Matrix | NA:
    """
    Append the second matrix to the first matrix.

    :param id1: Matrix object to concatenate into.
    :param id2: Matrix object whose elements will be appended.
    :return: The first matrix after concatenation.
    """
    if isinstance(id1, NA) or isinstance(id2, NA):
        return NA(Matrix)
    return id1.concat(id2)


# noinspection PyShadowingBuiltins
def det(id: Matrix | NA) -> float | int | NA:
    """
    Return the determinant of a square matrix.

    :param id: A matrix object.
    :return: The determinant value of the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.det()


# noinspection PyShadowingBuiltins
def diff(id1: Matrix | NA, id2: Matrix | int | float | NA) -> Matrix | NA:
    """
    Return a new matrix resulting from subtraction.

    :param id1: Matrix to subtract from.
    :param id2: Matrix object or scalar value to be subtracted.
    :return: A new matrix containing the difference.
    """
    if isinstance(id1, NA) or isinstance(id2, NA):
        return NA(Matrix)
    return id1.diff(id2)


# noinspection PyShadowingBuiltins
def eigenvalues(id: Matrix | NA) -> list[float | int] | NA:
    """
    Return an array containing the eigenvalues of a square matrix.

    :param id: A matrix object.
    :return: An array containing the eigenvalues.
    """
    if isinstance(id, NA):
        return NA(list)
    return id.eigenvalues()


# noinspection PyShadowingBuiltins
def eigenvectors(id: Matrix | NA) -> Matrix | NA:
    """
    Return a matrix of eigenvectors.

    :param id: A matrix object.
    :return: A new matrix containing the eigenvectors.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.eigenvectors()


# noinspection PyShadowingBuiltins
def elements_count(id: Matrix | NA) -> PyneInt:
    """
    Return the total number of all matrix elements.

    :param id: A matrix object.
    :return: The total number of elements.
    """
    if isinstance(id, NA):
        return na_int
    # A Pine int is a double at runtime
    return float(id.elements_count())


# noinspection PyShadowingBuiltins
def fill(id: Matrix | NA, value: Any, from_row: int = 0, to_row: int | None = None,
         from_column: int = 0, to_column: int | None = None) -> None:
    """
    Fill a rectangular area of the matrix with the specified value.

    :param id: A matrix object.
    :param value: The value to fill with.
    :param from_row: Row index from which the fill will begin (inclusive).
    :param to_row: Row index where the fill will end (exclusive).
    :param from_column: Column index from which the fill will begin (inclusive).
    :param to_column: Column index where the fill will end (exclusive).
    """
    if isinstance(id, NA):
        return
    id.fill(value, int(from_row), to_row if to_row is None else int(to_row),
            int(from_column), to_column if to_column is None else int(to_column))


# noinspection PyShadowingBuiltins,PyShadowingNames
def get(id: Matrix | NA, row: int, column: int) -> Any | NA:
    """
    Return the element with the specified index of the matrix.

    An na row or column index yields na and leaves the matrix untouched.

    :param id: A matrix object.
    :param row: Index of the required row.
    :param column: Index of the required column.
    :return: The value at the specified position, or na if either index is na.
    """
    if isinstance(id, NA):
        return NA(object)
    # Measured on TradingView (FX:EURUSD 240, bar 100, 2x2 float matrix):
    # matrix.get(m, na, 0), (m, 0, na) and (m, na, na) all return NaN. Without
    # this the na index reached list indexing and halted the script: an na never
    # compares true against the bounds check, so it slipped straight through.
    if not (row == row) or not (column == column):  # is_na_arg
        return _na_element(id)
    # Every index here is truncated to an integer, so a float that carries one
    # addresses the matrix instead of raising a TypeError; see ``array.get``.
    return id.get(int(row), int(column))


# noinspection PyShadowingBuiltins
def inv(id: Matrix | NA) -> Matrix | NA:
    """
    Return the inverse of a square matrix.

    :param id: A matrix object.
    :return: A new matrix which is the inverse.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.inv()


# noinspection PyShadowingBuiltins
def is_antidiagonal(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is anti-diagonal.

    :param id: Matrix object to test.
    :return: True if the matrix is anti-diagonal, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_antidiagonal()


# noinspection PyShadowingBuiltins
def is_antisymmetric(id: Matrix | NA) -> bool | NA:
    """
    Determine if a matrix is antisymmetric.

    :param id: Matrix object to test.
    :return: True if the matrix is antisymmetric, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_antisymmetric()


# noinspection PyShadowingBuiltins
def is_binary(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is binary.

    :param id: Matrix object to test.
    :return: True if the matrix is binary, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_binary()


# noinspection PyShadowingBuiltins
def is_diagonal(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is diagonal.

    :param id: Matrix object to test.
    :return: True if the matrix is diagonal, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_diagonal()


# noinspection PyShadowingBuiltins
def is_identity(id: Matrix | NA) -> bool | NA:
    """
    Determine if a matrix is an identity matrix.

    :param id: Matrix object to test.
    :return: True if the matrix is identity, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_identity()


# noinspection PyShadowingBuiltins
def is_square(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is square.

    :param id: Matrix object to test.
    :return: True if the matrix is square, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_square()


# noinspection PyShadowingBuiltins
def is_stochastic(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is stochastic.

    :param id: Matrix object to test.
    :return: True if the matrix is stochastic, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_stochastic()


# noinspection PyShadowingBuiltins
def is_symmetric(id: Matrix | NA) -> bool | NA:
    """
    Determine if a square matrix is symmetric.

    :param id: Matrix object to test.
    :return: True if the matrix is symmetric, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_symmetric()


# noinspection PyShadowingBuiltins
def is_triangular(id: Matrix | NA) -> bool | NA:
    """
    Determine if the matrix is triangular.

    :param id: Matrix object to test.
    :return: True if the matrix is triangular, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_triangular()


# noinspection PyShadowingBuiltins
def is_zero(id: Matrix | NA) -> bool | NA:
    """
    Determine if all elements of the matrix are zero.

    :param id: Matrix object to check.
    :return: True if all elements are zero, False otherwise.
    """
    if isinstance(id, NA):
        return NA(bool)
    return id.is_zero()


# noinspection PyShadowingBuiltins
def kron(id1: Matrix | NA, id2: Matrix | NA) -> Matrix | NA:
    """
    Return the Kronecker product for two matrices.

    :param id1: First matrix object.
    :param id2: Second matrix object.
    :return: A new matrix containing the Kronecker product.
    """
    if isinstance(id1, NA) or isinstance(id2, NA):
        return NA(Matrix)
    return id1.kron(id2)


# noinspection PyShadowingBuiltins
def max(id: Matrix | NA) -> float | int | NA:
    """
    Return the largest value from the matrix elements.

    :param id: A matrix object.
    :return: The maximum value from the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.max()


# noinspection PyShadowingBuiltins
def median(id: Matrix | NA) -> float | int | NA:
    """
    Calculate the median of matrix elements.

    :param id: A matrix object.
    :return: The median value from the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.median()


# noinspection PyShadowingBuiltins
def min(id: Matrix | NA) -> float | int | NA:
    """
    Return the smallest value from the matrix elements.

    :param id: A matrix object.
    :return: The minimum value from the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.min()


# noinspection PyShadowingBuiltins
def mode(id: Matrix | NA) -> float | int | NA:
    """
    Calculate the mode of the matrix.

    :param id: A matrix object.
    :return: The most frequently occurring value from the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.mode()


# noinspection PyShadowingBuiltins
def mult(id1: Matrix | NA, id2: Matrix | list[Any] | int | float | NA) -> Matrix | list[Any] | NA:
    """
    Return the product of matrices, matrix and vector, or matrix and scalar.

    :param id1: First matrix object.
    :param id2: Second matrix object, array, or scalar value.
    :return: A new matrix or array containing the product.
    """
    if isinstance(id1, NA) or isinstance(id2, NA):
        return NA(Matrix)
    return id1.mult(id2)


# noinspection PyShadowingBuiltins
def pinv(id: Matrix | NA) -> Matrix | NA:
    """
    Return the pseudoinverse of a matrix.

    :param id: A matrix object.
    :return: A new matrix containing the pseudoinverse.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.pinv()


# noinspection PyShadowingBuiltins
def pow(id: Matrix | NA, power: int) -> Matrix | NA:
    """
    Calculate the product of the matrix by itself power times.

    :param id: A matrix object.
    :param power: The number of times the matrix will be multiplied by itself.
    :return: The matrix raised to the specified power.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.pow(int(power))


# noinspection PyShadowingBuiltins
def rank(id: Matrix | NA) -> PyneInt:
    """
    Calculate the rank of the matrix.

    :param id: A matrix object.
    :return: The rank of the matrix.
    """
    if isinstance(id, NA):
        return na_int
    # A Pine int is a double at runtime
    return float(id.rank())


# noinspection PyShadowingBuiltins
def remove_col(id: Matrix | NA, column: int | None = None) -> list[Any] | NA:
    """
    Remove the column at the specified index and return its values.

    :param id: A matrix object.
    :param column: The index of the column to be removed.
    :return: An array containing the removed column's values.
    """
    if isinstance(id, NA):
        return NA(list)
    return id.remove_col(column if column is None else int(column))


# noinspection PyShadowingBuiltins,PyShadowingNames
def remove_row(id: Matrix | NA, row: int | None = None) -> list[Any] | NA:
    """
    Remove the row at the specified index and return its values.

    :param id: A matrix object.
    :param row: The index of the row to be removed.
    :return: An array containing the removed row's values.
    """
    if isinstance(id, NA):
        return NA(list)
    return id.remove_row(row if row is None else int(row))


# noinspection PyShadowingBuiltins,PyShadowingNames
def reshape(id: Matrix | NA, rows: int, columns: int) -> None:
    """
    Rebuild the matrix to the specified dimensions.

    :param id: A matrix object.
    :param rows: The number of rows of the reshaped matrix.
    :param columns: The number of columns of the reshaped matrix.
    """
    if isinstance(id, NA):
        return
    id.reshape(int(rows), int(columns))


# noinspection PyShadowingBuiltins
def reverse(id: Matrix | NA) -> None:
    """
    Reverse the order of rows and columns in the matrix.

    :param id: A matrix object.
    """
    if isinstance(id, NA):
        return
    id.reverse()


# noinspection PyShadowingBuiltins,PyShadowingNames
def row(id: Matrix | NA, row: int) -> list[Any] | NA:
    """
    Create a one-dimensional array from the elements of a matrix row.

    An na index yields an na array, exactly like an na matrix does.

    :param id: A matrix object.
    :param row: Index of the required row.
    :return: An array containing the row values, or na if the index is na.
    """
    if isinstance(id, NA):
        return NA(list)
    # Measured on TradingView: matrix.row(m, na) hands back an na array -- the
    # error only comes later, from the array function called on it (RE10052).
    if not (row == row):  # is_na_arg
        return NA(list)
    return id.row(int(row))


# noinspection PyShadowingBuiltins
def rows(id: Matrix | NA) -> PyneInt:
    """
    Return the number of rows in the matrix.

    :param id: A matrix object.
    :return: The number of rows.
    """
    if isinstance(id, NA):
        return na_int
    # A Pine int is a double at runtime
    return float(id.rows)


# noinspection PyShadowingBuiltins,PyShadowingNames
def set(id: Matrix | NA, row: int, column: int, value: Any) -> None:
    """
    Assign value to the element at the specified row and column.

    An na row or column index does nothing at all.

    :param id: A matrix object.
    :param row: The row index of the element to be modified.
    :param column: The column index of the element to be modified.
    :param value: The new value to be set.
    """
    if isinstance(id, NA):
        return
    # Measured on TradingView: matrix.set(m, na, 0, 5.0) leaves both the contents
    # and the size of a 2x2 matrix untouched -- it is a no-op, not an error.
    if not (row == row) or not (column == column):  # is_na_arg
        return
    id.set(int(row), int(column), value)


# noinspection PyShadowingBuiltins
def sort(id: Matrix | NA, column: int = 0, order: str = 'ascending') -> None:
    """
    Rearrange rows following the sorted order of values in the specified column.

    :param id: A matrix object to be sorted.
    :param column: Index of the column whose sorted values determine the new order of rows.
    :param order: The sort order ('ascending' or 'descending').
    """
    if isinstance(id, NA):
        return
    id.sort(int(column), order)


# noinspection PyShadowingBuiltins
def submatrix(id: Matrix | NA, from_row: int = 0, to_row: int | None = None,
              from_column: int = 0, to_column: int | None = None) -> Matrix | NA:
    """
    Extract a submatrix within the specified indices.

    :param id: A matrix object.
    :param from_row: Row index from which extraction begins (inclusive).
    :param to_row: Row index where extraction ends (exclusive).
    :param from_column: Column index from which extraction begins (inclusive).
    :param to_column: Column index where extraction ends (exclusive).
    :return: A new matrix containing the submatrix.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.submatrix(int(from_row), to_row if to_row is None else int(to_row),
                        int(from_column), to_column if to_column is None else int(to_column))


# noinspection PyShadowingBuiltins
def sum(id1: Matrix | NA, id2: Matrix | int | float | NA) -> Matrix | NA:
    """
    Return a new matrix resulting from addition.

    :param id1: First matrix object.
    :param id2: Second matrix object or scalar value.
    :return: A new matrix containing the sum.
    """
    if isinstance(id1, NA) or isinstance(id2, NA):
        return NA(Matrix)
    return id1.sum(id2)


# noinspection PyShadowingBuiltins
def swap_columns(id: Matrix | NA, column1: int, column2: int) -> None:
    """
    Swap the columns at the specified indices.

    :param id: A matrix object.
    :param column1: Index of the first column to be swapped.
    :param column2: Index of the second column to be swapped.
    """
    if isinstance(id, NA):
        return
    id.swap_columns(int(column1), int(column2))


# noinspection PyShadowingBuiltins
def swap_rows(id: Matrix | NA, row1: int, row2: int) -> None:
    """
    Swap the rows at the specified indices.

    :param id: A matrix object.
    :param row1: Index of the first row to be swapped.
    :param row2: Index of the second row to be swapped.
    """
    if isinstance(id, NA):
        return
    id.swap_rows(int(row1), int(row2))


# noinspection PyShadowingBuiltins
def trace(id: Matrix | NA) -> float | int | NA:
    """
    Calculate the trace of a matrix.

    :param id: A matrix object.
    :return: The trace value of the matrix.
    """
    if isinstance(id, NA):
        return na_float
    return id.trace()


# noinspection PyShadowingBuiltins
def transpose(id: Matrix | NA) -> Matrix | NA:
    """
    Create a new transposed version of the matrix.

    :param id: A matrix object.
    :return: A new matrix containing the transposed version.
    """
    if isinstance(id, NA):
        return NA(Matrix)
    return id.transpose()
