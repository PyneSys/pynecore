"""
@pyne

TradingView measurements (CAPITALCOM:EURUSD 60): a supplied row/column array
must match the opposite dimension, including an empty supplied array and a
matrix that has rows but no columns (or columns but no rows). Only a 0x0 matrix
takes its first dimension from the array; omitting the array inserts na cells.
"""
import pytest

from pynecore.lib import matrix
from pynecore.types.line import Line
from pynecore.types.na import NA


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_helper_is_float_na(value) -> bool:
    """Whether a cell is the float na (NaN), not an na object."""
    return isinstance(value, float) and value != value


@pytest.mark.parametrize("size", [0, 1, 2, 4])
@pytest.mark.parametrize("rows", [0, 2])
def __test_wrong_row_size_is_rejected_before_mutation__(size, rows):
    """A supplied row must match the column count, even when there are no rows."""
    m = matrix.new(rows, 3, 0.0)
    before = [row.copy() for row in m.data]
    values = [float(i + 1) for i in range(size)]
    with pytest.raises(ValueError, match="does not match the matrix column count"):
        matrix.add_row(m, rows, values)
    assert matrix.rows(m) == rows and matrix.columns(m) == 3
    assert m.data == before
    assert values == [float(i + 1) for i in range(size)]


@pytest.mark.parametrize("size", [0, 1, 3, 4])
@pytest.mark.parametrize("cols", [0, 3])
def __test_wrong_column_size_is_rejected_before_mutation__(size, cols):
    """A supplied column must match the row count, even when there are no columns."""
    m = matrix.new(2, cols, 0.0)
    before = [row.copy() for row in m.data]
    values = [float(i + 1) for i in range(size)]
    with pytest.raises(ValueError, match="does not match the matrix row count"):
        matrix.add_col(m, cols, values)
    assert matrix.rows(m) == 2 and matrix.columns(m) == cols
    assert m.data == before
    assert values == [float(i + 1) for i in range(size)]


@pytest.mark.parametrize("index", [0, 1, 2])
def __test_matching_row_is_inserted_as_a_copy__(index):
    """A matching row is copied at the requested index without changing other rows."""
    m = matrix.new(2, 3, 0.0)
    values = [1.0, 2.0, 3.0]
    matrix.add_row(m, index, values)
    expected = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    expected.insert(index, values.copy())
    assert m.data == expected
    assert matrix.rows(m) == 3 and matrix.columns(m) == 3
    values[0] = 9.0
    assert matrix.get(m, index, 0) == 1.0


@pytest.mark.parametrize("index", [0, 1, 3])
def __test_matching_column_is_inserted__(index):
    """A matching column is copied at the requested index without changing other cells."""
    m = matrix.new(2, 3, 0.0)
    values = [1.0, 2.0]
    matrix.add_col(m, index, values)
    expected = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for row, value in zip(expected, values):
        row.insert(index, value)
    assert m.data == expected
    assert matrix.rows(m) == 2 and matrix.columns(m) == 4
    values[0] = 9.0
    assert matrix.get(m, 0, index) == 1.0


@pytest.mark.parametrize("size", [0, 3])
def __test_empty_matrix_takes_dimensions_from_array__(size):
    """The first row or column of a 0x0 matrix can take an array of any size."""
    values = [float(i + 1) for i in range(size)]
    m = matrix.new()
    matrix.add_row(m, 0, values)
    assert matrix.rows(m) == 1 and matrix.columns(m) == size
    assert m.data == [values]
    m = matrix.new()
    matrix.add_col(m, 0, values)
    assert matrix.rows(m) == size and matrix.columns(m) == 1
    assert m.data == [[value] for value in values]


def __test_matching_array_with_only_opposite_dimension__():
    """An existing opposite dimension also accepts an exactly matching array."""
    m = matrix.new(0, 3, 0.0)
    matrix.add_row(m, 0, [1.0, 2.0, 3.0])
    assert matrix.rows(m) == 1 and matrix.columns(m) == 3
    assert m.data == [[1.0, 2.0, 3.0]]
    m = matrix.new(2, 0, 0.0)
    matrix.add_col(m, 0, [1.0, 2.0])
    assert matrix.rows(m) == 2 and matrix.columns(m) == 1
    assert m.data == [[1.0], [2.0]]


def __test_one_empty_dimension_does_not_take_size_from_array__():
    """A 3x0 or 0x3 matrix rejects a non-empty array and accepts an empty one."""
    m = matrix.new(3, 0, 0.0)
    with pytest.raises(ValueError, match="does not match the matrix column count"):
        matrix.add_row(m, 3, [1.0, 2.0])
    assert matrix.rows(m) == 3 and matrix.columns(m) == 0
    assert m.data == [[], [], []]
    matrix.add_row(m, 3, [])
    assert matrix.rows(m) == 4 and matrix.columns(m) == 0
    assert m.data == [[], [], [], []]
    m = matrix.new(0, 3, 0.0)
    with pytest.raises(ValueError, match="does not match the matrix row count"):
        matrix.add_col(m, 3, [1.0, 2.0])
    assert matrix.rows(m) == 0 and matrix.columns(m) == 3
    assert m.data == []
    matrix.add_col(m, 3, [])
    assert matrix.rows(m) == 0 and matrix.columns(m) == 4
    assert m.data == []


def __test_omitted_array_inserts_na_cells__():
    """Omitting the array inserts na cells rather than triggering the size check."""
    m = matrix.new(2, 3, 0.0)
    matrix.add_row(m)
    assert matrix.rows(m) == 3 and matrix.columns(m) == 3
    assert all(__test_helper_is_float_na(value) for value in m.data[2])
    matrix.add_col(m)
    assert matrix.rows(m) == 3 and matrix.columns(m) == 4
    assert all(__test_helper_is_float_na(row[3]) for row in m.data)
    empty_row = matrix.new()
    matrix.add_row(empty_row)
    assert matrix.rows(empty_row) == 1 and matrix.columns(empty_row) == 0
    empty_col = matrix.new()
    matrix.add_col(empty_col)
    assert matrix.rows(empty_col) == 0 and matrix.columns(empty_col) == 1
    rows_only = matrix.new(3, 0, 0.0)
    matrix.add_row(rows_only)
    assert matrix.rows(rows_only) == 4 and matrix.columns(rows_only) == 0
    cols_only = matrix.new(0, 3, 0.0)
    matrix.add_col(cols_only)
    assert matrix.rows(cols_only) == 0 and matrix.columns(cols_only) == 4


def __test_added_cells_are_the_na_of_the_element_type__():
    """A matrix created without cells pads with the na of its initial value's type."""
    floats = matrix.new(0, 2, NA(float))
    matrix.add_row(floats)
    matrix.add_col(floats)
    assert all(__test_helper_is_float_na(value) for value in floats.data[0])
    # The padded cells compare like the cells the matrix was created with
    seeded = matrix.new(1, 2, NA(float))
    matrix.add_row(seeded)
    assert [value != 0.0 for value in seeded.data[1]] == [value != 0.0 for value in seeded.data[0]]
    texts = matrix.new(0, 2, '')
    matrix.add_row(texts)
    assert texts.data == [['', '']]
    lines = matrix.new(0, 2, NA(Line))
    matrix.add_row(lines)
    assert all(value is NA(Line) for value in lines.data[0])


def __test_derived_matrix_keeps_the_na_element__():
    """A copy, transpose or submatrix still pads with its source's na."""
    source = matrix.new(0, 2, NA(float))
    copied = matrix.copy(source)
    matrix.add_row(copied)
    assert len(copied.data[0]) == 2
    assert all(__test_helper_is_float_na(value) for value in copied.data[0])
    transposed = matrix.transpose(source)
    matrix.add_col(transposed)
    assert len(transposed.data) == 2
    assert all(__test_helper_is_float_na(row[0]) for row in transposed.data)
    texts = matrix.new(2, 2, 'x')
    part = matrix.submatrix(texts, 0, 1, 0, 1)
    matrix.add_col(part)
    assert part.data == [['x', '']]


def __test_na_index_on_a_matrix_without_cells_is_its_element_na__():
    """An na index reads the matrix's element na even when there is no cell to sample."""
    assert __test_helper_is_float_na(matrix.get(matrix.new(0, 2, NA(float)), NA(int), 0))
    assert matrix.get(matrix.new(0, 2, ''), NA(int), 0) == ''


def __test_zeroth_power_is_the_identity__():
    """The zeroth power has zeros off the diagonal, not na cells (TV-measured: 1.0 / 0.0)."""
    m = matrix.new(2, 2, 3.0)
    assert matrix.pow(m, 0).data == [[1, 0], [0, 1]]
