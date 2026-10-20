"""
@pyne

TradingView measurements (CAPITALCOM:EURUSD 60): matrix.eigenvalues() and
matrix.eigenvectors() of 28 float matrices and one int matrix, printed with
str.tostring(x, "#.####################"). Every expected value below is a
TV-measured double; a value under 1e-3 was printed rounded to 20 decimals.
"""
import pytest

from pynecore.lib import matrix
from pynecore.types.matrix import Matrix


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_helper_matrix(rows: list[list[float]]) -> Matrix:
    """Build a float matrix from its rows through the matrix API."""
    m = matrix.new(len(rows), len(rows[0]), 0.0)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            matrix.set(m, i, j, value)
    return m


def __test_helper_assert_tv(actual: float, expected: float) -> None:
    """A value at or above 1e-3 matches to the bit; a smaller one to TV's 20 printed decimals."""
    if abs(expected) >= 1e-3 or expected == 0.0:
        assert actual == expected
    else:
        assert abs(actual - expected) <= 5e-21


__test_helper_cases = [
    pytest.param(
        [[2.0, 1.0], [1.0, 2.0]],
        [3.0, 1.0000000000000002],
        [
            [0.7071067811865475, 0.7071067811865475],
            [0.7071067811865475, -0.7071067811865475],
        ],
        id='symmetric_2x2',
    ),
    pytest.param(
        [[1.0, 2.0], [2.0, 1.0]],
        [2.9999999999999996, -0.9999999999999996],
        [
            [0.7071067811865476, 0.7071067811865474],
            [0.7071067811865474, -0.7071067811865476],
        ],
        id='symmetric_2x2_negative',
    ),
    pytest.param(
        [[5.0, 0.0], [0.0, 1.0]],
        [5.0, 1.0],
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        id='diagonal',
    ),
    pytest.param(
        [[1.0, 0.0], [0.0, 5.0]],
        [5.0, 1.0],
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        id='diagonal_sorted_descending',
    ),
    pytest.param(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        id='identity',
    ),
    pytest.param(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
        [3.414213562373095, 2.0, 0.5857864376269051],
        [
            [0.4999999999999997, 0.7071067811865476, -0.5000000000000002],
            [-0.7071067811865472, -0.00000000000000033307, -0.7071067811865474],
            [0.5000000000000002, -0.7071067811865472, -0.49999999999999983],
        ],
        id='symmetric_3x3_tridiagonal',
    ),
    pytest.param(
        [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
        [6.669079088282288, 3.476023602918134, 1.854897308799577],
        [
            [-0.631178968776483, 0.37436195478307144, -0.6793130619863368],
            [-0.1720265367929081, 0.7864356987513781, 0.5932333119173848],
            [-0.7563200248659911, -0.49129626351156847, 0.4319814827585529],
        ],
        id='symmetric_3x3',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [2.0, -4.0, 1.0], [3.0, 1.0, 0.0]],
        [4.125236535535355, -2.4228977576819086, -4.702338777853447],
        [
            [0.7500425448601045, 0.5683327601343983, -0.3382810291130651],
            [0.25949313727085915, 0.2175880092731859, 0.9409137951639671],
            [0.6083580299536654, -0.7935069829425249, 0.015721813265222873],
        ],
        id='symmetric_3x3_mixed_signs',
    ),
    pytest.param(
        [[4.0, 1.0, -2.0, 2.0], [1.0, 2.0, 0.0, 1.0], [-2.0, 0.0, 3.0, -2.0],
         [2.0, 1.0, -2.0, -1.0]],
        [6.844621107234967, 2.268531406431243, 1.084364463773217, -2.197516977439425],
        [
            [0.7180459594506795, 0.2017110966209424, -0.6422600067660327, 0.17670517062447633],
            [0.2211529881551806, 0.7894499124874393, 0.5441878481946862, 0.17810046990630102],
            [-0.5573513771375095, 0.5796341693167877, -0.5202218509386148, -0.2876680343197011],
            [0.35335647489399286, 0.010280998686330622, 0.14398227457619137, -0.9242849167461129],
        ],
        id='symmetric_4x4',
    ),
    pytest.param(
        [[4.0, 1.0], [2.0, 3.0]],
        [4.999999999999999, 1.9999999999999996],
        [
            [0.7071067811865475, -0.4714045207910316],
            [0.7071067811865475, 0.9428090415820634],
        ],
        id='non_symmetric_2x2',
    ),
    pytest.param(
        [[4.0, -2.0], [1.0, 1.0]],
        [2.9999999999999996, 1.9999999999999998],
        [
            [0.8944271909999159, 2.2360679774997907],
            [0.4472135954999579, 2.2360679774997902],
        ],
        id='non_symmetric_2x2_b',
    ),
    pytest.param(
        [[1.0, 2.0], [3.0, 4.0]],
        [-0.37228132326901425, 5.372281323269014],
        [
            [-0.8245648401323937, -0.42222915041526043],
            [0.5657674649689923, -0.9230523142501933],
        ],
        id='non_symmetric_2x2_unsorted',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
        [1.0, 4.0, 6.0],
        [
            [1.0, 0.6666666666666666, 1.6],
            [0.0, 1.0, 2.5],
            [0.0, 0.0, 1.0],
        ],
        id='upper_triangular_3x3',
    ),
    pytest.param(
        [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 5.0, 6.0]],
        [1.9999999999999991, 5.999999999999999, 2.9999999999999973],
        [
            [-0.6963106238227899, -0.00000000000000015266, -0.0000000000000019984],
            [0.6963106238227921, -0.00000000000000006939, -1.4142135623730958],
            [-0.17407765595569955, 1.015504800579495, 2.357022603955157],
        ],
        id='lower_triangular_3x3',
    ),
    pytest.param(
        [[1.0, 2.0, 1.0], [6.0, -1.0, 0.0], [-1.0, -2.0, -1.0]],
        [3.000000000000006, -3.9999999999999964, -0.00000000000000040685],
        [
            [-0.48507125007266616, -0.41649656391752155, 0.11785113019775706],
            [-0.7276068751089997, 0.8329931278350429, 0.7071067811865469],
            [0.48507125007266616, 0.41649656391752155, -1.5320646925708523],
        ],
        id='non_symmetric_3x3_singular',
    ),
    pytest.param(
        [[6.0, -2.0, 1.0], [3.0, 1.0, 0.0], [1.0, 1.0, 4.0]],
        [5.690647448028615, 2.6546762759856946, 2.6546762759856946],
        [
            [0.6523918785316141, -1.2032400944858939, -0.5006064382162043],
            [0.41725063699200093, -1.7765063000831192, 0.508200383091392],
            [0.6326821814748398, 1.1323781497041652, 1.1043464254892768],
        ],
        id='non_symmetric_3x3_complex_pair',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [2.0, 6.0, 4.0, 8.0], [3.0, 1.0, 1.0, 2.0]],
        [15.02817090290765, 1.1539652364168786, -1.5910680696622737, -1.5910680696622737],
        [
            [0.27772374993327253, -0.21126571795489266, -0.453027422266388, 0.3989947101175752],
            [0.7490473954931646, 0.862056113646549, 0.20071078832857586, 0.6483411211597028],
            [0.5781862774757784, 0.3920543018935354, -0.6127913655844275, -0.8467631512668938],
            [0.16582565111166858, -0.733200677296353, 0.525724204918107, -0.09171852505609115],
        ],
        id='non_symmetric_4x4_complex_pair',
    ),
    pytest.param(
        [[0.0, -1.0], [1.0, 0.0]],
        [0.0, 0.0],
        [
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        id='rotation_2x2',
    ),
    pytest.param(
        [[1.0, -2.0], [3.0, 4.0]],
        [2.5, 2.5],
        [
            [-0.6454972243679028, -0.5],
            [0.0, 1.0],
        ],
        id='complex_pair_2x2',
    ),
    pytest.param(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        [0.0, 0.0, 2.0],
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        id='rotation_block_3x3',
    ),
    pytest.param(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [-0.5000000000000002, -0.5000000000000002, 1.0000000000000002],
        [
            [0.06711560552140218, 0.8137334712067351, -0.5773502691896257],
            [-0.7382716607354269, -0.3487429162314583, -0.5773502691896258],
            [0.6711560552140247, -0.46499055497527714, -0.5773502691896258],
        ],
        id='cyclic_permutation_3x3',
    ),
    pytest.param(
        [[1.0, 2.0], [2.0, 4.0]],
        [5.0, 0.0],
        [
            [-0.4472135954999579, 0.8944271909999159],
            [-0.8944271909999159, -0.4472135954999579],
        ],
        id='singular_symmetric_2x2',
    ),
    pytest.param(
        [[1.0, 2.0], [3.0, 6.0]],
        [0.0, 7.0],
        [
            [-0.8944271909999159, -0.31943828249996997],
            [0.4472135954999579, -0.9583148474999098],
        ],
        id='singular_non_symmetric_2x2',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [16.116843969807043, -1.1168439698070438, -0.00000000000000092965],
        [
            [0.23197068724628567, 0.8169642040610363, 0.4082482904638629],
            [0.5253220933012341, 0.09018835790853769, -0.8164965809277261],
            [0.8186734993561818, -0.6365874882439638, 0.40824829046386285],
        ],
        id='singular_non_symmetric_3x3',
    ),
    pytest.param(
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        id='zero_2x2',
    ),
    pytest.param(
        [[2.0, 1.0], [0.0, 2.0]],
        [2.0, 2.0],
        [
            [1.0, -1.0],
            [0.0, 0.00000000000000055511],
        ],
        id='defective_jordan_block',
    ),
    pytest.param(
        [[5.0]],
        [5.0],
        [
            [1.0],
        ],
        id='one_by_one',
    ),
    pytest.param(
        [[-3.0]],
        [-3.0],
        [
            [1.0],
        ],
        id='one_by_one_negative',
    ),
]


@pytest.mark.parametrize("rows, values, vectors", __test_helper_cases)
def __test_eigenvalues_match_tv__(rows, values, vectors):
    """TV-measured: the eigenvalues, in TV's order, match to the last bit."""
    m = __test_helper_matrix(rows)
    result = matrix.eigenvalues(m)
    assert len(result) == len(values)
    for actual, expected in zip(result, values):
        __test_helper_assert_tv(actual, expected)


@pytest.mark.parametrize("rows, values, vectors", __test_helper_cases)
def __test_eigenvectors_match_tv__(rows, values, vectors):
    """TV-measured: the eigenvector columns, with TV's scale and sign, match to the last bit."""
    m = __test_helper_matrix(rows)
    result = matrix.eigenvectors(m)
    assert matrix.rows(result) == len(vectors) and matrix.columns(result) == len(vectors)
    for i, row in enumerate(vectors):
        for j, expected in enumerate(row):
            __test_helper_assert_tv(matrix.get(result, i, j), expected)


@pytest.mark.parametrize("rows, values, vectors", __test_helper_cases)
def __test_input_is_not_modified__(rows, values, vectors):
    """Neither function changes the matrix it decomposes."""
    m = __test_helper_matrix(rows)
    matrix.eigenvalues(m)
    matrix.eigenvectors(m)
    assert m.data == rows


def __test_int_matrix_keeps_fractional_values__():
    """TV-measured: an int matrix gives the same fractional values as the same float matrix."""
    m = matrix.new(2, 2, 0)
    matrix.set(m, 0, 0, 1)
    matrix.set(m, 0, 1, 2)
    matrix.set(m, 1, 0, 3)
    matrix.set(m, 1, 1, 4)
    assert matrix.eigenvalues(m) == [-0.37228132326901425, 5.372281323269014]
    vectors = matrix.eigenvectors(m)
    assert vectors.data == [[-0.8245648401323937, -0.42222915041526043],
                            [0.5657674649689923, -0.9230523142501933]]
    assert all(type(value) is float for row in vectors.data for value in row)


@pytest.mark.parametrize("function", [matrix.eigenvalues, matrix.eigenvectors])
def __test_empty_matrix_is_an_error__(function):
    """TV-measured: a 0x0 matrix is a runtime error in both functions."""
    with pytest.raises(ValueError, match="empty matrix"):
        function(matrix.new(0, 0, 0.0))


@pytest.mark.parametrize("function", [matrix.eigenvalues, matrix.eigenvectors])
def __test_na_element_is_an_error__(function):
    """TV-measured: a matrix with a na cell is a runtime error in both functions."""
    m = matrix.new(2, 2, 1.0)
    matrix.set(m, 0, 0, float('nan'))
    with pytest.raises(ValueError, match="contains na"):
        function(m)


@pytest.mark.parametrize("function", [matrix.eigenvalues, matrix.eigenvectors])
def __test_non_square_matrix_is_an_error__(function):
    """TV-measured: a 2x3 matrix is a runtime error in both functions."""
    with pytest.raises(ValueError, match=r"non-square \(2x3\)"):
        function(matrix.new(2, 3, 1.0))
