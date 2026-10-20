"""
@pyne

TradingView measurements (CAPITALCOM:EURUSD 60): matrix.pinv() of 29 float matrices, square,
wide and tall, full-rank, rank-deficient, near-singular and zero ones, matrix.inv() of two of
them, and matrix.pinv() of two int matrices, printed with str.tostring(x, "#.####################").
Every expected value below is a TV-measured double; a value under 1e-3 was printed rounded to 20
decimals.
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


def __test_helper_assert_tv(result: Matrix, expected: list[list[float]]) -> None:
    """
    Compare a matrix with TV's: a cell at or above 1e-3 matches to the bit, a smaller one to TV's
    20 printed decimals.
    """
    assert matrix.rows(result) == len(expected)
    assert matrix.columns(result) == len(expected[0])
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            actual = matrix.get(result, i, j)
            if abs(value) >= 1e-3 or value == 0.0:
                assert actual == value
            else:
                assert abs(actual - value) <= 5e-21


__test_helper_cases = [
    pytest.param(
        [[1.0, 2.0], [2.0, 4.0]],
        [[0.04, 0.08], [0.08, 0.16]],
        id='singular_square',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        [
            [0.014285714285714287, 0.028571428571428567],
            [0.028571428571428584, 0.05714285714285716],
            [0.04285714285714288, 0.08571428571428574],
        ],
        id='rank_one_wide',
    ),
    pytest.param(
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        id='zero_square',
    ),
    pytest.param(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [
            [-1.3333333333333317, -0.3333333333333325, 0.666666666666666],
            [1.0833333333333324, 0.3333333333333327, -0.4166666666666662],
        ],
        id='full_rank_tall',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [
            [-0.9444444444444444, 0.4444444444444444],
            [-0.11111111111111116, 0.11111111111111116],
            [0.7222222222222223, -0.22222222222222232],
        ],
        id='full_rank_wide',
    ),
    pytest.param(
        [[4.0, 7.0], [2.0, 6.0]],
        [[0.6000000000000001, -0.7000000000000001], [-0.2, 0.4]],
        id='invertible_2x2',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [
            [-0.6388888888888884, -0.166666666666667, 0.3055555555555557],
            [-0.05555555555555664, -0.00000000000000035041, 0.055555555555556066],
            [0.5277777777777783, 0.16666666666666718, -0.19444444444444492],
        ],
        id='singular_3x3',
    ),
    pytest.param(
        [[2.5, -1.3, 0.7], [3.1, 4.2, -2.2], [0.6, 1.8, 5.4]],
        [
            [0.2888744307091737, 0.08978529603122967, -0.0008674907829104276],
            [-0.1958360442420299, 0.14183474300585552, 0.08317067881153761],
            [0.03318152244632401, -0.057254391672088484, 0.15755801344610712],
        ],
        id='invertible_3x3',
    ),
    pytest.param(
        [[1.5, 2.25], [3.7, -0.4], [2.2, 1.1], [0.3, 0.9]],
        [
            [0.004207371545488586, 0.22002247773839367, 0.08368634909656783, -0.015013976542462754],
            [0.3080026512204259, -0.19451888994553473, 0.09890204893230747, 0.1337713610558774],
        ],
        id='full_rank_tall_4x2',
    ),
    pytest.param(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        [
            [0.014285714285714285, 0.028571428571428584, 0.04285714285714288],
            [0.028571428571428567, 0.05714285714285716, 0.08571428571428574],
        ],
        id='rank_one_tall',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0]],
        [[0.07142857142857142], [0.14285714285714285], [0.21428571428571427]],
        id='single_row',
    ),
    pytest.param(
        [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [4.0, 0.0, 2.0]],
        [[0.08, 0.0, 0.16], [0.0, 0.3333333333333333, 0.0], [0.04, 0.0, 0.08]],
        id='rank_two_3x3',
    ),
    pytest.param(
        [[0.1, 0.2, 0.3, 0.4], [1.7, -2.9, 3.3, 0.8]],
        [
            [0.12468951813214108, 0.06954793840039741],
            [1.1882762046696471, -0.1738698460009936],
            [0.6423248882265276, 0.11922503725782413],
            [1.3929458519622453, -0.019870839542970732],
        ],
        id='full_rank_wide_2x4',
    ),
    pytest.param(
        [[2.0], [-1.0], [3.5]],
        [[0.11594202898550726, -0.057971014492753624, 0.20289855072463767]],
        id='single_column',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        [
            [-0.4833333333333334, -0.24444444444444524, -0.005555555555555791, 0.233333333333334],
            [-0.03333333333333403, -0.011111111111111523, 0.011111111111111075,
             0.033333333333333694],
            [0.41666666666666724, 0.22222222222222315, 0.027777777777778002, -0.16666666666666746],
        ],
        id='rank_two_tall_4x3',
    ),
    pytest.param(
        [[1.0, 1.0], [1.0, 1.000000000001]],
        [[999851125986.2073, -999851125985.2076], [-999851125985.2078, 999851125985.2081]],
        id='near_singular_square_svd',
    ),
    pytest.param(
        [[1.0, 1.0], [1.0, 1.0000000001]],
        [[9999999173.59636, -9999999172.59636], [-9999999172.59636, 9999999172.59636]],
        id='near_singular_square_lu',
    ),
    pytest.param(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.000000000001]],
        [
            [499918478838.9433, -499918478838.44354],
            [500029695057.8648, -500029695057.3649],
            [-999948173895.808, 999948173895.8083],
        ],
        id='near_singular_wide',
    ),
    pytest.param(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0000001]],
        [
            [5000000.507942714, -5000000.007942713],
            [5000000.475356566, -4999999.975356567],
            [-9999999.983299278, 9999999.983299278],
        ],
        id='nearly_dependent_rows_wide',
    ),
    pytest.param(
        [[1.2, -0.7, 3.3, 2.0, 0.5], [4.1, 1.9, -2.6, 0.8, 1.4], [-1.5, 2.2, 0.9, 3.7, -0.3]],
        [
            [0.13314020698447973, 0.14377690524024614, -0.09970108403034984],
            [-0.08475768296168237, 0.06408607799385246, 0.1336054066801427],
            [0.19227491393949026, -0.07144483493082791, -0.0253371201905562],
            [0.06142169081101158, 0.04159875344063653, 0.15456866055397447],
            [0.04710155184621082, 0.04979683339572079, -0.024719477933187482],
        ],
        id='full_rank_wide_3x5',
    ),
    pytest.param(
        [[1.2, -0.7, 3.3], [2.0, 0.5, 4.1], [1.9, -2.6, 0.8], [1.4, -1.5, 2.2], [0.9, 3.7, -0.3]],
        [
            [-0.11958929642335397, -0.005430918889461088, 0.4062877542192426, 0.08225824840947246,
             0.2969563474412492],
            [-0.0024885753758107903, 0.06029400837334533, -0.10668704697820858,
             -0.04697835388263021, 0.16763706488728974],
            [0.15795700673983726, 0.1381587138264892, -0.19897272720703993, 0.01582379305741047,
             -0.12218996036420177],
        ],
        id='full_rank_tall_5x3',
    ),
    pytest.param(
        [
            [3.0, 1.0, -2.0, 0.5],
            [1.5, 4.0, 0.25, -1.0],
            [-0.75, 2.0, 5.0, 1.25],
            [2.0, -1.5, 0.5, 3.0],
        ],
        [
            [3.401129943502825, -2.4858757062146895, 1.694915254237288, -2.1016949152542375],
            [-2.3954802259887003, 2.056497175141243, -1.2203389830508473, 1.5932203389830508],
            [2.4361581920903954, -1.9480225988700568, 1.4372881355932203, -1.6542372881355933],
            [-3.8711864406779655, 3.0101694915254233, -1.9796610169491522, 2.806779661016949],
        ],
        id='invertible_4x4',
    ),
    pytest.param(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.5, -1.0, 2.5, 3.0], [1.5, 1.0, 5.5, 7.0]],
        [
            [0.009569377990430625, 0.01913875598086126, -0.01355661881977673,
             -0.0039872408293460965],
            [0.08931419457735242, 0.17862838915470491, -0.23763955342902707, -0.14832535885167455],
            [-0.006379585326953756, -0.012759170653907501, 0.0645933014354067, 0.05821371610845295],
            [0.003189792663476866, 0.006379585326953749, 0.05103668261562998, 0.05422647527910685],
        ],
        id='singular_4x4',
    ),
    pytest.param(
        [[2.0, -1.0, 0.0, 3.0], [1.0, 1.0, 4.0, -2.0], [0.5, 2.5, -1.5, 1.0]],
        [
            [0.17096211109970222, 0.10185850703357634, 0.07721531984803369],
            [-0.0691036040661259, 0.1023719067666085, 0.31953999383920323],
            [0.07269740219735085, 0.18205154533319642, -0.0716706027312866],
            [0.1963240579114899, -0.033781702433514735, 0.05503645138104529],
        ],
        id='full_rank_wide_3x4',
    ),
    pytest.param(
        [[0.3, -1.7], [2.9, 0.4], [-0.6, 1.1]],
        [
            [0.033409726178017414, 0.3274206154702134, -0.06742882880495979],
            [-0.39898260627657756, 0.09466530660908501, 0.25805767880562225],
        ],
        id='full_rank_tall_3x2',
    ),
    pytest.param(
        [[4.0]],
        [[0.25]],
        id='scalar',
    ),
    pytest.param(
        [[0.0]],
        [[0.0]],
        id='zero_scalar',
    ),
    pytest.param(
        [[0.0], [0.0]],
        [[0.0, 0.0]],
        id='zero_column',
    ),
    pytest.param(
        [[0.0, 0.0]],
        [[0.0], [0.0]],
        id='zero_row',
    ),
]


@pytest.mark.parametrize("rows, expected", __test_helper_cases)
def __test_pinv_matches_tv__(rows, expected):
    """TV-measured: the pseudoinverse matches to the last bit, rank-deficient matrices included."""
    __test_helper_assert_tv(matrix.pinv(__test_helper_matrix(rows)), expected)


@pytest.mark.parametrize("rows, expected", __test_helper_cases)
def __test_pinv_does_not_modify_its_input__(rows, expected):
    """The pseudoinverse leaves the matrix it decomposes unchanged."""
    m = __test_helper_matrix(rows)
    matrix.pinv(m)
    assert m.data == rows


@pytest.mark.parametrize("rows, expected", [
    pytest.param([[4.0, 7.0], [2.0, 6.0]],
                 [[0.6000000000000001, -0.7000000000000001], [-0.2, 0.4]],
                 id='invertible_2x2'),
    pytest.param([[2.5, -1.3, 0.7], [3.1, 4.2, -2.2], [0.6, 1.8, 5.4]],
                 [
                     [0.2888744307091737, 0.08978529603122967, -0.0008674907829104276],
                     [-0.1958360442420299, 0.14183474300585552, 0.08317067881153761],
                     [0.03318152244632401, -0.057254391672088484, 0.15755801344610712],
                 ],
                 id='invertible_3x3'),
])
def __test_inv_matches_tv__(rows, expected):
    """TV-measured: the inverse matches to the last bit, and is what pinv() returns for it."""
    m = __test_helper_matrix(rows)
    __test_helper_assert_tv(matrix.inv(m), expected)
    assert matrix.inv(m).data == matrix.pinv(m).data


@pytest.mark.parametrize("rows", [
    pytest.param([[1.0, 2.0], [2.0, 4.0]], id='singular'),
    pytest.param([[1.0, 1.0], [1.0, 1.000000000001]], id='pivot_under_1e-11'),
])
def __test_inv_of_a_singular_matrix_is_an_error__(rows):
    """TV-measured: a zero pivot, and one under 1e-11 too, is a runtime error in inv()."""
    with pytest.raises(ValueError, match="singular"):
        matrix.inv(__test_helper_matrix(rows))


def __test_int_matrix_pinv_keeps_fractional_values__():
    """TV-measured: an int matrix gives the same fractional values as the same float matrix."""
    singular = matrix.new(2, 2, 0)
    matrix.set(singular, 0, 1, 2)
    matrix.set(singular, 1, 0, 2)
    matrix.set(singular, 0, 0, 1)
    matrix.set(singular, 1, 1, 4)
    assert matrix.pinv(singular).data == [[0.04, 0.08], [0.08, 0.16]]
    wide = matrix.new(2, 3, 0)
    for i, value in enumerate([1, 2, 3, 4, 5, 6]):
        matrix.set(wide, i // 3, i % 3, value)
    result = matrix.pinv(wide)
    assert matrix.get(result, 0, 0) == -0.9444444444444444
    assert matrix.get(result, 2, 1) == -0.22222222222222232
