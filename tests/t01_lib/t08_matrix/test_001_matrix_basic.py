"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore import pine_range
from pynecore.lib import array, bar_index, color, log, matrix, na, order, plot, script, string
from pynecore.types import Matrix


@script.indicator("Matrix Operations Test", overlay=False)
def main():
    if bar_index == 0:
        m1: Matrix[float] = matrix.new(3, 3, 1.5)
        log.info('Test 1 - new(3,3,1.5): rows=' + string.tostring(matrix.rows(m1)) + ', cols=' + string.tostring(
            matrix.columns(m1)) + ', elements=' + string.tostring(matrix.elements_count(m1)))

        matrix.set(m1, 0, 0, 5.0)
        matrix.set(m1, 1, 1, 10.0)
        matrix.set(m1, 2, 2, 15.0)
        val = matrix.get(m1, 1, 1)
        log.info('Test 2 - get(1,1): ' + string.tostring(val))

        matrix.fill(m1, 2.0)
        log.info('Test 3 - fill(2.0) then get(0,0): ' + string.tostring(matrix.get(m1, 0, 0)))

        matrix.add_row(m1, 1)
        matrix.add_col(m1, 2)
        log.info('Test 4 - after add_row/col: rows=' + string.tostring(matrix.rows(m1)) + ', cols=' + string.tostring(
            matrix.columns(m1)))

        m2: Matrix[float] = matrix.new(3, 3, 0)
        matrix.set(m2, 0, 0, 1.0)
        matrix.set(m2, 1, 1, 1.0)
        matrix.set(m2, 2, 2, 1.0)
        log.info('Test 5 - is_identity: ' + string.tostring(matrix.is_identity(m2)))
        log.info('Test 5 - is_diagonal: ' + string.tostring(matrix.is_diagonal(m2)))
        log.info('Test 5 - is_symmetric: ' + string.tostring(matrix.is_symmetric(m2)))

        m3: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m3, 0, 0, 1.0)
        matrix.set(m3, 0, 1, 2.0)
        matrix.set(m3, 1, 0, 3.0)
        matrix.set(m3, 1, 1, 4.0)

        det = matrix.det(m3)
        log.info('Test 6 - det([[1,2],[3,4]]): ' + string.tostring(det))

        trace_val = matrix.trace(m3)
        log.info('Test 6 - trace: ' + string.tostring(trace_val))

        m4: Matrix[float] = matrix.new(2, 3, 0)
        matrix.set(m4, 0, 0, 1.0)
        matrix.set(m4, 0, 1, 2.0)
        matrix.set(m4, 0, 2, 3.0)
        matrix.set(m4, 1, 0, 4.0)
        matrix.set(m4, 1, 1, 5.0)
        matrix.set(m4, 1, 2, 6.0)

        avg_val = matrix.avg(m4)
        max_val = matrix.max(m4)
        min_val = matrix.min(m4)

        log.info('Test 7 - avg: ' + string.tostring(avg_val) + ', max: ' + string.tostring(
            max_val) + ', min: ' + string.tostring(min_val))

        row_arr = matrix.row(m4, 0)
        col_arr = matrix.col(m4, 1)
        log.info('Test 8 - row(0): [' + string.tostring(array.get(row_arr, 0)) + ',' + string.tostring(
            array.get(row_arr, 1)) + ',' + string.tostring(array.get(row_arr, 2)) + ']')

        log.info('Test 8 - col(1): [' + string.tostring(array.get(col_arr, 0)) + ',' + string.tostring(
            array.get(col_arr, 1)) + ']')

        m5: Matrix[float] = matrix.new(2, 3, 0)
        matrix.set(m5, 0, 0, 1.0)
        matrix.set(m5, 0, 1, 2.0)
        matrix.set(m5, 0, 2, 3.0)
        matrix.set(m5, 1, 0, 4.0)
        matrix.set(m5, 1, 1, 5.0)
        matrix.set(m5, 1, 2, 6.0)

        m6: Matrix[float] = matrix.new(3, 2, 0)
        matrix.set(m6, 0, 0, 7.0)
        matrix.set(m6, 0, 1, 8.0)
        matrix.set(m6, 1, 0, 9.0)
        matrix.set(m6, 1, 1, 10.0)
        matrix.set(m6, 2, 0, 11.0)
        matrix.set(m6, 2, 1, 12.0)

        m_mult = matrix.mult(m5, m6)
        log.info('Test 9 - mult result shape: ' + string.tostring(matrix.rows(m_mult)) + 'x' + string.tostring(
            matrix.columns(m_mult)))

        log.info('Test 9 - mult[0,0]: ' + string.tostring(matrix.get(m_mult, 0, 0)) + ', mult[1,1]: ' + string.tostring(
            matrix.get(m_mult, 1, 1)))

        m_trans = matrix.transpose(m5)
        log.info('Test 10 - transpose shape: ' + string.tostring(matrix.rows(m_trans)) + 'x' + string.tostring(
            matrix.columns(m_trans)))

        m_copy = matrix.copy(m3)
        matrix.set(m_copy, 0, 0, 99.0)
        log.info(
            'Test 11 - original[0,0]: ' + string.tostring(matrix.get(m3, 0, 0)) + ', copy[0,0]: ' + string.tostring(
                matrix.get(m_copy, 0, 0)))

        m7: Matrix[float] = matrix.new(3, 3, 0)
        for i in pine_range(0, 2):
            for j in pine_range(0, 2):
                matrix.set(m7, i, j, i * 3 + j + 1.0)

        m_sub = matrix.submatrix(m7, 0, 2, 0, 2)
        log.info('Test 12 - submatrix shape: ' + string.tostring(matrix.rows(m_sub)) + 'x' + string.tostring(
            matrix.columns(m_sub)))

        log.info(
            'Test 12 - submatrix[0,0]: ' + string.tostring(matrix.get(m_sub, 0, 0)) + ', [1,1]: ' + string.tostring(
                matrix.get(m_sub, 1, 1)))

        m8: Matrix[float] = matrix.new(2, 3, 0)
        for i in pine_range(0, 1):
            for j in pine_range(0, 2):
                matrix.set(m8, i, j, i * 3 + j + 1.0)

        matrix.reshape(m8, 3, 2)
        log.info('Test 13 - reshape to 3x2: rows=' + string.tostring(matrix.rows(m8)) + ', cols=' + string.tostring(
            matrix.columns(m8)))

        m9: Matrix[float] = matrix.new(3, 3, 1.0)
        matrix.remove_row(m9, 1)
        matrix.remove_col(m9, 1)
        log.info(
            'Test 14 - after remove_row/col: rows=' + string.tostring(matrix.rows(m9)) + ', cols=' + string.tostring(
                matrix.columns(m9)))

        m10: Matrix[float] = matrix.new(3, 3, 0)
        matrix.set(m10, 0, 0, 1.0)
        matrix.set(m10, 1, 0, 2.0)
        matrix.set(m10, 2, 0, 3.0)

        matrix.swap_rows(m10, 0, 2)
        log.info(
            'Test 15 - after swap_rows: [0,0]=' + string.tostring(matrix.get(m10, 0, 0)) + ', [2,0]=' + string.tostring(
                matrix.get(m10, 2, 0)))

        matrix.swap_columns(m10, 0, 1)
        log.info('Test 15 - after swap_columns: [0,0]=' + string.tostring(
            matrix.get(m10, 0, 0)) + ', [0,1]=' + string.tostring(matrix.get(m10, 0, 1)))

        m11: Matrix[float] = matrix.new(2, 2, 1.0)
        m12: Matrix[float] = matrix.new(2, 2, 2.0)
        m_concat = matrix.concat(m11, m12)
        log.info('Test 16 - concat shape: ' + string.tostring(matrix.rows(m_concat)) + 'x' + string.tostring(
            matrix.columns(m_concat)))

        m13: Matrix[float] = matrix.new(3, 3, 0)
        matrix.set(m13, 0, 0, 9.0)
        matrix.set(m13, 0, 1, 3.0)
        matrix.set(m13, 0, 2, 6.0)
        matrix.set(m13, 1, 0, 2.0)
        matrix.set(m13, 1, 1, 8.0)
        matrix.set(m13, 1, 2, 5.0)
        matrix.set(m13, 2, 0, 7.0)
        matrix.set(m13, 2, 1, 1.0)
        matrix.set(m13, 2, 2, 4.0)

        matrix.sort(m13, 0, order.ascending)
        log.info('Test 17 - after sort col 0: [0,0]=' + string.tostring(
            matrix.get(m13, 0, 0)) + ', [1,0]=' + string.tostring(matrix.get(m13, 1, 0)) + ', [2,0]=' + string.tostring(
            matrix.get(m13, 2, 0)))

        m14: Matrix[float] = matrix.new(2, 3, 0)
        matrix.set(m14, 0, 0, 1.0)
        matrix.set(m14, 0, 1, 2.0)
        matrix.set(m14, 0, 2, 3.0)
        matrix.set(m14, 1, 0, 4.0)
        matrix.set(m14, 1, 1, 5.0)
        matrix.set(m14, 1, 2, 6.0)

        matrix.reverse(m14)
        log.info(
            'Test 18 - after reverse: [0,0]=' + string.tostring(matrix.get(m14, 0, 0)) + ', [1,2]=' + string.tostring(
                matrix.get(m14, 1, 2)))

        m15: Matrix[float] = matrix.new(3, 3, 0)
        matrix.set(m15, 0, 0, 1.0)
        matrix.set(m15, 0, 1, 2.0)
        matrix.set(m15, 0, 2, 3.0)
        matrix.set(m15, 1, 0, 2.0)
        matrix.set(m15, 1, 1, 4.0)
        matrix.set(m15, 1, 2, 6.0)
        matrix.set(m15, 2, 0, 3.0)
        matrix.set(m15, 2, 1, 6.0)
        matrix.set(m15, 2, 2, 9.0)

        rank_val = matrix.rank(m15)
        log.info('Test 19 - rank: ' + string.tostring(rank_val))

        m16: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m16, 0, 0, 2.0)
        matrix.set(m16, 0, 1, 0.0)
        matrix.set(m16, 1, 0, 0.0)
        matrix.set(m16, 1, 1, 2.0)

        m_pow = matrix.pow(m16, 3)
        log.info('Test 20 - pow(3): [0,0]=' + string.tostring(matrix.get(m_pow, 0, 0)) + ', [1,1]=' + string.tostring(
            matrix.get(m_pow, 1, 1)))

        m17: Matrix[float] = matrix.new(2, 3, 0)
        matrix.set(m17, 0, 0, 1.0)
        matrix.set(m17, 0, 1, 2.0)
        matrix.set(m17, 0, 2, 2.0)
        matrix.set(m17, 1, 0, 3.0)
        matrix.set(m17, 1, 1, 2.0)
        matrix.set(m17, 1, 2, 4.0)

        median_val = matrix.median(m17)
        mode_val = matrix.mode(m17)
        log.info('Test 21 - median: ' + string.tostring(median_val) + ', mode: ' + string.tostring(mode_val))

        m18: Matrix[float] = matrix.new(2, 3, 0.0)
        log.info('Test 22 - is_zero: ' + string.tostring(matrix.is_zero(m18)) + ', is_square: ' + string.tostring(
            matrix.is_square(m18)))

        m19: Matrix[float] = matrix.new(2, 2, 3.0)
        m20: Matrix[float] = matrix.new(2, 2, 1.0)
        m_sum = matrix.sum(m19, m20)
        log.info('Test 23 - sum: [0,0]=' + string.tostring(matrix.get(m_sum, 0, 0)) + ' (should be 4.0)')

        m_diff = matrix.diff(m19, m20)
        log.info('Test 23b - diff: [0,0]=' + string.tostring(matrix.get(m_diff, 0, 0)) + ' (should be 2.0)')

        m21: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m21, 0, 0, 1.0)
        matrix.set(m21, 0, 1, 2.0)
        matrix.set(m21, 1, 0, 3.0)
        matrix.set(m21, 1, 1, 4.0)

        m22: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m22, 0, 0, 0.0)
        matrix.set(m22, 0, 1, 5.0)
        matrix.set(m22, 1, 0, 6.0)
        matrix.set(m22, 1, 1, 7.0)

        m_kron = matrix.kron(m21, m22)
        log.info('Test 24 - kron shape: ' + string.tostring(matrix.rows(m_kron)) + 'x' + string.tostring(
            matrix.columns(m_kron)))

        log.info(
            'Test 24 - kron[0,1]: ' + string.tostring(matrix.get(m_kron, 0, 1)) + ', kron[3,3]: ' + string.tostring(
                matrix.get(m_kron, 3, 3)))

        m23: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m23, 0, 0, 4.0)
        matrix.set(m23, 0, 1, -2.0)
        matrix.set(m23, 1, 0, 1.0)
        matrix.set(m23, 1, 1, 1.0)

        eigenvals = matrix.eigenvalues(m23)
        if not na(eigenvals):
            log.info('Test 25 - eigenvalues count: ' + string.tostring(array.size(eigenvals)))
            if array.size(eigenvals) >= 2:
                log.info('Test 25 - eigenvalue[0]: ' + string.tostring(
                    array.get(eigenvals, 0)) + ', eigenvalue[1]: ' + string.tostring(array.get(eigenvals, 1)))

        m24: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m24, 0, 0, 4.0)
        matrix.set(m24, 0, 1, 7.0)
        matrix.set(m24, 1, 0, 2.0)
        matrix.set(m24, 1, 1, 6.0)

        m_inv = matrix.inv(m24)
        if not na(m_inv):
            log.info(
                'Test 26 - inv[0,0]: ' + string.tostring(matrix.get(m_inv, 0, 0)) + ', inv[1,1]: ' + string.tostring(
                    matrix.get(m_inv, 1, 1)))
        else:
            log.info('Test 26 - inv: matrix is singular')

        m25: Matrix[float] = matrix.new(3, 2, 0)
        matrix.set(m25, 0, 0, 1.0)
        matrix.set(m25, 0, 1, 2.0)
        matrix.set(m25, 1, 0, 3.0)
        matrix.set(m25, 1, 1, 4.0)
        matrix.set(m25, 2, 0, 5.0)
        matrix.set(m25, 2, 1, 6.0)

        m_pinv = matrix.pinv(m25)
        log.info('Test 27 - pinv shape: ' + string.tostring(matrix.rows(m_pinv)) + 'x' + string.tostring(
            matrix.columns(m_pinv)))

        m26: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m26, 0, 0, 0.3)
        matrix.set(m26, 0, 1, 0.7)
        matrix.set(m26, 1, 0, 0.4)
        matrix.set(m26, 1, 1, 0.6)
        log.info('Test 28 - is_stochastic: ' + string.tostring(matrix.is_stochastic(m26)))

        m27: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m27, 0, 0, 0.0)
        matrix.set(m27, 0, 1, 1.0)
        matrix.set(m27, 1, 0, 2.0)
        matrix.set(m27, 1, 1, 0.0)
        log.info('Test 28 - is_antidiagonal: ' + string.tostring(matrix.is_antidiagonal(m27)))

        m28: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m28, 0, 0, 0.0)
        matrix.set(m28, 0, 1, 2.0)
        matrix.set(m28, 1, 0, -2.0)
        matrix.set(m28, 1, 1, 0.0)
        log.info('Test 29 - is_antisymmetric: ' + string.tostring(matrix.is_antisymmetric(m28)))

        m29: Matrix[float] = matrix.new(2, 2, 0)
        matrix.set(m29, 0, 0, 0.0)
        matrix.set(m29, 0, 1, 1.0)
        matrix.set(m29, 1, 0, 1.0)
        matrix.set(m29, 1, 1, 0.0)
        log.info('Test 29 - is_binary: ' + string.tostring(matrix.is_binary(m29)))

        m30: Matrix[float] = matrix.new(3, 3, 0)
        matrix.set(m30, 0, 0, 1.0)
        matrix.set(m30, 0, 1, 2.0)
        matrix.set(m30, 0, 2, 3.0)
        matrix.set(m30, 1, 0, 0.0)
        matrix.set(m30, 1, 1, 4.0)
        matrix.set(m30, 1, 2, 5.0)
        matrix.set(m30, 2, 0, 0.0)
        matrix.set(m30, 2, 1, 0.0)
        matrix.set(m30, 2, 2, 6.0)
        log.info('Test 30 - is_triangular: ' + string.tostring(matrix.is_triangular(m30)))

        log.info('All matrix operation tests completed!')

    plot(0, title='Test Complete', color=color.green)


def __test_matrix_basic__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Basic """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)
