"""
@pyne
"""
from pynecore.core.series import SeriesImpl
__series_main_s2__ = SeriesImpl()
__series_main_test_s__ = SeriesImpl()
__series_function_vars__ = {'main.test': ['__series_main_test_s__'], 'main': ['__series_main_s2__']}

def main():

    def test(length):
        s = __series_main_test_s__.add(1)
        print(__series_main_test_s__[length])
    s = 1
    s2 = __series_main_s2__.add(0.5)
    print(__series_main_s2__[1], s)
