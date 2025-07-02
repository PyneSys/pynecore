"""
@pyne
"""
from pynecore.core.series import SeriesImpl
from pynecore import lib
from pynecore.core.function_isolation import isolate_function
__series_main·nested·__lib·close__ = SeriesImpl()
__series_main·nested·__lib·high__ = SeriesImpl()
__series_main·nested·__lib·low__ = SeriesImpl()
__series_main·nested·__lib·open__ = SeriesImpl()
__series_function_vars__ = {'main.nested': ('__series_main·nested·__lib·low__', '__series_main·nested·__lib·high__', '__series_main·nested·__lib·open__', '__series_main·nested·__lib·close__')}
__scope_id__ = ''

def main():
    global __scope_id__

    def nested():
        __lib·low = __series_main·nested·__lib·low__.add(lib.low)
        __lib·high = __series_main·nested·__lib·high__.add(lib.high)
        __lib·open = __series_main·nested·__lib·open__.add(lib.open)
        __lib·close = __series_main·nested·__lib·close__.add(lib.close)
        value = __series_main·nested·__lib·low__[1] if True else lib.low
        my_box: lib.box = lib.box(__series_main·nested·__lib·high__[2] if False else lib.high)
        result = __series_main·nested·__lib·close__[3] if __series_main·nested·__lib·open__[1] > 100 else lib.close
        return (value, my_box, result)
    res = isolate_function(nested, 'main|nested|0', __scope_id__, -1)()
    print(res)
