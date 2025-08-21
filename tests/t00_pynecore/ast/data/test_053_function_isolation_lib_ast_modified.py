"""
@pyne
"""
from pynecore.core.series import SeriesImpl
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.function_isolation import isolate_function
__series_main·e__ = SeriesImpl()
__series_function_vars__ = {'main': ('__series_main·e__',)}
__scope_id__ = ''

def main():
    global __scope_id__
    __call_counter·main·lib.ta.ema·0__ = 0
    e = __series_main·e__.add(isolate_function(lib.ta.ema, 'main·lib.ta.ema·0', __scope_id__, -1, (__call_counter·main·lib.ta.ema·0__ := (__call_counter·main·lib.ta.ema·0__ + 1)))(lib.close, 9))
    print(__series_main·e__[1])
