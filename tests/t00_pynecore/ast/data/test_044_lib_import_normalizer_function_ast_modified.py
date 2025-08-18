"""
@pyne
"""
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.function_isolation import isolate_function
__scope_id__ = ''

def main():
    global __scope_id__
    __call_counter·main·lib.ta.sma·0__ = 0
    print(lib.close, lib.hl2, lib.ta, isolate_function(lib.ta.sma, 'main·lib.ta.sma·0', __scope_id__, -1, (__call_counter·main·lib.ta.sma·0__ := (__call_counter·main·lib.ta.sma·0__ + 1)))(lib.close, 12))
