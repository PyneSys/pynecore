"""
@pyne
"""
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'main': {'init': (None, None), 'series': ((0, None, 'float'),), 'varip': (), 'children': ((1, 'main·lib.ta.ema·0', False),), 'names': ('e', 'main·lib.ta.ema·0')}}

def main(__state__):
    e = __state__[0].add(lib.ta.ema(__st__ if (__st__ := __state__[1]) is not None else __resolve_slot__(__state__, 1, lib.ta.ema), lib.close, 9))
    print(__state__[0][1])
main.__pyne_layout__ = __pyne_slot_layout__['main']
