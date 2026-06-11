"""
@pyne
"""
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': (), 'varip': (), 'children': ((0, 'main·lib.ta.sma·0', False),), 'names': ('main·lib.ta.sma·0',)}}

def main(__state__):
    print(lib.close, lib.hl2, lib.ta, lib.ta.sma(__st__ if (__st__ := __state__[0]) is not None else __resolve_slot__(__state__, 0, lib.ta.sma), lib.close, 12))
main.__pyne_layout__ = __pyne_slot_layout__['main']
