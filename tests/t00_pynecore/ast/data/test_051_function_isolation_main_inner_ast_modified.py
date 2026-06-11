"""
@pyne
"""
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'main': {'init': (None, None), 'series': (), 'varip': (), 'children': ((0, 'main·t·0', False), (1, 'main·t·1', False)), 'names': ('main·t·0', 'main·t·1')}, 'main·t': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('a',)}}

def main(__state·main__):

    def t(__state__):
        a = __state__[0].add(1)
        a = __state__[0].set(a + 1)
        return __state__[0][1]
    t.__pyne_layout__ = __pyne_slot_layout__['main·t']
    a = t(__st__ if (__st__ := __state·main__[0]) is not None else __resolve_slot__(__state·main__, 0, t))
    print(a)
    b = t(__st__ if (__st__ := __state·main__[1]) is not None else __resolve_slot__(__state·main__, 1, t))
    print(b)
main.__pyne_layout__ = __pyne_slot_layout__['main']
