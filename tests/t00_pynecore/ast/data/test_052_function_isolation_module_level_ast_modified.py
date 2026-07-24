"""
@pyne
"""
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'t1': {'init': (None,), 'series': ((0, None, 'float'),), 'varip': (), 'children': (), 'names': ('a',)}, 't2': {'init': (None,), 'series': ((0, None, 'float'),), 'varip': (), 'children': (), 'names': ('a',)}, 'main': {'init': (None, None, None), 'series': (), 'varip': (), 'children': ((0, 'main·t1·0', False), (1, 'main·t1·1', False), (2, 'main·t2·2', False)), 'names': ('main·t1·0', 'main·t1·1', 'main·t2·2')}}

def t1(__state__):
    a = __state__[0].add(1)
    a = __state__[0].set(a + 1)
    return __state__[0][1]
t1.__pyne_layout__ = __pyne_slot_layout__['t1']

def t2(__state__):
    a = __state__[0].add(1)
    a = __state__[0].set(a + 1)
    return __state__[0][1]
t2.__pyne_layout__ = __pyne_slot_layout__['t2']

def main(__state__):
    a = t1(__st__ if (__st__ := __state__[0]) is not None else __resolve_slot__(__state__, 0, t1))
    print(a)
    b = t1(__st__ if (__st__ := __state__[1]) is not None else __resolve_slot__(__state__, 1, t1))
    print(b)
    c = t2(__st__ if (__st__ := __state__[2]) is not None else __resolve_slot__(__state__, 2, t2))
    print(c)
main.__pyne_layout__ = __pyne_slot_layout__['main']
