"""
@pyne
"""
from pynecore import lib
__pyne_slot_layout__ = {'main': {'init': (None, None, None), 'series': ((0, None, 'float'), (1, None, 'float'), (2, None, 'float')), 'varip': (), 'children': (), 'names': ('__lib·close', '__lib·high', '__lib·low')}}

def main(__state·main__):
    __lib·close = __state·main__[0].add(lib.close)
    __lib·high = __state·main__[1].add(lib.high)
    __lib·low = __state·main__[2].add(lib.low)
    a: float = __state·main__[0][10]
    print(a)

    def nested():
        b: float = __state·main__[1][1]
        return b
    result = nested()
    c: float = __state·main__[2][2]
    print(result, c)
main.__pyne_layout__ = __pyne_slot_layout__['main']
