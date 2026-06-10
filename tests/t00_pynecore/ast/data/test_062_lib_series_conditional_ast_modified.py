"""
@pyne
"""
from pynecore import lib
__pyne_slot_layout__ = {'main': {'init': (None, None, None, None), 'series': ((0, None), (1, None), (2, None), (3, None)), 'varip': (), 'children': (), 'names': ('__lib·low', '__lib·high', '__lib·open', '__lib·close')}}

def main(__state·main__):
    __lib·low = __state·main__[0].add(lib.low)
    __lib·high = __state·main__[1].add(lib.high)
    __lib·open = __state·main__[2].add(lib.open)
    __lib·close = __state·main__[3].add(lib.close)

    def nested():
        value = __state·main__[0][1] if True else lib.low
        my_box: lib.box = lib.box(__state·main__[1][2] if False else lib.high)
        result = __state·main__[3][3] if __state·main__[2][1] > 100 else lib.close
        return (value, my_box, result)
    res = nested()
    print(res)
main.__pyne_layout__ = __pyne_slot_layout__['main']
