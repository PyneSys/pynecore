"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (None, 0.5), 'series': ((0, None, 'float'),), 'varip': (), 'children': (), 'names': ('s2', 's2')}, 'main·t': {'init': (None, 1), 'series': ((0, None, None),), 'varip': (), 'children': (), 'names': ('s', 's')}}

def main(__state·main__):

    def t(__state__, length):
        __state__[1] = __state__[0].add(__state__[1])
        __state__[1] = __state__[0].set(__state__[1] + 1)
        print(__state__[1], __state__[0][length])
    t.__pyne_layout__ = __pyne_slot_layout__['main·t']
    s = 1
    __state·main__[1] = __state·main__[0].add(__state·main__[1])
    __state·main__[1] = __state·main__[0].set(__state·main__[1] + 1)
    print(__state·main__[0][1], s)
main.__pyne_layout__ = __pyne_slot_layout__['main']
