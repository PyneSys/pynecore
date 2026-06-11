"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('s2',)}, 'main·test': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('s',)}}

def main(__state·main__):

    def test(__state__, length):
        s = __state__[0].add(1)
        print(__state__[0][length])
    test.__pyne_layout__ = __pyne_slot_layout__['main·test']
    s = 1
    s2 = __state·main__[0].add(0.5)
    print(__state·main__[0][1], s)
main.__pyne_layout__ = __pyne_slot_layout__['main']
