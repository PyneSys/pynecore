"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (1,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}, 'main·test': {'init': (1,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}}

def main(__state·main__):
    __state·main__[0] += 1

    def test(__state__):
        __state__[0] += 1
        return __state__[0]
    test.__pyne_layout__ = __pyne_slot_layout__['main·test']
main.__pyne_layout__ = __pyne_slot_layout__['main']
