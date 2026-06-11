"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('s',)}}

def main(__state__):
    s = __state__[0].add(1)
    print(__state__[0][5])
main.__pyne_layout__ = __pyne_slot_layout__['main']
