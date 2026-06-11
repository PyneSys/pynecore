"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (1, None, False), 'series': (), 'varip': (), 'children': (), 'names': ('p1', 'p2', 'p2·flag')}}

def main(__state__):
    if not __state__[2]:
        __state__[1] = __state__[0] + 1
        __state__[2] = True
    print(__state__[1])
main.__pyne_layout__ = __pyne_slot_layout__['main']
