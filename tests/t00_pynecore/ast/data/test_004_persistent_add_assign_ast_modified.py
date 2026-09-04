"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (0.0, 0.0), 'series': (), 'varip': (), 'children': (), 'names': ('cumulative', 'counter')}}

def main(__state__):
    __state__[0] += some_value
    __state__[1] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
