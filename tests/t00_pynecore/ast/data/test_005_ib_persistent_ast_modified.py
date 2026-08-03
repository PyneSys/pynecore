"""
@pyne
"""
from pynecore.types import IBPersistent
__pyne_slot_layout__ = {'main': {'init': (0, 0, 0.0), 'series': (), 'varip': (1, 2), 'children': (), 'names': ('var_count', 'varip_count', 'varip_total')}}

def main(__state__):
    __state__[0] += 1
    __state__[1] += 1
    __state__[2] += some_value
main.__pyne_layout__ = __pyne_slot_layout__['main']
