"""
@pyne
"""
from pynecore.types import IBPersistent
__pyne_slot_layout__ = {'main': {'init': (0, 0, 0.0, 0.0), 'series': (), 'varip': (1, 2, 3), 'children': (), 'names': ('var_count', 'varip_count', 'varip_total', 'varip_total·kahan')}}

def main(__state__):
    __state__[0] += 1
    __state__[1] += 1
    __kahan_corrected__ = some_value - __state__[3]
    __kahan_new_sum__ = __state__[2] + __kahan_corrected__
    __state__[3] = __kahan_new_sum__ - __state__[2] - __kahan_corrected__
    __state__[2] = __kahan_new_sum__
main.__pyne_layout__ = __pyne_slot_layout__['main']
