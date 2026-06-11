"""
@pyne
"""
__pyne_slot_layout__ = {'main': {'init': (0.0, 0.0, 0), 'series': (), 'varip': (), 'children': (), 'names': ('cumulative', 'cumulative·kahan', 'counter')}}

def main(__state__):
    __kahan_corrected__ = some_value - __state__[1]
    __kahan_new_sum__ = __state__[0] + __kahan_corrected__
    __state__[1] = __kahan_new_sum__ - __state__[0] - __kahan_corrected__
    __state__[0] = __kahan_new_sum__
    __state__[2] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
