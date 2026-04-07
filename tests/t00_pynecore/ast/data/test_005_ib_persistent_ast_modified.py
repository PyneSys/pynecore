"""
@pyne
"""
from pynecore.types import IBPersistent
__persistent_main·var_count__ = 0
__persistent_main·varip_count__ = 0
__persistent_main·varip_total__ = 0.0
__persistent_main·varip_total___kahan_c__ = 0.0
__persistent_function_vars__ = {'main': ('__persistent_main·var_count__', '__persistent_main·varip_count__', '__persistent_main·varip_total__', '__persistent_main·varip_total___kahan_c__')}
__varip_function_vars__ = {'main': ('__persistent_main·varip_count__', '__persistent_main·varip_total__', '__persistent_main·varip_total___kahan_c__')}

def main():
    global __persistent_main·var_count__, __persistent_main·varip_count__, __persistent_main·varip_total__, __persistent_main·varip_total___kahan_c__
    __persistent_main·var_count__ += 1
    __persistent_main·varip_count__ += 1
    ((__kahan_corrected__ := (some_value - __persistent_main·varip_total___kahan_c__)), (__kahan_new_sum__ := (__persistent_main·varip_total__ + __kahan_corrected__)), (__persistent_main·varip_total___kahan_c__ := (__kahan_new_sum__ - __persistent_main·varip_total__ - __kahan_corrected__)), (__persistent_main·varip_total__ := __kahan_new_sum__))[-1]
