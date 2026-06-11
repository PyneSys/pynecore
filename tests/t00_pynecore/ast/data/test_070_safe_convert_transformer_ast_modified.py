"""
@pyne
"""
from pynecore import lib
from pynecore.core import safe_convert
__pyne_slot_layout__ = {}

def main():
    na_value = lib.na(float)
    float_val = safe_convert.safe_float(na_value)
    na_int = lib.na(int)
    int_val = safe_convert.safe_int(na_int)
    regular_val = 3.14
    regular_float = safe_convert.safe_float(regular_val)
