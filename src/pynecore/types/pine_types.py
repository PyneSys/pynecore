from typing import TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.types.type_checker import *
from pynecore.types.na import NA
from pynecore.types.series import Series

PyneFloat: TypeAlias = float | NA[float] | Series[float]
# Pine's int is a static type only: at runtime an int-typed value is a double,
# so a Pine int arrives as a native float (a literal may still be a Python int)
# and its na is the same nan a float has
PyneInt: TypeAlias = float | int | Series[int]
PyneStr: TypeAlias = str | NA[str] | Series[str]
PyneBool: TypeAlias = bool | NA[bool] | Series[bool]
