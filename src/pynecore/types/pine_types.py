from typing import TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.types.type_checker import *
from pynecore.types.na import NA
from pynecore.types.series import Series

PyneFloat: TypeAlias = float | NA[float] | Series[float]
PyneInt: TypeAlias = int | NA[int] | Series[int]
PyneStr: TypeAlias = str | NA[str] | Series[str]
PyneBool: TypeAlias = bool | NA[bool] | Series[bool]
