"""
NA-free static view of the Pyne scalar aliases (the "plain T" policy — see
persistent.pyi). The ``type_checker`` wildcard import swaps ``float``/``int``/
``bool`` for their ``[n]``-indexable subclasses under PyCharm
(``TYPECHECKER=pycharm``); under pyright they stay the builtins.
Runtime (pine_types.py) is unchanged.
"""
from typing import SupportsFloat, TypeAlias

from pynecore.types.type_checker import *

PyneFloat: TypeAlias = float
PyneInt: TypeAlias = int
PyneStr: TypeAlias = str
PyneBool: TypeAlias = bool


def pine_int(value: SupportsFloat, /) -> PyneInt:
    """
    A native number handed back to a script as a Pine int.

    A Pine int is a static type only: its runtime form is a double, so this is
    ``float`` itself and costs nothing beyond the conversion a ``float(...)``
    call would make. It does not truncate, the value is expected to be integral
    already.

    :param value: The native number.
    :return: The same number in the Pine int representation.
    """
