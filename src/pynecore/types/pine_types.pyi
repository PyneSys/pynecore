"""
NA-free static view of the Pyne scalar aliases (the "plain T" policy — see
persistent.pyi). The ``type_checker`` wildcard import swaps ``float``/``int``/
``bool`` for their ``[n]``-indexable subclasses under PyCharm
(``TYPECHECKER=pycharm``); under pyright they stay the builtins.
Runtime (pine_types.py) is unchanged.
"""
from typing import TypeAlias

from pynecore.types.type_checker import *

PyneFloat: TypeAlias = float
PyneInt: TypeAlias = int
PyneStr: TypeAlias = str
PyneBool: TypeAlias = bool
