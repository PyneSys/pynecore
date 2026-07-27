"""
Transparent aliases for type checkers, same rationale as persistent.pyi:
NA-free static view, runtime module unchanged.
"""
from typing import TypeVar, TypeAlias

T = TypeVar('T')

IBPersistent: TypeAlias = T
"""
A ``Persistent`` that survives a re-run of the same bar instead of being
rolled back — Pine's ``varip``.

On historical bars the script runs once per bar, so it behaves exactly like
``Persistent``. The difference shows on live ticks and on strategy
re-executions (``calc_on_order_fills``): ``Persistent`` returns to the bar's
committed state before each run, ``IBPersistent`` keeps what it accumulated.
"""

IBPersistentSeries: TypeAlias = T
"""
An ``IBPersistent`` that also keeps its history — ``varip`` applied to a series.

The value carries over between bars and survives a re-run of the same bar, and
``[n]`` still looks back over past bars: ``my_var[1]`` is its value one bar ago.
"""
