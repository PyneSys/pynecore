"""
Transparent aliases for type checkers, same rationale as persistent.pyi:
NA-free static view, runtime module unchanged.
"""
from typing import TypeVar, TypeAlias

T = TypeVar('T')

IBPersistent: TypeAlias = T
IBPersistentSeries: TypeAlias = T
