"""
Series is a transparent alias: ``Series[T]`` IS ``T`` for type checkers.

The Pine dual behavior (scalar AND ``[n]``-history-indexable at once) cannot be
expressed in a stub — Python typing has no intersection type, and a Union means
"either", not "both at once". The alias makes the scalar side (arithmetic,
calls, assignments) fully type-correct; the history-indexing side is covered by
the ``type_checker`` compat layer (PyCharm) or by suppressing
``reportIndexIssue`` in the generated pyrightconfig (pyright/Pylance).

The old ``TypeAlias`` syntax (not PEP 695 ``type``) is kept deliberately:
PyCharm resolves unbound-TypeVar aliases this way.
"""
from typing import TypeVar, TypeAlias

T = TypeVar('T')

Series: TypeAlias = T
PersistentSeries: TypeAlias = T
