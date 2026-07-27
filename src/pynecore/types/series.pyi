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
"""
A Pine ``series``: the current bar's value, with its history kept.

Annotate a variable as ``Series[float]`` and you can look back with ``[n]`` —
``close[1]`` is the previous bar's close, ``my_ema[3]`` its value three bars
ago. Everywhere else it behaves as a plain ``float``.

Type checkers see ``Series[T]`` as plain ``T``: Python has no intersection
type, so the scalar side is typed exactly and the ``[n]`` indexing is allowed
separately.
"""

PersistentSeries: TypeAlias = T
"""
A ``Series`` that also survives the bar: the value carries over to the next
bar instead of being re-initialised — Pine's ``var`` applied to a series.

Equivalent to declaring the variable both ``Persistent`` and ``Series``, which
is exactly what the AST transformer splits it into.
"""
