"""
Persistent is a transparent alias for type checkers: ``Persistent[T]`` IS ``T``.

NA is deliberately dropped from the static view (the "plain T" policy): in Pine
semantics any value can be na, so a ``T | NA[T]`` union only produces noise on
every arithmetic use. Runtime (persistent.py) is unchanged — this stub exists
because a bare TypeVar is not subscriptable at runtime.
"""
from typing import TypeVar, TypeAlias

T = TypeVar('T')

Persistent: TypeAlias = T
"""
A variable that keeps its value across bars — Pine's ``var``.

The initialiser runs only on the first bar; every later bar starts from
whatever the previous bar left behind. So ``bar_count: Persistent[int] = 0``
followed by ``bar_count += 1`` counts bars instead of staying at 1.

When the same bar is executed more than once — live ticks, or a strategy with
``calc_on_order_fills`` — the value is rolled back to the bar's committed state
first. Use ``IBPersistent`` if you want it to keep accumulating instead.
"""
