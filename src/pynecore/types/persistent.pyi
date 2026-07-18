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
