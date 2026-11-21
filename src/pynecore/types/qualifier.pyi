"""
The qualifier markers are transparent aliases for type checkers: ``Simple[T]``
and ``Const[T]`` ARE ``T``, as they are at run time.

The old ``TypeAlias`` syntax (not PEP 695 ``type``) is kept deliberately:
PyCharm resolves unbound-TypeVar aliases this way.
"""
from typing import TypeVar, TypeAlias

T = TypeVar('T')

Simple: TypeAlias = T
"""
A parameter Pine declares ``simple``: its argument is known before the first bar.

Only PyneComp writes it, and only in an overload group whose implementations
differ by nothing but this qualifier, so that they stay distinct and the
implementation Pine selects for the call's argument is the one that runs.
Everywhere else it behaves as a plain ``T``.
"""

Const: TypeAlias = T
"""
A parameter Pine declares ``const``: its argument is known at compile time.

Only PyneComp writes it, and only in an overload group whose implementations
differ by nothing but this qualifier, so that they stay distinct and the
implementation Pine selects for the call's argument is the one that runs.
Everywhere else it behaves as a plain ``T``.
"""
