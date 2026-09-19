"""
Name anchors for the call-inlining pass.

:class:`~pynecore.transformers.call_inline.CallInlineTransformer` copies the
body of a handful of trivial Pine builtins into the call site as one
expression. The copied body keeps referring to whatever its own module
referred to -- ``na_float``, ``builtins``, ``math``, ``isinstance`` -- and
those names mean something else in a script: ``math`` is ``pynecore.lib.math``
there, and ``BuiltinShadowTransformer`` lets a script rebind ``abs`` or
``float`` outright.

This module is the one place the inlined bodies resolve through. Every object
a derivable body can reach is re-exported here under a fixed name, the
transformer imports the ones it needs under a collision-safe alias, and the
mapping is checked object-by-object by the pass's own tests, so an inlined
body can never bind a different object than the function it was copied from.
"""
import builtins
import math

from . import fdlibm, pine_math
# noinspection PyProtectedMember
from ..lib.array import _na_element as array_na_element
# noinspection PyProtectedMember
from ..lib.math import _na_of_operands as math_na_of_operands
from ..types.na import NA, na_float, na_int

__all__ = [
    'NA', 'na_float', 'na_int',
    'py_builtins', 'py_math', 'pine_math', 'fdlibm',
    'py_float', 'py_int', 'py_isinstance', 'py_len',
    'array_na_element', 'math_na_of_operands',
]

#: The stdlib modules the copied bodies call into, under names a script cannot
#: mean by accident (a script's ``math`` is ``pynecore.lib.math``)
py_builtins = builtins
py_math = math

#: The builtins the copied bodies call. A script may rebind any of these
#: (see ``transformers/builtin_shadow.py``), so the bodies must not go through
#: the call site's own name for them.
py_float = builtins.float
py_int = builtins.int
py_isinstance = builtins.isinstance
py_len = builtins.len
