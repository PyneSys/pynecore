"""Dispatch table of ``udt_copy``, the runtime target of Pine's method-form ``.copy()``.

Deliberately not a ``@pyne`` module: the locally defined dataclass standing in for a
user UDT must not be run through the Pyne AST transform.
"""
from dataclasses import dataclass

import pytest

from pynecore.core import viz
from pynecore.core.pine_udt import udt_copy
from pynecore.lib import box, label, line, matrix
from pynecore.lib import map as map_lib
from pynecore.types.label import Label
from pynecore.types.na import NA


@dataclass(slots=True)
class _Point:
    x: int = 0
    y: int = 0


def __test_udt_copy_dispatches_on_the_runtime_type__():
    """Builtin objects go to their namespace copy, everything else is field-copied.

    The compiler emits ``udt_copy(...)`` for all three shapes of Pine's method-form
    ``.copy()`` — a named variable, a map-value receiver and an array-element
    receiver — with no receiver type attached, so the concrete runtime type is the
    only thing that can select the right copy.
    """
    viz.reset_state()
    try:
        src = label.new(1, 10.0, "A")
        clone = udt_copy(src)
        assert clone is not src
        assert clone.vid != src.vid
        assert list(label._registry) == [src, clone]
        label.set_text(clone, "B")
        assert src.text == "A"

        ln = line.new(1, 1.0, 2, 2.0)
        assert udt_copy(ln) in line._registry
        bx = box.new(1, 2.0, 3, 1.0)
        assert udt_copy(bx) in box._registry

        # Measured on TradingView: copying an na drawing leaves the registry
        # untouched and returns na.
        registry_size = len(label._registry)
        assert udt_copy(NA(Label)) is NA(Label)
        assert len(label._registry) == registry_size

        # Containers are not dataclasses: only the namespace copy can clone them.
        lst = [1, 2, 3]
        lst_copy = udt_copy(lst)
        assert lst_copy == lst and lst_copy is not lst
        dct = map_lib.new()
        map_lib.put(dct, "k", 1.0)
        dct_copy = udt_copy(dct)
        assert dct_copy == dct and dct_copy is not dct
        mx = matrix.new(2, 2, 0.0)
        mx_copy = udt_copy(mx)
        matrix.set(mx_copy, 0, 0, 7.0)
        assert matrix.get(mx, 0, 0) == 0.0 and matrix.get(mx_copy, 0, 0) == 7.0

        # A user-defined type stays a plain field copy, including field overrides.
        p = _Point(1, 2)
        q = udt_copy(p)
        assert q is not p and (q.x, q.y) == (1, 2)
        assert udt_copy(p, y=9).y == 9 and p.y == 2

        # Field overrides are not expressible on a builtin: Pine's ``.copy()`` takes
        # no arguments, so this can only be a programming error.
        with pytest.raises(TypeError):
            udt_copy(src, text="C")
    finally:
        viz.reset_state()
