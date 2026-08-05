"""Dispatch table of ``udt_copy``, the runtime target of Pine's method-form ``.copy()``.

Deliberately not a ``@pyne`` module: the locally defined dataclass standing in for a
user UDT must not be run through the Pyne AST transform.
"""
from dataclasses import dataclass

import pytest

from pynecore.core import viz
from pynecore.core.pine_udt import udt_copy
from pynecore.lib import (array, box, chart, color, label, line, linefill, matrix, polyline,
                          position, table)
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

        # An na carries no fields either, so overrides are refused there too rather
        # than dropped into a plausible-looking na.
        with pytest.raises(TypeError):
            udt_copy(NA(Label), text="C")
    finally:
        viz.reset_state()


def __test_container_copy_shares_its_drawing_handles__():
    """Copying a container copies the container, never the drawings it holds.

    Measured on TradingView: after ``mx2 = mx.copy()``, setting the text through
    ``mx2.get(0, 0)`` changes the ORIGINAL label, and ``array.copy()`` behaves the
    same. Cloning the elements would hand back drawings with duplicate vids that no
    registry knows and the chart never shows.
    """
    viz.reset_state()
    try:
        src = label.new(1, 10.0, "A")

        mx = matrix.new(1, 1, src)
        mx_copy = matrix.copy(mx)
        assert matrix.get(mx_copy, 0, 0) is src
        label.set_text(matrix.get(mx_copy, 0, 0), "B")
        assert src.text == "B"

        arr = array.new_label()
        array.push(arr, src)
        assert array.get(array.copy(arr), 0) is src
    finally:
        viz.reset_state()


def __test_matrix_copy_shares_every_reference_element__():
    """``matrix.copy()`` is shallow for chart points and UDT instances too.

    Pine's matrix.copy() gives back new row storage holding the very same elements, so
    a mutation through the copy is visible on the original — the same contract
    ``array.copy()`` already honours. Only the storage must be independent: writing a
    whole cell of the copy must leave the original cell alone.
    """
    point = chart.point.new(0, 1, 1.0)
    udt = _Point(1, 2)

    mx = matrix.new(1, 2, point)
    matrix.set(mx, 0, 1, udt)
    mx_copy = matrix.copy(mx)

    assert matrix.get(mx_copy, 0, 0) is point
    assert matrix.get(mx_copy, 0, 1) is udt

    matrix.get(mx_copy, 0, 0).price = 9.0
    matrix.get(mx_copy, 0, 1).x = 9
    assert point.price == 9.0
    assert udt.x == 9

    # The storage itself is independent: replacing a cell of the copy is not seen
    # by the original.
    matrix.set(mx_copy, 0, 0, None)
    assert matrix.get(mx, 0, 0) is point


def __test_udt_copy_refuses_the_drawings_pine_cannot_copy__():
    """Copying a linefill, polyline or table raises instead of orphaning it.

    Measured on TradingView: these three receivers are rejected at compile time, in
    the method and the namespace-function form alike, so the compiler stops every
    receiver it can type. A container element it cannot type still arrives here, and
    they are dataclasses — a field copy would succeed and hand back a drawing with a
    duplicate vid that is in no registry and never reaches the chart.
    """
    viz.reset_state()
    try:
        l1 = line.new(1, 1.0, 2, 2.0)
        l2 = line.new(1, 2.0, 2, 3.0)
        lf = linefill.new(l1, l2, color.red)
        pl = polyline.new([chart.point.new(0, 1, 1.0), chart.point.new(0, 2, 2.0)])
        tb = table.new(position.top_right, 1, 1)

        for drawing in (lf, pl, tb):
            with pytest.raises(TypeError):
                udt_copy(drawing)
    finally:
        viz.reset_state()
