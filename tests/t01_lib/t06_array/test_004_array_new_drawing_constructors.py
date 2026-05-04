"""
@pyne

Direct unit tests for the typed-array constructors of drawing-object types
(`array.new_box`, `array.new_line`, `array.new_label`, `array.new_linefill`).

These do not go through the Pine runner; they verify the four functions return
`list[T]` of the requested size, matching the `new_bool`/`new_int`/`new_float`
pattern. Pine's `array.new<box>(size, initial_value)` semantics is "create an
array of N box references", not "create one box".
"""
from pynecore.lib import array
from pynecore.types.box import Box
from pynecore.types.line import Line
from pynecore.types.label import Label
from pynecore.types.linefill import LineFill
from pynecore.types.na import NA


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_new_box_zero_arg__():
    """ array.new_box() returns an empty list """
    result = array.new_box()
    assert isinstance(result, list)
    assert result == []


def __test_new_box_with_size__():
    """ array.new_box(N) returns N NA(Box) elements """
    result = array.new_box(3)
    assert len(result) == 3
    for el in result:
        assert isinstance(el, NA)


def __test_new_box_with_initial_value__():
    """ array.new_box(N, initial_value) returns N copies of the initial value """
    box = NA(Box)
    result = array.new_box(2, box)
    assert len(result) == 2
    assert all(el is box for el in result)


def __test_new_line_zero_arg__():
    """ array.new_line() returns an empty list """
    assert array.new_line() == []


def __test_new_line_with_size__():
    """ array.new_line(N) returns N NA(Line) elements """
    result = array.new_line(5)
    assert len(result) == 5
    for el in result:
        assert isinstance(el, NA)


def __test_new_label_zero_arg__():
    """ array.new_label() returns an empty list """
    assert array.new_label() == []


def __test_new_label_with_size__():
    """ array.new_label(N) returns N NA(Label) elements """
    result = array.new_label(2)
    assert len(result) == 2
    for el in result:
        assert isinstance(el, NA)


def __test_new_linefill_zero_arg__():
    """ array.new_linefill() returns an empty list """
    assert array.new_linefill() == []


def __test_new_linefill_with_size__():
    """ array.new_linefill(N) returns N NA(LineFill) elements """
    result = array.new_linefill(4)
    assert len(result) == 4
    for el in result:
        assert isinstance(el, NA)
