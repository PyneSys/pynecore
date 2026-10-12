"""
@pyne

``str.format`` and ``log.*`` format int, float, bool and string values and arrays of them,
and reject every other argument type instead of printing a Python object representation.
"""
import re

import pytest

from pynecore.lib import array, color, log, map, matrix, string


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_formattable_arguments_pass__():
    """scalars, na and arrays of them are formatted"""
    assert string.format("{0} {1} {2} {3}", 1, 2.5, True, "s") == "1 2.5 true s"
    assert string.format("{0}", array.from_items(1.5, 2.5)) == "[1.5, 2.5]"
    assert string.format("{0}", array.from_items("a", "b")) == "[a, b]"
    assert string.format("{0}", array.new_float(0)) == "[]"
    assert string.format("{0}", array.new_float(1)) == "[NaN]"


@pytest.mark.parametrize("value, type_name", [
    (matrix.new(2, 2, 1.5), "Matrix"),
    (color.red, "Color"),
    (map.new(), "dict"),
])
def __test_unformattable_argument_is_rejected__(value, type_name):
    """a matrix, color or map argument raises, even when no placeholder reads it"""
    with pytest.raises(TypeError, match=re.escape(f"argument {{0}} is a {type_name}")):
        string.format("{0}", value)
    with pytest.raises(TypeError, match=re.escape(f"argument {{1}} is a {type_name}")):
        string.format("{0}", 1, value)


def __test_array_of_unformattable_elements_is_rejected__():
    """an array of colors is not an array of a formattable type"""
    with pytest.raises(TypeError, match=re.escape("argument {0} is an array of Color")):
        string.format("{0}", array.from_items(color.red))


def __test_log_rejects_at_the_call__():
    """log.* raises at the call site, not inside the logging handler that formats lazily"""
    with pytest.raises(TypeError, match=re.escape("argument {0} is a Matrix")):
        log.info("{0}", matrix.new(1, 1, -3))
    with pytest.raises(TypeError, match=re.escape("argument {0} is a Color")):
        log.warning("{0}", color.red)
    with pytest.raises(TypeError, match=re.escape("argument {0} is a dict")):
        log.error("{0}", map.new())
