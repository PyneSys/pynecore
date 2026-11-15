"""
@pyne

``str.format`` and ``log.*`` format int, float, bool and string values and arrays of them,
and reject every other argument type instead of printing a Python object representation.
"""
import pytest

from pynecore.errorcodes import WrongOperandTypeError
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
    with pytest.raises(WrongOperandTypeError) as raised:
        string.format("{0}", value)
    assert raised.value.params == {'operation': 'str.format',
                                   'operand': f"a {type_name} as argument {{0}}"}
    with pytest.raises(WrongOperandTypeError) as raised:
        string.format("{0}", 1, value)
    assert raised.value.params['operand'] == f"a {type_name} as argument {{1}}"


def __test_array_of_unformattable_elements_is_rejected__():
    """an array of colors is not an array of a formattable type"""
    with pytest.raises(WrongOperandTypeError) as raised:
        string.format("{0}", array.from_items(color.red))
    assert raised.value.params['operand'] == "an array of Color as argument {0}"


def __test_log_rejects_at_the_call__():
    """log.* raises at the call site, not inside the logging handler that formats lazily"""
    for function, args in (('log.info', matrix.new(1, 1, -3)), ('log.warning', color.red),
                           ('log.error', map.new())):
        with pytest.raises(WrongOperandTypeError) as raised:
            getattr(log, function.removeprefix('log.'))("{0}", args)
        assert raised.value.params['operation'] == function
