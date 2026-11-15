"""
The user errors the runtime raises from its own frames carry a code: the call
arguments of a builtin (R0050), a series subscript (R0051), an argument out of its
range (R0052), the export machinery (R0053), an unrewritten ``request.security``
(R0054), and the existing codes the remaining places took over.
"""
import math

import pytest

from pynecore import errorcodes
from pynecore.core import pine_export
from pynecore.core.pine_udt import udt_copy
from pynecore.core.rolling_sum import PYTHON_SUM_MACHINE
from pynecore.core.script import input as pyne_input
from pynecore.core.series import SeriesImpl
from pynecore.lib import array, box, label, line, request
from pynecore.lib import math as lib_math
from pynecore.types.color import Color
from pynecore.types.na import NA
from pynecore.types.table import Table


def __test_helper_series(*values: float) -> SeriesImpl:
    """A float series holding ``values``, oldest first."""
    series = SeriesImpl(10, na_value=math.nan)
    for value in values:
        series._push(value)  # noqa
    return series


def __test_call_arguments_are_r0050__():
    """Too many positionals and a positional/keyword clash of a builtin"""
    with pytest.raises(errorcodes.CallArgumentsError) as raised:
        box.new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24)
    assert isinstance(raised.value, TypeError) and raised.value.code is errorcodes.CALL_ARGUMENTS
    assert raised.value.params['function'] == 'box.new'
    with pytest.raises(errorcodes.CallArgumentsError) as raised:
        label.new(1, 2, x=1)
    assert str(raised.value) == "`label.new`: `x` is given both by position and by name."
    with pytest.raises(errorcodes.CallArgumentsError) as raised:
        line.new(1, 2, 3, 4, x1=1)
    assert raised.value.params['function'] == 'line.new'
    with pytest.raises(errorcodes.CallArgumentsError) as raised:
        pyne_input.float(1.0, 't', 0.0, minval=1.0, _id='t')
    assert raised.value.params['function'] == 'input.float'


def __test_copy_misuse_is_coded__():
    """A copy of an uncopyable drawing is R0039, field overrides on a builtin R0050"""
    with pytest.raises(errorcodes.UnknownMethodAtRunError) as raised:
        udt_copy(Table.__new__(Table))
    assert isinstance(raised.value, TypeError)
    assert raised.value.params == {'type': 'Table', 'method': 'copy'}
    with pytest.raises(errorcodes.CallArgumentsError):
        udt_copy(NA(Table), x=1)


def __test_series_subscript_is_r0051__():
    """A non-number index, a negative or stepped slice and a slice past the history"""
    series = __test_helper_series(1.0, 2.0, 3.0)
    with pytest.raises(errorcodes.SeriesSubscriptError) as raised:
        series['x']  # noqa
    assert isinstance(raised.value, TypeError)
    assert str(raised.value) == "`x` is not a valid series subscript: it is neither a number nor " \
                                "a slice."
    with pytest.raises(errorcodes.SeriesSubscriptError) as raised:
        series[-1:2]
    assert isinstance(raised.value, IndexError) and raised.value.params['subscript'] == '-1:2'
    with pytest.raises(errorcodes.SeriesSubscriptError) as raised:
        series[0:2:2]
    assert isinstance(raised.value, ValueError) and raised.value.params['subscript'] == '0:2:2'
    with pytest.raises(errorcodes.SeriesSubscriptError) as raised:
        series[0:5]
    assert raised.value.params['why'] == "it reaches past the stored history"
    view = series[0:2]
    assert view[1] == 2.0
    for index in ('x', -1, 2):
        with pytest.raises(errorcodes.SeriesSubscriptError):
            view[index]  # noqa
    with pytest.raises(errorcodes.SeriesSubscriptError) as raised:
        PYTHON_SUM_MACHINE()._oldest(0, 3)  # noqa
    assert raised.value.params['subscript'] == ':3'


def __test_argument_out_of_range_is_r0052__():
    """max_bars_back above its limit and a color alpha outside 0..255"""
    series = __test_helper_series(1.0)
    with pytest.raises(errorcodes.ArgumentOutOfRangeError) as raised:
        series.max_bars_back = SeriesImpl.MAXIMUM_MAX_BARS_BACK + 1
    assert isinstance(raised.value, ValueError)
    assert str(raised.value) == "`max_bars_back`: `num` must be between 1 and 5000, but it is 5001."
    with pytest.raises(errorcodes.ArgumentOutOfRangeError):
        PYTHON_SUM_MACHINE()._resize_values(SeriesImpl.MAXIMUM_MAX_BARS_BACK + 1)  # noqa
    color = Color('#000000')
    with pytest.raises(errorcodes.ArgumentOutOfRangeError) as raised:
        color.a = 256
    assert raised.value.params == {'function': 'Color.a', 'parameter': 'alpha', 'low': 0,
                                   'high': 255, 'value': 256}


def __test_export_misuse_is_r0053__():
    """An Exported proxy called before its function, and @export without a proxy"""
    proxy = pine_export.Exported()
    with pytest.raises(errorcodes.ExportMisuseError) as raised:
        proxy()
    assert isinstance(raised.value, ValueError)
    with pytest.raises(errorcodes.ExportMisuseError) as raised:
        pine_export.export(lambda: None, func_globals={})
    assert raised.value.params['name'] == '<lambda>'


def __test_a_direct_security_call_is_r0054__():
    """request.security reached as a function, not through the transform's rewrite"""
    with pytest.raises(errorcodes.SecurityNotRewrittenError) as raised:
        request.security('A:B', '1D', 1.0)
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value) == "`request.security` runs only where the transform rewrote it; " \
                                "call it directly in a `@pyne` module."
    with pytest.raises(errorcodes.SecurityNotRewrittenError) as raised:
        request.security_lower_tf('A:B', '1', 1.0)
    assert raised.value.params == {'function': 'request.security_lower_tf'}


def __test_wrong_operand_types_are_r0032__():
    """An array.new_* initial value of another type keeps its AssertionError base"""
    with pytest.raises(errorcodes.WrongOperandTypeError) as raised:
        array.new_box(2, 1.0)
    assert isinstance(raised.value, AssertionError)
    assert str(raised.value) == "`array.new_box` cannot take a float as its initial value."
    with pytest.raises(errorcodes.WrongOperandTypeError):
        array.new_string(1, 1.0)


def __test_na_object_misuse_is_coded__():
    """int() of an na object is R0002; a write, a loop and float() on one are R0016"""
    na_table = NA(Table)
    with pytest.raises(errorcodes.NaToIntegerError) as raised:
        int(na_table)  # type: ignore[call-overload]
    assert isinstance(raised.value, TypeError)
    with pytest.raises(errorcodes.NaObjectError) as raised:
        na_table.x = 1
    assert isinstance(raised.value, AttributeError) and raised.value.params == {'name': 'x'}
    with pytest.raises(errorcodes.NaObjectError) as raised:
        iter(na_table)
    assert isinstance(raised.value, TypeError)
    assert str(raised.value) == "`for` cannot take an `na` object."
    with pytest.raises(errorcodes.NaObjectError) as raised:
        math.sqrt(na_table)  # type: ignore[arg-type]
    assert raised.value.params == {'name': 'float'}


def __test_math_of_no_numbers_is_r0011__():
    """math.avg/max/min with no argument keep their AssertionError base"""
    for function in (lib_math.avg, lib_math.max, lib_math.min):
        with pytest.raises(errorcodes.EmptyCollectionError) as raised:
            function()
        assert isinstance(raised.value, AssertionError)
        assert raised.value.params == {'function': f"math.{function.__name__}"}
