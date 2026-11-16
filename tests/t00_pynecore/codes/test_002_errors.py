"""
The coded errors: the generated exception classes of ``pynecore.errorcodes``, their
messages and pickling, :func:`~pynecore.core.errors.format_arg` over the vectors
the Rust side runs too, and the report line of a located error.
"""
import builtins
import importlib
import pickle
import re
import tomllib
from pathlib import Path

import pytest

from pynecore import errorcodes
from pynecore.core.errors import PyneError, SiteLoc, bar_time_text, format_arg
from pynecore.types.na import NA

_REPO_ROOT = Path(__file__).resolve().parents[3]

with (_REPO_ROOT / 'scripts' / 'errorcodes' / 'registry.toml').open('rb') as _f:
    _REGISTRY = tomllib.load(_f)


def _vectors() -> list[tuple[float, str]]:
    with (_REPO_ROOT / 'scripts' / 'errorcodes' / 'format_arg.toml').open('rb') as f:
        return [(float(row['value']), row['text']) for row in tomllib.load(f)['case']]


@pytest.mark.parametrize(('value', 'text'), _vectors())
def __test_format_arg_writes_the_shared_vectors__(value: float, text: str):
    """A number is written as the Rust ``format_arg`` writes it"""
    assert format_arg(value) == text


def __test_format_arg_writes_pine_values__():
    """na, bools, ints and strings as a message shows them"""
    assert [format_arg(v) for v in (NA(float), True, False, 3, -12, 'abc')] \
        == ['na', 'true', 'false', '3', '-12', 'abc']


def _args(code: str) -> dict[str, object]:
    """One value per parameter of ``code``, under its keyword argument."""
    return {(p + '_' if p in ('import', 'type') else p): f"<{p}>" for p in errorcodes.ALL[code].params}


@pytest.mark.parametrize('code', sorted(errorcodes.EXCEPTIONS))
def __test_every_class_derives_from_its_registry_bases_and_pickles__(code: str):
    """The class of a code is a PyneError and every built-in its row names, and a
    located error survives pickling whole"""
    cls = errorcodes.EXCEPTIONS[code]
    assert cls.code.code == code and issubclass(cls, PyneError)
    python = _REGISTRY[code].get('python', [])
    for name in [python] if isinstance(python, str) else python:
        module, _, attr = name.rpartition('.')
        base = getattr(importlib.import_module(module) if module else builtins, attr)
        assert issubclass(cls, base), name
    error = cls(**_args(code))
    error.bar_index, error.bar_time, error.in_security = 3, 1704164640000, True
    error.at = SiteLoc('main', 'script.py', 4, 2, 9)
    back = pickle.loads(pickle.dumps(error))
    assert type(back) is cls and back.params == error.params
    assert back.report() == error.report() and str(back) == str(error)


def __test_a_message_formats_its_parameters__():
    """The message is the registry's, with the values formatted"""
    error = errorcodes.IndexOutOfRangeError(function='array.get', index=5.0, size=3)
    assert isinstance(error, IndexError) and error.code is errorcodes.INDEX_OUT_OF_RANGE
    assert str(error) == '`array.get`: index 5 is out of range for a size of 3.'
    assert errorcodes.AssertionFailedError(message='').message == 'Assertion failed.'
    assert isinstance(errorcodes.InvalidRegexError(pattern='(', reason='x'), re.error)


def __test_a_keyword_parameter_takes_a_trailing_underscore__():
    """A parameter named as a Python keyword is passed with a trailing underscore"""
    error = errorcodes.EdgeImportError(import_='`import os`')
    assert error.params == {'import': '`import os`'}


@pytest.mark.parametrize(('ms', 'text'), [
    (1704164640000, '2024-01-02 03:04 UTC'),
    (1704164645000, '2024-01-02 03:04:05 UTC'),
    (1704164645250, '2024-01-02 03:04:05.250 UTC'),
    (0, '1970-01-01 00:00 UTC'),
])
def __test_bar_time_text_writes_seconds_only_when_there_are__(ms: int, text: str):
    """A bar time with seconds and milliseconds only when it has them"""
    assert bar_time_text(ms) == text


def __test_report_names_the_bar_the_code_and_the_location__():
    """The report line of a run: the Pyne location, the Pine line only for a run of the Pine
    source"""
    error = errorcodes.DivisionByZeroError()
    assert error.report() == 'R0001: Division by zero.'
    error.bar_index, error.bar_time = 7, 1704164640000
    error.at = SiteLoc('main', '/s/script.py', 12, 5)
    assert error.report() == 'bar 7 (2024-01-02 03:04 UTC): R0001 at /s/script.py:12:5: ' \
                             'Division by zero.'
    error.at = SiteLoc('main', '/s/script.py', 12, 5, pine_line=3)
    assert error.report() == 'bar 7 (2024-01-02 03:04 UTC): R0001 at /s/script.py:12:5: ' \
                             'Division by zero.'
    error.in_security = True
    assert error.report(pine=True) == 'bar 7 (2024-01-02 03:04 UTC): R0001 at Pine line 3: ' \
                                      'Division by zero. (in a `request.security` context)'
    error.at = SiteLoc('main', '/s/script.py', 12, 5)
    assert error.report(pine=True) == 'bar 7 (2024-01-02 03:04 UTC): R0001 at /s/script.py:12:5: ' \
                                      'Division by zero. (in a `request.security` context)'
