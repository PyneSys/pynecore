"""
Where the run locates an error: the coded error a Python exception of the
script's code stands for, the line of the script, the Pine line a sourcemap
gives, the sourcemap a compile writes, and the report a ``request.security``
child sends when a coded error stops it.
"""
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from pynecore import errorcodes
from pynecore.core.error_location import (INFINITY_TO_INT, NAN_TO_INT, coded, located,
                                          pine_line_for)
from pynecore.core.errors import SiteLoc
from pynecore.core.security_process import _failure_report
from pynecore.pynesys.api import CompileResponse
from pynecore.pynesys.compiler import PyneComp


def __test_helper_raised(source: str, path: Path) -> BaseException:
    """The exception ``source`` raises when it runs as the file ``path``, the
    module of a library imported as ``signals``."""
    path.write_text(source, encoding='utf-8')
    try:
        exec(compile(source, str(path), 'exec'), {'__name__': 'signals'})
    except Exception as e:  # noqa: BLE001 - the test inspects any exception
        return e
    raise AssertionError("the source did not raise")


def __test_python_exceptions_get_their_codes__():
    """The four Python exceptions with a code, by type and CPython's message"""
    assert type(coded(ZeroDivisionError("float division by zero"))) is errorcodes.DivisionByZeroError
    unbound = UnboundLocalError("cannot access local variable 'x' where it is not associated "
                                "with a value")
    error = coded(unbound)
    assert type(error) is errorcodes.UnboundVariableError and error.params == {'name': 'x'}
    assert type(coded(OverflowError(INFINITY_TO_INT))) is errorcodes.InfinityToIntegerError
    assert type(coded(ValueError(NAN_TO_INT))) is errorcodes.NaToIntegerError
    assert coded(OverflowError("math range error")) is None
    assert coded(ValueError("bad")) is None
    assert coded(KeyError("k")) is None


@pytest.mark.parametrize(('source', 'message'), [
    ('int(float("inf"))\n', INFINITY_TO_INT),
    ('int(float("nan"))\n', NAN_TO_INT),
    ('round(float("inf"))\n', INFINITY_TO_INT),
    ('round(float("nan"))\n', NAN_TO_INT),
])
def __test_cpython_messages_the_codes_are_read_from__(tmp_path: Path, source: str,
                                                       message: str):
    """CPython's messages of ``int()`` / ``round()`` of inf and NaN, which the codes match"""
    assert str(__test_helper_raised(source, tmp_path / 'script.py')) == message


def __test_located_fills_in_the_bar_and_the_scripts_line__(tmp_path: Path):
    """A Python exception becomes its coded error at the script's line and the bar"""
    script = tmp_path / 'script.py'
    exc = __test_helper_raised('x = 1\ny = 2 // 0\n', script)
    error = located(exc, 7, 1704164640000)
    assert type(error) is errorcodes.DivisionByZeroError
    assert (error.bar_index, error.bar_time) == (7, 1704164640000)
    assert error.at == SiteLoc('signals', str(script), 2, 5, None)
    assert error.__traceback__ is exc.__traceback__
    other = KeyError('k')
    assert located(other, 7, 0) is other


def __test_located_keeps_a_location_the_error_already_has__():
    """An error a security context located keeps the context's line"""
    error = errorcodes.DivisionByZeroError()
    error.at = SiteLoc('main', 'child.py', 4, 1)
    assert located(error, 3, 0) is error and error.at == SiteLoc('main', 'child.py', 4, 1)
    assert error.bar_index == 3


def __test_helper_compiled(tmp_path: Path, **stored) -> Path:
    """A compiled ``script.py`` with its ``.pine`` source and a map of ``stored``."""
    py = tmp_path / 'script.py'
    py.write_text('header\nheader\na = 1\nb = 2\n\nc = 3\n', encoding='utf-8')
    (tmp_path / 'script.pine').write_text('//@version=6\n', encoding='utf-8')
    data = {'version': 1, 'mappings': [[3, 4], [4, 5], [6, 9]], **stored}
    (tmp_path / 'script.py.map').write_text(json.dumps(data), encoding='utf-8')
    return py


def __test_the_pine_line_is_the_last_mapping_up_to_the_line__(tmp_path: Path):
    """Forward fill, and no Pine line before the first mapping"""
    py = str(__test_helper_compiled(tmp_path))
    assert [pine_line_for(py, n) for n in range(1, 8)] == [None, None, 4, 5, 5, 9, 9]


def __test_a_map_is_used_only_when_it_holds__(tmp_path: Path):
    """A stale, converted or malformed map, or one without its .pine, gives no line"""
    py = __test_helper_compiled(tmp_path, py_sha256='0' * 64)
    assert pine_line_for(str(py), 3) is None
    digest = hashlib.sha256(py.read_bytes()).hexdigest()
    py = __test_helper_compiled(tmp_path, py_sha256=digest)
    assert pine_line_for(str(py), 3) == 4
    py = __test_helper_compiled(tmp_path, converted_source=True)
    assert pine_line_for(str(py), 3) is None
    py = __test_helper_compiled(tmp_path, mappings=[None])
    assert pine_line_for(str(py), 3) is None
    py = __test_helper_compiled(tmp_path)
    (tmp_path / 'script.pine').unlink()
    assert pine_line_for(str(py), 3) is None


def __test_a_compile_writes_its_sourcemap_stamped__(tmp_path: Path):
    """The map of a compile, stamped as PyneIDE stamps it; none from the server removes it"""
    py = tmp_path / 'script.py'
    py.write_text('code\n', encoding='utf-8')
    response = CompileResponse(success=True, compiled_code='code\n', sourcemap={
        'version': 1, 'pine_version': 5, 'mappings': [[1, 2]]})
    PyneComp.write_sourcemap(py, response, '//@version=5\n', strict=True)
    stored = json.loads((tmp_path / 'script.py.map').read_text(encoding='utf-8'))
    assert stored == {
        'version': 1, 'pine_version': 5, 'mappings': [[1, 2]], 'strict': True,
        'py_sha256': hashlib.sha256(b'code\n').hexdigest(),
        'pine_sha256': hashlib.sha256(b'//@version=5\n').hexdigest(),
        'converted_source': True,
    }
    PyneComp.write_sourcemap(py, CompileResponse(success=True, compiled_code='code\n'), '', False)
    assert not (tmp_path / 'script.py.map').exists()


def __test_a_child_stopped_by_a_coded_error_reports_the_error__(tmp_path: Path):
    """The error itself, located at the script's line, without the child's bar"""
    script = tmp_path / 'script.py'
    exc = __test_helper_raised('x = 1 // 0\n', script)
    report = _failure_report(exc)
    assert type(report) is errorcodes.DivisionByZeroError
    assert report.at == SiteLoc('signals', str(script), 1, 5, None) and report.bar_index is None
    assert pickle.loads(pickle.dumps(report)).report() == report.report()
    text = _failure_report(__test_helper_raised('{}["k"]\n', script))
    assert isinstance(text, str) and text.startswith("KeyError: 'k' (at ")


def __test_a_plain_subscript_of_the_script_keeps_its_python_error__(tmp_path: Path):
    """An IndexError at a real (not zero-width) position is the script's own"""
    script = tmp_path / 'script.py'
    source = ('__pyne_inline_sites__ = ((2, 4, "array.get", (("n", "a"), ("c", 5))),)\n'
              'a = [1]\n'
              'b = a[5]\n')
    exc = __test_helper_raised(source, script)
    assert isinstance(exc, IndexError) and coded(exc) is None
