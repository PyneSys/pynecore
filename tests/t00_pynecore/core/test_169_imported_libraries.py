"""
A run uses the libraries its script imports, and no other.

The library registry is process-wide: it keeps the library of every script the process
has imported. A later script that imports none of them must not run their mains, read
their ``request.security`` contexts or profile their files.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from pynecore.core import profiler
from pynecore.core.script import imported_libraries
from pynecore.core.script_runner import ScriptRunner, import_script
from pynecore.types.ohlcv import OHLCV

COUNTER_SOURCE = '''
calls = []


def hit():
    calls.append(1)
'''

LIBRARY_SOURCE = '''"""
@pyne
"""
from pynecore.lib import script
{imports}


@script.library("{title}")
def main():
    {body}


def {function}(x: float) -> float:
    return x * 2
'''

SCRIPT_SOURCE = '''"""
@pyne
"""
from pynecore.lib import script, close, plot
{imports}


@script.indicator("{title}")
def main():
    plot({expression}, "value")
'''


def __test_helper_library(path: Path, title: str, function: str, *, imports: str = '',
                          body: str = 'pass') -> None:
    path.write_text(LIBRARY_SOURCE.format(title=title, function=function,
                                          imports=imports, body=body), encoding='utf-8')


def __test_helper_script(path: Path, title: str, *, imports: str = '',
                         expression: str = 'close') -> Path:
    path.write_text(SCRIPT_SOURCE.format(title=title, imports=imports,
                                         expression=expression), encoding='utf-8')
    return path


def __test_helper_bars(count: int = 10) -> list[OHLCV]:
    return [OHLCV(timestamp=1735689600000 + index * 300000, open=1.0, high=2.0, low=0.5,
                  close=1.0, volume=1.0) for index in range(count)]


def __test_helper_titles(module) -> list[str]:
    return [title for title, _main in imported_libraries(module)]


def __test_a_later_script_runs_no_foreign_library__(tmp_path, syminfo):
    """ The library of an earlier script does not run on the bars of a later one """
    (tmp_path / 't169_counter.py').write_text(COUNTER_SOURCE, encoding='utf-8')
    __test_helper_library(tmp_path / 't169_counted_lib.py', 'T169 Counted', 'counted',
                          imports='import t169_counter', body='t169_counter.hit()')
    first = __test_helper_script(tmp_path / 't169_first.py', 'T169 First',
                                 imports='from t169_counted_lib import counted',
                                 expression='counted(close)')
    second = __test_helper_script(tmp_path / 't169_second.py', 'T169 Second')
    bars = __test_helper_bars()

    ScriptRunner(first, bars, syminfo).run()
    counter = sys.modules['t169_counter']
    assert len(counter.calls) == len(bars), "the imported library runs on every bar"

    counter.calls.clear()
    ScriptRunner(second, bars, syminfo).run()
    assert counter.calls == [], "a script that does not import the library must not run it"


def __test_libraries_reached_through_imports__(tmp_path):
    """ A library imported through a library or from a package counts, in registration order """
    __test_helper_library(tmp_path / 't169_inner_lib.py', 'T169 Inner', 'inner')
    __test_helper_library(tmp_path / 't169_outer_lib.py', 'T169 Outer', 'outer',
                          imports='from t169_inner_lib import inner')
    package = tmp_path / 't169_pkg'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    __test_helper_library(package / 'packed_lib.py', 'T169 Packed', 'packed')
    __test_helper_library(tmp_path / 't169_unused_lib.py', 'T169 Unused', 'unused')
    script = __test_helper_script(
        tmp_path / 't169_reaching.py', 'T169 Reaching',
        imports='from t169_outer_lib import outer\nfrom t169_pkg import packed_lib',
        expression='packed_lib.packed(outer(close))')

    import_script(__test_helper_script(tmp_path / 't169_unused.py', 'T169 Unused Script',
                                       imports='from t169_unused_lib import unused'))
    module = import_script(script)

    assert __test_helper_titles(module) == ['T169 Inner', 'T169 Outer', 'T169 Packed']


def __test_a_reimported_library_leaves_its_old_main_out__(tmp_path):
    """ Only the main of the library module the import reaches now counts """
    __test_helper_library(tmp_path / 't169_reloaded_lib.py', 'T169 Reloaded', 'reloaded')
    script = __test_helper_script(tmp_path / 't169_reloading.py', 'T169 Reloading',
                                  imports='from t169_reloaded_lib import reloaded',
                                  expression='reloaded(close)')
    import_script(script)
    sys.modules.pop('t169_reloaded_lib')
    module = import_script(script)

    libraries = imported_libraries(module)
    assert [title for title, _main in libraries] == ['T169 Reloaded']
    _title, main = libraries[0]
    assert main.__globals__ is vars(sys.modules['t169_reloaded_lib'])


@pytest.mark.aot_unsupported("the profiler measures CPython code objects")
def __test_profile_skips_a_foreign_library__(tmp_path, syminfo):
    """ A later script's profile lists no library of an earlier one, even when its file is gone """
    if sys.version_info < (3, 13):
        pytest.skip("profiling needs Python 3.13+")
    if profiler._native_profiler is None:  # noqa
        if os.environ.get('PYNE_REQUIRE_NATIVE_PROFILER'):
            pytest.fail("the compiled profiler extension is missing from this build")
        pytest.skip("the compiled profiler extension is not built")

    library = tmp_path / 't169_vanished_lib.py'
    __test_helper_library(library, 'T169 Vanished', 'vanished')
    import_script(__test_helper_script(tmp_path / 't169_earlier.py', 'T169 Earlier',
                                       imports='from t169_vanished_lib import vanished'))
    library.unlink()
    later = __test_helper_script(tmp_path / 't169_later.py', 'T169 Later')
    profile_path = tmp_path / 'profile.json'

    ScriptRunner(later, __test_helper_bars(), syminfo, profile_path=profile_path).run()

    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    assert [(file['role'], Path(file['path'])) for file in profile['files']] == \
        [('script', later.resolve())]
