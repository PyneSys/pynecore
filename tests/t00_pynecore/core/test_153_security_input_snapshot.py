"""Security replicas reuse resolved inputs without registering them twice."""
import inspect
import pickle
import sys
from dataclasses import replace

import pytest

from pynecore.core import script as script_mod
from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.script_runner import ScriptRunner, import_script
from pynecore.types.ohlcv import OHLCV


SOURCE = '''"""@pyne"""
from enum import Enum
from pynecore.lib import color, high, input, plot, request, script, syminfo

class Mode(Enum):
    first = 'First'
    second = 'Second'

@script.indicator('Snapshot')
def main(length=input.int(3), mode=input.enum(Mode.first),
         tint=input.color(color.red), src=input.source(high),
         generic=input(color.rgb(1, 2, 3)), *, extra=3.0):
    value = src * length + color.g(tint) + color.r(generic) + extra
    if mode == Mode.second:
        value += 1000
    remote = request.security(syminfo.tickerid, '60', value)
    plot(remote, 'remote')
'''


@pytest.fixture(autouse=True)
def __test_helper_clean_state():
    libraries = list(script_mod._registered_libraries)
    modules = set(sys.modules)
    yield
    script_mod._registered_libraries[:] = libraries
    script_mod.inputs.clear()
    for name in set(sys.modules) - modules:
        module = sys.modules.get(name)
        if (getattr(module, '__file__', None) or '').endswith('/snapshot_lib.py'):
            sys.modules.pop(name, None)


def __test_helper_write(tmp_path, source=SOURCE):
    path = tmp_path / 'snapshot.py'
    path.write_text(source, encoding='utf-8')
    return path


def __test_helper_config(module):
    return {str(module.__file__): module.main.script.resolved_config(module.main)}


def __test_clones_register_each_input_once__(tmp_path, monkeypatch):
    path = __test_helper_write(tmp_path)
    calls = []
    original = script_mod.input.int

    def record(*args, **kwargs):
        calls.append(kwargs.get('_id'))
        return original(*args, **kwargs)

    monkeypatch.setattr(script_mod.input, 'int', record)
    module = import_script(path, inputs={'length': 7})
    clones = [value for name, value in vars(module).items() if name.startswith('__sec_main_')]
    assert clones
    assert calls == ['length']
    assert all(clone.__defaults__ is module.main.__defaults__ for clone in clones)
    assert all(clone.__kwdefaults__ is module.main.__kwdefaults__ for clone in clones)
    assert not script_mod.inputs

    next_path = __test_helper_write(tmp_path, '''"""@pyne"""
from pynecore.lib import script
@script.indicator('No Inputs')
def main():
    pass
''')
    following = import_script(next_path)
    assert following.main.script.inputs == {}


@pytest.mark.parametrize('replica', [False, True])
def __test_failed_import_clears_partial_input_metadata__(tmp_path, replica):
    path = __test_helper_write(tmp_path, '''"""@pyne"""
from pynecore.lib import input, script
@script.indicator('Broken')
def main(first=input.int(1), second=input.int(undefined)):
    pass
''')
    configs = {str(path): script_mod.ScriptConfig(inputs={'first': 3}, settings={})} if replica else None
    with pytest.raises(NameError):
        import_script(path, resolved_configs=configs)
    assert not script_mod.inputs
    assert not script_mod._old_input_values
    assert script_mod._resolved_configs is None


@pytest.mark.parametrize('suffix', ['', '__global__'])
def __test_snapshot_skips_toml_and_transports_value_types__(tmp_path, monkeypatch, suffix):
    source = SOURCE.replace('length', 'length' + suffix)
    if suffix:
        source = source.replace('@pyne', '@pyne edge')
    path = __test_helper_write(tmp_path, source)
    path.with_suffix('.toml').write_text('''[script]
precision = 4
[inputs.length]
value = 5
[inputs.mode]
value = "Second"
[inputs.tint]
value = "#00FF00FF"
[inputs.src]
value = "low"
''', encoding='utf-8')
    original = import_script(path, inputs={'length': 7})
    configs = pickle.loads(pickle.dumps(__test_helper_config(original)))
    assert configs[str(path)].inputs == {
        'length' + suffix: 7.0, 'mode': 'Second', 'tint': '#00FF00FF', 'src': 'low',
        'generic': '#010203FF'}
    path.with_suffix('.toml').write_text('invalid TOML', encoding='utf-8')

    def forbid_load(*args):
        pytest.fail('a resolved security replica must not read script TOML')

    monkeypatch.setattr(script_mod.Script, 'load', forbid_load)
    replica = import_script(path, resolved_configs=configs)
    assert replica.main.script.resolved_config(replica.main) == configs[str(path)]
    assert inspect.signature(replica.main).parameters['mode'].default is replica.Mode.second
    assert replica.main.script.precision == 4
    assert not script_mod.inputs
    assert script_mod._resolved_configs is None


def __test_library_inputs_have_their_own_snapshot__(tmp_path):
    library_path = tmp_path / 'snapshot_lib.py'
    library_path.write_text('''"""@pyne"""
from pynecore.lib import input, script
@script.library('Library')
def main(length=input.int(2)):
    pass
''', encoding='utf-8')
    library_path.with_suffix('.toml').write_text(
        '[script]\noverlay = true\n[inputs.length]\nvalue = 11\n', encoding='utf-8')
    path = __test_helper_write(tmp_path, SOURCE.replace('from enum import Enum',
                                                      'import snapshot_lib\nfrom enum import Enum'))
    original = import_script(path, inputs={'length': 7})
    library = sys.modules['snapshot_lib']
    configs = __test_helper_config(original) | __test_helper_config(library)
    path.with_suffix('.toml').write_text('invalid', encoding='utf-8')
    library_path.with_suffix('.toml').write_text('invalid', encoding='utf-8')
    sys.modules.pop('snapshot_lib')
    replica = import_script(path, resolved_configs=configs)
    imported = sys.modules['snapshot_lib']
    assert inspect.signature(replica.main).parameters['length'].default == 7
    assert inspect.signature(imported.main).parameters['length'].default == 11
    assert imported.main.script.overlay is True
    assert not script_mod.inputs


def __test_security_process_uses_snapshot_after_toml_changes__(tmp_path, syminfo):
    path = __test_helper_write(tmp_path)
    path.with_suffix('.toml').write_text('''[script]
[inputs.length]
value = 5
[inputs.mode]
value = "Second"
[inputs.tint]
value = "#00FF00FF"
[inputs.src]
value = "low"
''', encoding='utf-8')
    start = 1_735_689_600_000
    chart_symbol = replace(syminfo, prefix='EXCH', ticker='SNAP', period='5')
    feed = tmp_path / 'hourly.ohlcv'
    with OHLCVWriter(feed, '60') as writer:
        for hour in range(4):
            writer.write(OHLCV(timestamp=start + hour * 3_600_000,
                               open=10.0, high=20.0, low=2.0, close=10.0, volume=1.0))
    replace(chart_symbol, period='60').save_toml(feed.with_suffix('.toml'))
    bars = [OHLCV(timestamp=start + bar * 300_000, open=10.0, high=20.0,
                  low=2.0, close=10.0, volume=1.0) for bar in range(48)]
    runner = ScriptRunner(path, bars, chart_symbol, inputs={'length': 7},
                          security_data={'60': feed})
    path.with_suffix('.toml').write_text('invalid TOML', encoding='utf-8')
    rows = list(runner.run_iter())
    assert rows[-1][1]['remote'] == 2 * 7 + 255 + 1 + 3 + 1000
    assert path.with_suffix('.toml').read_text() == 'invalid TOML'
