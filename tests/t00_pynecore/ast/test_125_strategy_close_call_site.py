"""Close statements carry static identities through the real transform pipeline."""
import ast
import marshal
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore.core.import_hook import _analyse_tree, _lower_tree
from pynecore.core.script_runner import ScriptRunner
from pynecore.lib import strategy
from pynecore.types.ohlcv import OHLCV


def __test_helper_lower(source: str, path: Path) -> ast.Module:
    tree = _analyse_tree(ast.parse(source), source, path, None)
    return _lower_tree(tree, path, None)[0]


def __test_helper_stamps(tree: ast.Module) -> list[str]:
    return [kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == '_call_site']


def __test_source_stamps_are_stable_unique_and_stateless__():
    source = ('from pynecore.lib.strategy import close as shut, close_all as flatten\n'
              'def helper():\n    shut("L", qty=2); shut("L", qty=3)\n'
              'def main():\n    helper()\n    helper()\n    flatten()\n')
    path = Path('/tmp/static-close-a.py')
    first = __test_helper_lower(source, path)
    stamps = __test_helper_stamps(first)
    assert len(stamps) == len(set(stamps)) == 3
    assert stamps == __test_helper_stamps(__test_helper_lower(source, path))
    assert set(stamps).isdisjoint(__test_helper_stamps(
        __test_helper_lower(source, Path('/tmp/static-close-b.py'))))
    dump = ast.unparse(first)
    assert '__bind_' not in dump
    assert '__resolve_slot' not in dump
    assert '_call_site' not in ast.unparse(_analyse_tree(ast.parse(source), source, path, None))


def __test_shadowed_library_parameter_receives_no_private_keyword__():
    source = ('from pynecore import lib\n'
              'def helper(lib):\n    lib.strategy.close("L")\n')
    assert not __test_helper_stamps(__test_helper_lower(source, Path('/tmp/close-shadow.py')))


def __test_bytecode_keeps_baked_stamps_without_runtime_registration__(monkeypatch):
    source = ('from pynecore.lib import strategy\n'
              'def main():\n    strategy.close("L")\n    strategy.close_all()\n')
    tree = __test_helper_lower(source, Path('/tmp/close-cached.py'))
    code = marshal.loads(marshal.dumps(compile(tree, '/tmp/close-cached.py', 'exec')))
    calls = []
    monkeypatch.setattr(strategy, 'close', lambda *args, **kwargs: calls.append(kwargs['_call_site']))
    monkeypatch.setattr(strategy, 'close_all', lambda *args, **kwargs: calls.append(kwargs['_call_site']))
    namespace = {'__name__': 'close_cached_probe'}
    exec(code, namespace)
    namespace['main']()
    namespace['main']()
    stamps = __test_helper_stamps(tree)
    assert calls == stamps + stamps


def __test_helper_run(tmp_path, syminfo, body: str, preamble: str = '') -> list[float]:
    path = tmp_path / 'strategy.py'
    path.write_text('"""@pyne"""\nfrom pynecore.lib import script, strategy, bar_index\n'
                    + preamble
                    + '\n@script.strategy("static close", initial_capital=100000)\n'
                    'def main():\n'
                    '    if bar_index == 0:\n        strategy.entry("L", strategy.long, qty=10)\n'
                    '    if bar_index == 1:\n'
                    + '\n'.join('        ' + line for line in body.splitlines()) + '\n')
    bars = [OHLCV(timestamp=1704067200000+i*300000, open=100, high=101,
                  low=99, close=100, volume=1) for i in range(4)]
    runner = ScriptRunner(path, iter(bars), syminfo)
    return [abs(trade.size) for _, _, trades in runner.run_iter()
            for trade in trades if trade.size != 0]


@pytest.mark.parametrize('body,preamble,sizes', [
    ('strategy.close("L", qty=2)\nstrategy.close("L", qty=3)', '', [2, 3]),
    ('part(2)\npart(3)', 'def part(qty):\n    strategy.close("L", qty=qty)\n', [3]),
    ('for qty in range(2, 4):\n    strategy.close("L", qty=qty)', '', [3]),
    ('first()\nsecond()',
     'def first():\n    strategy.close("L", qty=2)\n'
     'def second():\n    strategy.close("L", qty=3)\n', [2, 3]),
    ('strategy.close("L", qty=3)\nstrategy.close_all()', '', [3, 7]),
    ('strategy.close("L", qty=3, immediately=True)\nstrategy.close_all(immediately=True)', '', [3, 7]),
    ('flatten()\nflatten()', 'def flatten():\n    strategy.close_all()\n', [10]),
    ('shut("L", qty=2)\nshut("L", qty=3)\nflatten()',
     'from pynecore.lib.strategy import close as shut, close_all as flatten\n', [2, 3, 5]),
])
def __test_backtest_closes_never_read_frames__(tmp_path, syminfo, monkeypatch,
                                             body, preamble, sizes):
    def fail(*args):
        raise AssertionError('A transformed close must not inspect a frame')

    monkeypatch.setattr(strategy, '_sys', SimpleNamespace(_getframe=fail))
    assert __test_helper_run(tmp_path, syminfo, body, preamble) == sizes


def __test_dynamic_close_alias_keeps_statement_semantics__(tmp_path, syminfo):
    assert __test_helper_run(tmp_path, syminfo,
                            'shut = strategy.close\nshut("L", qty=2)\nshut("L", qty=3)') == [2, 3]


def __test_dynamic_close_alias_in_loop_modifies_one_order__(tmp_path, syminfo):
    assert __test_helper_run(tmp_path, syminfo,
                            'shut = strategy.close\nfor qty in range(2, 4):\n    shut("L", qty=qty)') == [3]
