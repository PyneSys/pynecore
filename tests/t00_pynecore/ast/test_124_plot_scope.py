"""Plot scope validation runs before module execution and per-bar evaluation."""
import ast
from pathlib import Path

import pytest

from pynecore.core.import_hook import _analyse_tree


def __test_helper_analyse(body: str, preamble: str = '') -> ast.Module:
    source = ('from pynecore.lib import script, plot, hline, fill, close, bar_index, na\n'
              + preamble + '\n@script.indicator("scope")\ndef main():\n'
              + '\n'.join('    ' + line for line in body.splitlines()) + '\n')
    return _analyse_tree(ast.parse(source), source, Path('/tmp/plot-scope-test.py'), None)


@pytest.mark.parametrize('call', [
    'plot(close)', 'lib.plot(close)', 'lib.plot.plot(close)', 'lib.plotshape(True)',
    'lib.plotchar(close)', 'lib.plotarrow(close)', 'lib.plotcandle(close, close, close, close)',
    'lib.plotbar(close, close, close, close)', 'hline(1)', 'lib.hline.hline(1)',
    'fill(None, None)', 'lib.bgcolor(None)', 'lib.barcolor(None)', 'lib.alertcondition(True)',
])
def __test_plot_family_rejects_conditional_declarations__(call):
    with pytest.raises(SyntaxError, match='unconditionally') as info:
        __test_helper_analyse('if bar_index >= 3:\n    '+call)
    assert info.value.filename == str(Path('/tmp/plot-scope-test.py').resolve())
    assert info.value.text.strip() == call


@pytest.mark.parametrize('body', [
    'if False:\n    plot(close)',
    'for i in range(1):\n    plot(close)',
    'while False:\n    plot(close)',
    'try:\n    plot(close)\nexcept ValueError:\n    pass',
    'with open("file"):\n    plot(close)',
    'match bar_index:\n    case 3:\n        plot(close)',
    'plot(close) if bar_index >= 3 else None',
    'bar_index >= 3 and plot(close)',
    '[plot(close) for _ in range(1)]',
    '(plot(close) for _ in range(1))',
    'draw = lambda: plot(close)\ndraw()',
    'def draw():\n    plot(close)\ndraw()',
    'def main():\n    plot(close)\nmain()',
    'class Drawing:\n    def main(self):\n        plot(close)',
    'if bar_index < 3:\n    return\nplot(close)',
    'for i in range(1):\n    if bar_index < 3:\n        return\nplot(close)',
])
def __test_local_and_skipped_plot_calls_are_rejected__(body):
    with pytest.raises(SyntaxError, match='unconditionally'):
        __test_helper_analyse(body)


@pytest.mark.parametrize('preamble,body', [
    ('from pynecore.lib import plot as draw', 'if bar_index >= 3:\n    draw(close)'),
    ('import pynecore.lib as builtins', 'if bar_index >= 3:\n    builtins.plot(close)'),
    ('from pynecore.lib.plot import plot as draw', 'if bar_index >= 3:\n    draw(close)'),
    ('draw = plot', 'if bar_index >= 3:\n    draw(close)'),
    ('', 'draw = plot\nrender = draw\nif bar_index >= 3:\n    render(close)'),
])
def __test_aliases_cannot_bypass_plot_scope__(preamble, body):
    with pytest.raises(SyntaxError, match='unconditionally'):
        __test_helper_analyse(body, preamble)


@pytest.mark.parametrize('body', [
    'helper(plot)', 'draw = [plot]\ndraw[0](close)',
    'draw = plot if bar_index >= 3 else helper\ndraw(close)',
    'def draw(fn=plot):\n    fn(close)\ndraw()',
    'plot.__call__(close)',
    'draw = plot\ndraw = helper\ndraw(close)',
])
def __test_plot_callbacks_and_dynamic_aliases_are_rejected__(body):
    with pytest.raises(SyntaxError, match='function value'):
        __test_helper_analyse(body)


@pytest.mark.parametrize('body,preamble', [
    ('plot(close if bar_index >= 3 else na)', ''),
    ('p = plot(close)\nh = hline(1)\nfill(p, p)', ''),
    ('plot(close, style=plot.style_line)', ''),
    ('draw(close)', 'from pynecore.lib import plot as draw'),
    ('draw(close)', 'draw = plot'),
    ('draw = plot\nrender = draw\nrender(close)', ''),
    ('draw(close)', 'from pynecore.lib.plot import plot as draw'),
    ('lib.plot.plot(close)', ''),
    ('def helper():\n    return close\nplot(helper())', ''),
    ('def helper(plot):\n    if bar_index >= 3:\n        plot(close)\nhelper(lambda x: x)', ''),
    ('if bar_index >= 3:\n    line.new(0, 1, 2, 3)\nplot(close)',
     'from pynecore.lib import line'),
])
def __test_unconditional_declarations_and_non_plot_calls_are_accepted__(body, preamble):
    __test_helper_analyse(body, preamble)


def __test_module_level_calls_and_function_defaults_are_rejected__():
    for preamble in ('plot(close)', 'def helper(value=plot(close)):\n    pass'):
        with pytest.raises(SyntaxError, match='unconditionally'):
            __test_helper_analyse('plot(close)', preamble)


def __test_test_harness_functions_are_exempt__():
    __test_helper_analyse('plot(close)',
                         'def __test_helper_probe():\n    if True:\n        plot(1)')


def __test_same_named_non_builtin_functions_are_allowed__():
    source = ('from pynecore.lib import script, close, bar_index\n'
              'def plot(x):\n    return x\n'
              '@script.indicator("local")\ndef main():\n'
              '    if bar_index >= 3:\n        plot(close)\n')
    _analyse_tree(ast.parse(source), source, Path('/tmp/plot-scope-shadow-test.py'), None)


@pytest.mark.parametrize('body', [
    'if bar_index >= 3:\n    plot(close)',
    'plot(close, "always")\nif bar_index >= 3:\n    plot(close, "late")',
    'if bar_index < 3:\n    plot(close, "early")\nelse:\n    plot(close, "late")',
])
def __test_invalid_plot_is_rejected_before_runner_creates_output__(tmp_path, syminfo, body):
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    script = tmp_path / 'invalid_plot_scope.py'
    script.write_text('"""\n@pyne\n"""\n'
                      'from pynecore.lib import script, plot, close, bar_index\n'
                      '@script.indicator("late")\ndef main():\n'
                      + '\n'.join('    ' + line for line in body.splitlines()) + '\n')
    output = tmp_path / 'plot.csv'
    bars = [OHLCV(timestamp=1704067200000+i*300000, open=100, high=101,
                  low=99, close=100, volume=1) for i in range(6)]
    with pytest.raises(SyntaxError, match='unconditionally'):
        ScriptRunner(script, iter(bars), syminfo, plot_path=output)
    assert not output.exists()
