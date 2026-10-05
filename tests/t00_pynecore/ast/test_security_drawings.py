"""Validate handwritten Pyne before security lowering in both language profiles."""
import pytest

from pynecore.core.import_hook import PyneLoader


HEAD = '''
from pynecore.lib import script, request, label, close, bar_index, syminfo, na
from pynecore.core.pine_udt import udt
from pynecore.types import Label
'''


def _load(tmp_path, source, mode=''):
    path = tmp_path / 'security_drawing_probe.py'
    path.write_text(f'"""@pyne{mode}"""\n' + HEAD + source)
    return PyneLoader('security_drawing_probe', str(path)).get_code('security_drawing_probe')


@pytest.mark.parametrize('mode', ['', ' edge'])
@pytest.mark.parametrize('expression,setup', [
    ('label.new(bar_index, close)', ''),
    ('helper()', '    def helper():\n        return label.new(bar_index, close)\n'),
    ('outer()', '    def helper():\n        return label.new(bar_index, close)\n'
                 '    def outer():\n        return helper()\n'),
    ('obj', '    obj = label.new(bar_index, close)\n'),
    ('obj', '    obj = na(Label)\n    obj = label.new(bar_index, close)\n'),
    ('make(bar_index, close)', '    make = label.new\n'),
])
@pytest.mark.parametrize('request_name', ['security', 'security_lower_tf'])
def __test_reject_drawing_dependencies__(tmp_path, expression, setup, mode, request_name):
    with pytest.raises(SyntaxError, match='label.new.*cannot be used'):
        _load(tmp_path, '@script.indicator("Probe")\ndef main():\n' + setup
              + f'    x = request.{request_name}(syminfo.tickerid, "D", {expression})\n', mode)


@pytest.mark.parametrize('mode', ['', ' edge'])
def __test_empty_drawing_fields_and_chart_creation__(tmp_path, mode):
    _load(tmp_path, '''
@udt
class Data:
    value: float = na(float)
    tag: Label = na(Label)

@script.indicator("Probe")
def main():
    def get():
        return Data(close)
    x = request.security(expression=get(), timeframe="D", symbol=syminfo.tickerid)
    x.tag = label.new(bar_index, x.value)
''', mode)


def __test_unused_helper_and_shadowed_parameter__(tmp_path):
    _load(tmp_path, '''
@script.indicator("Probe")
def main():
    obj = label.new(bar_index, close)
    def unused():
        return label.new(bar_index, close)
    def helper(obj):
        return obj + 1
    x = request.security(syminfo.tickerid, "D", helper(close))
''')


@pytest.mark.parametrize('expression', [
    'label.new(0, 1)', 'line.new(0, 1, 1, 2)', 'box.new(0, 2, 1, 1)',
    'table.new(position.top_right, 1, 1)', 'polyline.new([])',
    'linefill.new(na(Line), na(Line), color.red)',
    'label.copy(na(Label))', 'line.copy(na(Line))', 'box.copy(na(Box))',
])
def __test_all_drawing_constructors_and_copies__(tmp_path, expression):
    with pytest.raises(SyntaxError, match='cannot be used in the expression'):
        _load(tmp_path, '''
from pynecore.lib import line, box, table, position, polyline, linefill, color
from pynecore.types import Line, Box
@script.indicator("Probe")
def main():
    x = request.security(syminfo.tickerid, "D", ''' + expression + ')\n')


@pytest.mark.parametrize('body', [
    '''    a: Label = na(Label)
    x = request.security(syminfo.tickerid, "D", method_call('copy', a))
''',
    '''    def get(value):
        return request.security(syminfo.tickerid, "D", value)
    x = get(label.get_y(label.new(0, 1)))
''',
])
def __test_methods_and_request_parameters__(tmp_path, body):
    with pytest.raises(SyntaxError, match='cannot be used in the expression'):
        _load(tmp_path, 'from pynecore.core.pine_method import method_call\n'
              '@script.indicator("Probe")\ndef main():\n' + body)


def __test_empty_drawing_field_survives_security_worker__(tmp_path):
    import sys
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    from pynecore.types.ohlcv import OHLCV

    _load(tmp_path, '''
from pynecore.lib import array, plot
@udt
class Data:
    value: float = na(float)
    tag: Label = na(Label)

@script.indicator("Probe")
def main():
    def get():
        return Data(close)
    x = request.security(syminfo.tickerid, "15", get())
    if not na(x):
        x.tag = label.new(bar_index, x.value)
    plot(array.size(label.all), "labels")
    plot(x.value if not na(x) else na, "value")
''')
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    info = SymInfo(prefix='TEST', description='Test', ticker='TEST', currency='USD',
                   period='15', type='crypto', mintick=0.01, pricescale=100,
                   minmove=1, pointvalue=1, timezone='UTC', volumetype='base',
                   mincontract=0.0001, opening_hours=opening_hours,
                   session_starts=session_starts, session_ends=session_ends)
    feed = tmp_path / 'feed.ohlcv'
    start = 1_704_067_200_000
    with OHLCVWriter(feed, '15') as writer:
        for i in range(4):
            writer.write(OHLCV(timestamp=start + i * 900_000, open=100, high=110,
                               low=90, close=100 + i, volume=1))
    info.save_toml(feed.with_suffix('.toml'))
    info.period = '5'
    bars = (OHLCV(timestamp=start + i * 300_000, open=100, high=110,
                  low=90, close=100, volume=1) for i in range(12))
    runner = ScriptRunner(tmp_path / 'security_drawing_probe.py', bars, info,
                          security_data={'15': feed})
    try:
        rows = [dict(row[1]) for row in runner.run_iter()]
        assert rows[-1]['value'] == 103
        assert rows[-1]['labels'] > 0
        assert len(runner.drawings()['labels']) == rows[-1]['labels']
    finally:
        sys.modules.pop('security_drawing_probe', None)
