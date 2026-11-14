"""
Every ``ta`` builtin with a length stops the script when the length is not
positive.

Measured on TradingView (CAPITALCOM:EURUSD 1D): each call below with a zero
length, the series-length forms (a length that is 0 on the first bar) and a
negative length stop the script on bar 0. ``ta.lowest``, ``ta.highest`` and their
``bars`` forms used to answer the source value instead.
"""
import pytest

from pynecore.core.script_runner import ScriptRunner
from pynecore.types.ohlcv import OHLCV

#: One Pyne expression per builtin and length parameter, each with a zero length
__test_helper_CALLS = (
    'ta.alma(close, 0, 0.85, 6)',
    'ta.atr(0)',
    'ta.bb(close, 0, 2)[0]',
    'ta.bbw(close, 0, 2)',
    'ta.cci(close, 0)',
    'ta.change(close, 0)',
    'ta.cmo(close, 0)',
    'ta.cog(close, 0)',
    'ta.correlation(close, open, 0)',
    'ta.dev(close, 0)',
    'ta.dmi(0, 14)[0]',
    'ta.dmi(14, 0)[2]',
    'ta.ema(close, 0)',
    'ta.falling(close, 0)',
    'ta.highest(close, 0)',
    'ta.highest(0)',
    'ta.highestbars(close, 0)',
    'ta.highestbars(0)',
    'ta.hma(close, 0)',
    'ta.kc(close, 0, 1.5)[0]',
    'ta.kcw(close, 0, 1.5)',
    'ta.linreg(close, 0, 0)',
    'ta.lowest(close, 0)',
    'ta.lowest(0)',
    'ta.lowestbars(close, 0)',
    'ta.lowestbars(0)',
    'ta.macd(close, 0, 26, 9)[0]',
    'ta.macd(close, 12, 0, 9)[0]',
    'ta.macd(close, 12, 26, 0)[1]',
    'ta.median(close, 0)',
    'ta.mfi(hlc3, 0)',
    'ta.mode(close, 0)',
    'ta.mom(close, 0)',
    'ta.percentile_linear_interpolation(close, 0, 50)',
    'ta.percentile_nearest_rank(close, 0, 50)',
    'ta.percentrank(close, 0)',
    'ta.range(close, 0)',
    'ta.rci(close, 0)',
    'ta.rising(close, 0)',
    'ta.rma(close, 0)',
    'ta.roc(close, 0)',
    'ta.rsi(close, 0)',
    'ta.sma(close, 0)',
    'ta.stdev(close, 0)',
    'ta.stoch(close, high, low, 0)',
    'ta.supertrend(3, 0)[0]',
    'ta.tsi(close, 0, 25)',
    'ta.tsi(close, 13, 0)',
    'ta.variance(close, 0)',
    'ta.vwma(close, 0)',
    'ta.wma(close, 0)',
    'ta.wpr(0)',
    'ta.lowest(close, 0 if bar_index % 2 == 0 else 3)',
    'ta.highest(close, 0 if bar_index % 2 == 0 else 3)',
    'ta.sma(close, 0 if bar_index % 2 == 0 else 3)',
    'ta.lowest(close, -1)',
)


@pytest.mark.parametrize('call', __test_helper_CALLS)
def __test_a_non_positive_length_stops_the_script__(call, tmp_path, monkeypatch, syminfo):
    """The first bar already fails, like on TradingView"""
    monkeypatch.setenv('PYNE_SAVE_SCRIPT_TOML', '0')
    path = tmp_path / 'length_probe.py'
    path.write_text(f'''"""@pyne"""
from pynecore.lib import script, ta, close, open, high, low, hlc3, bar_index
@script.indicator("length probe")
def main():
    return {{"value": {call}}}
''')
    bars = [OHLCV(1735689600000 + i * 86400000, 10.0 + i, 12.0 + i, 9.0 + i, 11.0 + i, 100.0)
            for i in range(4)]
    with pytest.raises(AssertionError, match='must be'):
        for _ in ScriptRunner(path, bars, syminfo).run_iter():
            pass
