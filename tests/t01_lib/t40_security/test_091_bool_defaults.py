"""Bool security results keep their type on missing bars and through history."""
import sys

import pytest

from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.script_runner import ScriptRunner
from pynecore.types.na import NA, na_bool, set_bool_na
from pynecore.types.ohlcv import OHLCV


_SOURCE = '''"""
@pyne
"""
from pynecore.core.series import inline_series
from pynecore.lib import barmerge, close, nz, open, request, script, syminfo, ta
from pynecore.types import Series


def pair():
    return close > open, close


@script.indicator("Bool security defaults", na_bool=BOOL_MODE)
def main():
    flag: Series[bool] = close > open
    value = request.security(syminfo.tickerid, "60", flag, gaps=barmerge.gaps_on)
    child_na = request.security(syminfo.tickerid, "60", ta.change(flag))
    both = request.security(syminfo.tickerid, "60", pair(), gaps=barmerge.gaps_on)
    tuple_bool, tuple_price = both
    previous = inline_series(value, 1)
    return {
        "value": value, "eq": value == False, "ne": value != True,
        "nz": nz(value), "child_na": child_na, "previous": previous,
        "tuple_bool": tuple_bool, "tuple_price": tuple_price,
    }
'''


@pytest.mark.parametrize('three_state', [False, True])
@pytest.mark.parametrize('batch', [False, True])
def __test_missing_bool_security_results_keep_the_script_mode__(tmp_path, syminfo,
                                                               monkeypatch, three_state, batch):
    """Gaps, initial missing values, tuple fields and history retain bool semantics."""
    from pynecore.core import security

    monkeypatch.setattr(security, 'NO_BATCH', not batch)
    monkeypatch.setenv('PYNE_NO_SECURITY_BATCH', '0' if batch else '1')
    monkeypatch.setenv('PYNE_SAVE_SCRIPT_TOML', '0')
    source = tmp_path / 'bool_security_defaults.py'
    source.write_text(_SOURCE.replace('BOOL_MODE', str(three_state)), encoding='utf-8')
    t0 = 1_735_689_600_000
    feed = tmp_path / 'HTF60.ohlcv'
    with OHLCVWriter(feed, '60') as writer:
        for hour in range(3):
            writer.write(OHLCV(timestamp=t0 + hour * 3_600_000,
                               open=1.0, high=2.0, low=1.0, close=2.0, volume=1.0))
    syminfo.period = '60'
    syminfo.save_toml(feed.with_suffix('.toml'))
    syminfo.period = '5'
    bars = [OHLCV(timestamp=t0 + i * 300_000,
                  open=1.0, high=2.0, low=1.0, close=2.0, volume=1.0) for i in range(25)]
    missing = na_bool if three_state else False
    try:
        runner = ScriptRunner(source, iter(bars), syminfo, security_data={'60': str(feed)})
        rows = [dict(row) for _, row in runner.run_iter()]
        assert len(rows) == 25
        for i, row in enumerate(rows):
            if i in (11, 23):
                assert row['value'] is True
                assert row['tuple_bool'] is True
                assert row['tuple_price'] == 2.0
                assert row['eq'] is False and row['ne'] is False
                assert row['nz'] is True
            else:
                assert row['value'] is missing, (i, row)
                assert row['tuple_bool'] is missing, (i, row)
                assert isinstance(row['tuple_price'], NA)
                assert row['eq'] is (na_bool if three_state else True), (i, row)
                assert row['ne'] is (na_bool if three_state else True), (i, row)
                assert row['nz'] is False, (i, row)
            if i > 0:
                assert row['previous'] is rows[i - 1]['value'], (i, row)
        # The child's first ta.change(bool) publishes a real typed bool na.
        assert rows[11]['child_na'] is missing
        assert rows[23]['child_na'] is False
    finally:
        sys.modules.pop(source.stem, None)
        set_bool_na(False)
