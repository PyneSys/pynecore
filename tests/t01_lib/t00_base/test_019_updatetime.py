"""
@pyne

Update timestamps are bar opens on history and clock snapshots on realtime bars.
"""
from pynecore.lib import plot, script, updatetime


@script.indicator("Update timestamp")
def main():
    plot(updatetime, "updated")
    plot(updatetime[1], "previous")


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=1_735_689_600_000 + i * 300_000,
                  open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0) for i in range(3)]


def __test_historical_updates_and_history__(script_path, module_key, syminfo, monkeypatch):
    """History uses each bar's open even when timenow is pinned to another instant."""
    import sys
    from pynecore import lib
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.na import NA

    monkeypatch.setattr(lib, "_timenow_ms", 1_800_000_000_000)
    bars = __test_helper_bars()
    sys.modules.pop(module_key, None)
    sys.modules.pop(script_path.stem, None)
    runner = ScriptRunner(script_path, bars, syminfo)
    rows = [dict(data) for _, data in runner.run_iter()]
    assert [row["updated"] for row in rows] == [bar.timestamp for bar in bars]
    assert isinstance(rows[0]["previous"], NA)
    assert [row["previous"] for row in rows[1:]] == [bar.timestamp for bar in bars[:-1]]


def __test_realtime_updates_share_one_fractional_clock_snapshot__(monkeypatch):
    """Repeated reads stay stable; a new tick of the same bar gets a fresh clock."""
    from datetime import datetime, UTC
    from pynecore import lib

    instants = [datetime(2026, 1, 1, 0, 0, 1, 125_000, UTC),
                datetime(2026, 1, 1, 0, 0, 2, 125_500, UTC)]
    expected = [instant.timestamp() * 1000.0 for instant in instants]

    class Clock:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return instants.pop(0)

    monkeypatch.setattr(lib, "datetime", Clock)
    monkeypatch.setattr(lib.barstate, "isrealtime", True)
    monkeypatch.setattr(lib, "_timenow_ms", 0)
    monkeypatch.setattr(lib, "_updatetime_dt", None)
    monkeypatch.setattr(lib, "_updatetime_ms", 0.0)
    monkeypatch.setattr(lib, "_datetime", datetime(2026, 1, 1, tzinfo=UTC))
    assert lib.updatetime() == expected[0]
    assert lib.updatetime() == expected[0]
    monkeypatch.setattr(lib, "_datetime", datetime(2026, 1, 1, tzinfo=UTC))
    assert lib.updatetime() == expected[1]
    assert not instants


def __test_realtime_update_respects_clock_pin__(monkeypatch):
    """A replay pin takes precedence over the realtime wall clock."""
    from pynecore import lib

    monkeypatch.setattr(lib.barstate, "isrealtime", True)
    monkeypatch.setattr(lib, "_timenow_ms", 1_800_000_000_123)
    assert lib.updatetime() == 1_800_000_000_123.0
