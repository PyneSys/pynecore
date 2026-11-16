"""
@pyne

The provider-mode path outside the CLI: download a provider string's history with
``download_provider_history``, start its live stream with ``open_live_stream`` and
run a script on history + live, the way an IDE bridge does. Driven offline by the
fixture-fed ``replay`` provider.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import pytest

from pynecore.lib import close, plot, script


@script.indicator("Provider Download Live Stream", overlay=True)
def main():
    plot(close, 'close')


def __test_helper_write_fixture(tmp_path: Path, last_ts: int) -> tuple[Path, Path]:
    """A replay fixture of 10 closed warmup bars and one live bar, plus its config."""
    warmup = [[last_ts - (9 - i) * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 1.0]
              for i in range(10)]
    live_ts = last_ts + 60_000
    live = [
        [live_ts, 110.0, 110.5, 110.0, 110.5, 1.0, False],
        [live_ts, 110.0, 111.0, 110.0, 111.0, 1.0, True],
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"warmup": warmup, "live": live}), encoding="utf-8")
    config_dir = tmp_path / "config"
    (config_dir / "plugins").mkdir(parents=True)
    (config_dir / "plugins" / "replay.toml").write_text(
        f'fixture_path = "{fixture.as_posix()}"\n', encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir, config_dir


def __test_download_then_live_stream_runs_the_script__(script_path, module_key, tmp_path):
    """The downloaded bars run, then the bars the stream brought in run on

    The replay provider streams its whole recording as soon as it connects, so the
    live rows land while the history runs: the closed bar is caught up after the
    history, and the update of the bar it closed is dropped, as for any feed whose
    bars arrive during the warmup.
    """
    from pynecore.core.ohlcv import OHLCVReader
    from pynecore.core.provider_download import download_provider_history, open_live_stream
    from pynecore.core.script_runner import ScriptRunner

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)

    history = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                        config_dir=config_dir)

    assert history.time_from_ts == last_ts - 4 * 60_000
    with OHLCVReader(history.ohlcv_path) as reader:
        bars = list(reader.read_from(history.time_from_ts, last_ts))
        assert len(bars) == 5
        live = open_live_stream(history, last_historical_timestamp=last_ts, shutdown_timeout=5.0)
        runner = ScriptRunner(script_path, itertools.chain(bars, live), history.syminfo,
                              live=True)
        results = []
        for candle, plot_data in runner.run_iter(yield_ticks=True):
            results.append((candle.is_closed, plot_data['close']))
            if len(results) == 6:
                break

    assert results == [(True, 100.0 + i) for i in range(5, 10)] + [(True, 111.0)]


def __test_live_stream_needs_a_live_provider__(tmp_path):
    """A provider without live data is refused before anything connects"""
    from types import SimpleNamespace
    from pynecore.core.provider_download import (ProviderHistory, ProviderSetupError,
                                                  open_live_stream)
    from pynecore.core.provider_string import parse_provider_string

    history = ProviderHistory(tmp_path / "x.ohlcv", SimpleNamespace(),  # type: ignore[arg-type]
                              parse_provider_string("csvonly:TEST@1", require_timeframe=True),
                              SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(ProviderSetupError, match="does not support live data"):
        open_live_stream(history, last_historical_timestamp=0)


def __test_in_memory_download_writes_nothing_and_matches_the_file__(tmp_path):
    """A history kept in memory holds the bars a file download writes, and leaves the
    data directory as it was: a dataset of the same symbol and timeframe stays intact"""
    from pynecore.core.ohlcv import OHLCVReader
    from pynecore.core.provider_download import download_provider_history

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)

    saved = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                      config_dir=config_dir)
    assert saved.ohlcv_path is not None and saved.bars is None
    files = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    with OHLCVReader(saved.ohlcv_path) as reader:
        saved_bars = list(reader.read_from(0))

    history = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                        config_dir=config_dir, in_memory=True)

    assert history.ohlcv_path is None
    assert history.bars is not None
    assert history.time_from_ts == saved.time_from_ts == last_ts - 4 * 60_000
    assert [bar.timestamp for bar in history.bars] == [bar.timestamp for bar in saved_bars]
    assert [bar.close for bar in history.bars] == [bar.close for bar in saved_bars]
    assert history.syminfo.ticker == saved.syminfo.ticker
    assert {path.name: path.read_bytes() for path in data_dir.iterdir()} == files


def __test_in_memory_download_into_a_missing_data_dir__(tmp_path):
    """Nothing is created for a history kept in memory, not even the data directory"""
    from pynecore.core.provider_download import download_provider_history

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)
    data_dir.rmdir()

    history = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                        config_dir=config_dir, in_memory=True)

    assert history.bars is not None and len(history.bars) >= 5
    assert not data_dir.exists()


def __test_in_memory_download_refuses_a_bar_the_file_refuses__(tmp_path):
    """A malformed bar stops a history kept in memory just as it stops a file download"""
    from pynecore.core.provider_download import download_provider_history
    from pynecore.errorcodes import InvalidDataError

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)
    fixture = tmp_path / "fixture.json"
    recording = json.loads(fixture.read_text(encoding="utf-8"))
    # The high is below the open and the close
    recording["warmup"][-2] = [last_ts - 60_000, 100.0, 90.0, 80.0, 95.0, 1.0]
    fixture.write_text(json.dumps(recording), encoding="utf-8")

    for in_memory in (False, True):
        with pytest.raises(InvalidDataError, match="high of the bar"):
            download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                      config_dir=config_dir, in_memory=in_memory)
