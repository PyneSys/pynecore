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
