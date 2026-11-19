"""
@pyne

A script runs on a history and a live stream served by a provider hub as it runs on
the provider itself: the core's live flow is the same on a remote provider.
"""
import asyncio
import contextlib
import itertools
import json
import sys
import threading
import time
from pathlib import Path

from pynecore.lib import close, plot, script


@script.indicator("Provider Hub Script", overlay=True)
def main():
    plot(close, 'close')


@contextlib.contextmanager
def __test_helper_hub(monkeypatch, config_dir=None, load=None):
    """A hub on a thread of its own, and this process set up as its client."""
    from pynecore.core.plugin.hub import HUB_ENV, TOKEN_ENV, remote
    from pynecore.core.plugin.hub.server import ProviderHub

    loop = asyncio.new_event_loop()
    hub = ProviderHub("secret", config_dir=config_dir) if load is None \
        else ProviderHub("secret", config_dir=config_dir, load=load)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    port = asyncio.run_coroutine_threadsafe(hub.start(), loop).result(10)
    monkeypatch.setenv(HUB_ENV, f"127.0.0.1:{port}")
    monkeypatch.setenv(TOKEN_ENV, "secret")
    remote._remote_classes.clear()
    remote._class_channel = None
    try:
        yield hub, loop
    finally:
        monkeypatch.delenv(HUB_ENV)
        remote._remote_classes.clear()
        remote._class_channel = None
        asyncio.run_coroutine_threadsafe(hub.close(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


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


def __test_download_then_live_stream_through_the_hub__(script_path, module_key, tmp_path,
                                                       monkeypatch):
    """A script runs on a hub-served history and live stream as on the provider itself"""
    from pynecore.core.plugin import load_plugin
    from pynecore.core.provider_download import download_provider_history, open_live_stream
    from pynecore.core.plugin.hub.remote import RemoteLiveProvider
    from pynecore.core.script_runner import ScriptRunner

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    last_ts = (int(time.time()) // 60 - 2) * 60_000
    data_dir, config_dir = __test_helper_write_fixture(tmp_path, last_ts)

    with __test_helper_hub(monkeypatch, config_dir):
        assert issubclass(load_plugin("replay"), RemoteLiveProvider)
        history = download_provider_history("replay:TEST@1", "-5", data_dir=data_dir,
                                            config_dir=config_dir, in_memory=True)
        assert isinstance(history.provider_instance, RemoteLiveProvider)
        assert history.time_from_ts == last_ts - 4 * 60_000
        assert history.bars is not None
        bars = [bar for bar in history.bars if bar.timestamp >= history.time_from_ts]
        assert len(bars) == 5
        live = open_live_stream(history, last_historical_timestamp=last_ts, shutdown_timeout=5.0)
        runner = ScriptRunner(script_path, itertools.chain(bars, live), history.syminfo,
                              live=True)
        results = []
        for candle, plot_data in runner.run_iter(yield_ticks=True):
            # The live bar's forming update is dropped when it lands during the
            # history and runs as a tick when it lands after: either is right
            if candle.is_closed:
                results.append(plot_data['close'])
            if len(results) == 6:
                break
        live.close()

    assert results == [100.0 + i for i in range(5, 10)] + [111.0]
