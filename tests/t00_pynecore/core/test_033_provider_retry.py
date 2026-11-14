"""
Tests for ``pyne run`` provider-mode startup resilience.

``download_provider_history_resilient`` rides out *transient* broker outages
(maintenance, lost route) for long-running ``--broker`` / ``--live`` runs by
waiting and retrying, while still failing fast on *permanent* errors and for
one-shot backtests. ``pyne run`` turns such a failure into a one-line error.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer import Exit

from pynecore.cli.commands import run as run_command
from pynecore.core import provider_download
from pynecore.core.provider_download import (
    download_provider_history_resilient,
    _PROVIDER_RETRY_MAX_DELAY,
)
from pynecore.core.plugin import ProviderError, TransientProviderError

_DIR = Path("unused")


def __test_transient_error_retried_until_success__(monkeypatch):
    """A transient failure is retried (with waits) until the download succeeds."""
    waits: list[float] = []
    monkeypatch.setattr(provider_download, "_wait_before_retry",
                        lambda d, _stop=None: waits.append(d))

    sentinel = SimpleNamespace(ohlcv_path="x")
    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProviderError("broker maintenance")
        return sentinel

    monkeypatch.setattr(provider_download, "download_provider_history", fake_download)

    result = download_provider_history_resilient(
        "ctrader:pepperstoneuk:BTCUSD@1", "-500", data_dir=_DIR, config_dir=_DIR,
        retry_transient=True,
    )

    assert result is sentinel
    assert calls["n"] == 3          # two failures + one success
    assert len(waits) == 2          # one wait per failure
    # Capped exponential backoff: 2s then 4s.
    assert waits == [2.0, 4.0]


def __test_backoff_saturates_at_cap__(monkeypatch):
    """The retry delay grows exponentially but never exceeds the cap."""
    waits: list[float] = []
    monkeypatch.setattr(provider_download, "_wait_before_retry",
                        lambda d, _stop=None: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 8:
            raise TransientProviderError("still down")
        return SimpleNamespace()

    monkeypatch.setattr(provider_download, "download_provider_history", fake_download)

    download_provider_history_resilient("p:s@1", None, data_dir=_DIR, config_dir=_DIR,
                                        retry_transient=True)

    # 2, 4, 8, 16, 32, 60 (capped), 60, 60 — never above the cap.
    assert max(waits) == _PROVIDER_RETRY_MAX_DELAY
    assert all(w <= _PROVIDER_RETRY_MAX_DELAY for w in waits)
    assert waits[:6] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0]


def __test_permanent_error_fails_fast_in_live_mode__(monkeypatch):
    """A permanent failure is raised immediately even with retry enabled — no wait."""
    waits: list[float] = []
    monkeypatch.setattr(provider_download, "_wait_before_retry",
                        lambda d, _stop=None: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        raise ProviderError("symbol not found")

    monkeypatch.setattr(provider_download, "download_provider_history", fake_download)

    with pytest.raises(ProviderError):
        download_provider_history_resilient("p:s@1", None, data_dir=_DIR, config_dir=_DIR,
                                            retry_transient=True)

    assert calls["n"] == 1          # tried once
    assert waits == []              # never waited


def __test_backtest_does_not_retry_transient__(monkeypatch):
    """With retry disabled (one-shot backtest), even a transient error fails fast."""
    waits: list[float] = []
    monkeypatch.setattr(provider_download, "_wait_before_retry",
                        lambda d, _stop=None: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        raise TransientProviderError("broker maintenance")

    monkeypatch.setattr(provider_download, "download_provider_history", fake_download)

    with pytest.raises(TransientProviderError):
        download_provider_history_resilient("p:s@1", None, data_dir=_DIR, config_dir=_DIR,
                                            retry_transient=False)

    assert calls["n"] == 1
    assert waits == []


def __test_cli_turns_a_provider_failure_into_a_clean_exit__(monkeypatch, capsys):
    """``pyne run`` prints the failure on one line and exits, without a traceback."""
    def fake_resilient(*_args, **_kwargs):
        raise ProviderError("symbol not found")

    monkeypatch.setattr(run_command, "download_provider_history_resilient", fake_resilient)

    with pytest.raises(Exit):
        run_command._download_provider_data_resilient("p:s@1", None, retry_transient=True)

    assert "Error: symbol not found" in capsys.readouterr().err


def __test_stop_ends_the_retry_wait__(monkeypatch):
    """A stop set from another thread ends the backoff at once, no retry follows."""
    import threading
    import time

    from pynecore.core.live_runner import LiveStop
    from pynecore.core.provider_download import DownloadStopped

    calls = {"n": 0}

    def fake_download(*_args, **_kwargs):
        calls["n"] += 1
        raise TransientProviderError("broker maintenance")

    monkeypatch.setattr(provider_download, "download_provider_history", fake_download)
    monkeypatch.setattr(provider_download, "_PROVIDER_RETRY_BASE_DELAY", 30.0)

    stop = LiveStop()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    started = time.monotonic()
    with pytest.raises(DownloadStopped):
        download_provider_history_resilient(
            "ctrader:pepperstoneuk:BTCUSD@1", "-500", data_dir=_DIR, config_dir=_DIR,
            retry_transient=True, stop=stop,
        )
    timer.join()

    assert time.monotonic() - started < 5.0
    assert calls["n"] == 1
