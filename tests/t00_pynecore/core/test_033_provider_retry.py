"""
Tests for ``pyne run`` provider-mode startup resilience.

``_download_provider_data_resilient`` rides out *transient* broker outages
(maintenance, lost route) for long-running ``--broker`` / ``--live`` runs by
waiting and retrying, while still failing fast on *permanent* errors and for
one-shot backtests.
"""
from types import SimpleNamespace

import pytest
from typer import Exit

from pynecore.cli.commands import run as run_command
from pynecore.cli.commands.run import (
    _download_provider_data_resilient,
    _PROVIDER_RETRY_MAX_DELAY,
)
from pynecore.core.plugin import ProviderError, TransientProviderError


def __test_transient_error_retried_until_success__(monkeypatch):
    """A transient failure is retried (with waits) until the download succeeds."""
    waits: list[float] = []
    monkeypatch.setattr(run_command, "_wait_before_retry", lambda d: waits.append(d))

    sentinel = SimpleNamespace(ohlcv_path="x")
    calls = {"n": 0}

    def fake_download(*_args):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProviderError("broker maintenance")
        return sentinel

    monkeypatch.setattr(run_command, "_download_provider_data", fake_download)

    result = _download_provider_data_resilient(
        "ctrader:pepperstoneuk:BTCUSD@1", "-500", retry_transient=True,
    )

    assert result is sentinel
    assert calls["n"] == 3          # two failures + one success
    assert len(waits) == 2          # one wait per failure
    # Capped exponential backoff: 2s then 4s.
    assert waits == [2.0, 4.0]


def __test_backoff_saturates_at_cap__(monkeypatch):
    """The retry delay grows exponentially but never exceeds the cap."""
    waits: list[float] = []
    monkeypatch.setattr(run_command, "_wait_before_retry", lambda d: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args):
        calls["n"] += 1
        if calls["n"] <= 8:
            raise TransientProviderError("still down")
        return SimpleNamespace()

    monkeypatch.setattr(run_command, "_download_provider_data", fake_download)

    _download_provider_data_resilient("p:s@1", None, retry_transient=True)

    # 2, 4, 8, 16, 32, 60 (capped), 60, 60 — never above the cap.
    assert max(waits) == _PROVIDER_RETRY_MAX_DELAY
    assert all(w <= _PROVIDER_RETRY_MAX_DELAY for w in waits)
    assert waits[:6] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0]


def __test_permanent_error_fails_fast_in_live_mode__(monkeypatch):
    """A permanent failure exits immediately even with retry enabled — no wait."""
    waits: list[float] = []
    monkeypatch.setattr(run_command, "_wait_before_retry", lambda d: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args):
        calls["n"] += 1
        raise ProviderError("symbol not found")

    monkeypatch.setattr(run_command, "_download_provider_data", fake_download)

    with pytest.raises(Exit):
        _download_provider_data_resilient("p:s@1", None, retry_transient=True)

    assert calls["n"] == 1          # tried once
    assert waits == []              # never waited


def __test_backtest_does_not_retry_transient__(monkeypatch):
    """With retry disabled (one-shot backtest), even a transient error fails fast."""
    waits: list[float] = []
    monkeypatch.setattr(run_command, "_wait_before_retry", lambda d: waits.append(d))

    calls = {"n": 0}

    def fake_download(*_args):
        calls["n"] += 1
        raise TransientProviderError("broker maintenance")

    monkeypatch.setattr(run_command, "_download_provider_data", fake_download)

    with pytest.raises(Exit):
        _download_provider_data_resilient("p:s@1", None, retry_transient=False)

    assert calls["n"] == 1
    assert waits == []
