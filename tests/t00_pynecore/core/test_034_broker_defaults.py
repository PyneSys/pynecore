"""Tests for cross-broker runtime defaults loaded from ``brokers.toml``."""

from pathlib import Path

import pytest

from pynecore.core.broker.defaults import (
    BrokerDefaults,
    VALID_UNEXPECTED_CANCEL_POLICIES,
    load_broker_defaults,
)


@pytest.fixture(autouse=True)
def _clear_ensured_cache():
    """Drop the ``_ensured`` cache between tests so each call hits disk."""
    yield
    if hasattr(BrokerDefaults, '_ensured'):
        del BrokerDefaults._ensured


def __test_missing_file_returns_safe_defaults__(tmp_path: Path) -> None:
    """Missing ``brokers.toml`` auto-creates the file and returns safe defaults.

    First-run smoke: no ``brokers.toml`` exists → file auto-created,
    safe defaults returned (graceful stop). This is the most common
    production path — users get a working broker without writing any
    TOML by hand."""
    defaults = load_broker_defaults(tmp_path)
    assert defaults.on_unexpected_cancel == 'stop'
    assert (tmp_path / 'brokers.toml').exists()


def __test_user_override_is_loaded__(tmp_path: Path) -> None:
    """A user-edited ``on_unexpected_cancel = "ignore"`` line is loaded unchanged.

    A user-edited ``on_unexpected_cancel = "ignore"`` line must round-
    trip through the self-healing TOML loader unchanged."""
    (tmp_path / 'brokers.toml').write_text(
        'on_unexpected_cancel = "ignore"\n',
        encoding='utf-8',
    )
    defaults = load_broker_defaults(tmp_path)
    assert defaults.on_unexpected_cancel == 'ignore'


def __test_invalid_policy_raises_value_error__(tmp_path: Path) -> None:
    """An invalid policy value fails closed at load time with a ``ValueError``.

    Typos in the policy value must fail closed at startup rather than
    surface at the first reconcile cycle — the misconfiguration would
    otherwise silently fall through to the ``stop`` branch via the
    ``policy not in {…}`` checks in :mod:`reconcile`, which is harder
    to diagnose than an explicit load-time error."""
    (tmp_path / 'brokers.toml').write_text(
        'on_unexpected_cancel = "abort"\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError) as excinfo:
        load_broker_defaults(tmp_path)
    msg = str(excinfo.value)
    assert 'on_unexpected_cancel' in msg
    assert "'abort'" in msg


def __test_valid_policy_set_contents__() -> None:
    """``VALID_UNEXPECTED_CANCEL_POLICIES`` holds exactly the five expected policies.

    Guard against accidental drift between the validator's allow-set
    and the policy branches implemented in
    :meth:`DisappearanceTracker._apply_policy`."""
    assert VALID_UNEXPECTED_CANCEL_POLICIES == frozenset({
        'stop', 'stop_and_cancel', 're_place', 'ignore', 'halt',
    })
