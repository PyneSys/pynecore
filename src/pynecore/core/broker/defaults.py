"""
Cross-broker runtime defaults loaded from ``workdir/config/brokers.toml``.

Holds policies that are broker-agnostic by design — the four
``on_unexpected_cancel`` modes, for instance, share identical semantics
regardless of which exchange the plugin talks to. Living here rather than
in each plugin's own config keeps the user-facing knob in a single place
and prevents every new broker plugin from copy-pasting the same field.

The CLI (``pyne run --broker``) loads :class:`BrokerDefaults` once and
injects the resolved values onto the plugin instance just before the
script runner starts. Plugin code reads them through the
:class:`~pynecore.core.plugin.broker.BrokerPlugin` class attributes that
they shadow.
"""
from dataclasses import dataclass
from pathlib import Path

from pynecore.core.config import ensure_config

__all__ = [
    'BrokerDefaults',
    'VALID_UNEXPECTED_CANCEL_POLICIES',
    'load_broker_defaults',
]


VALID_UNEXPECTED_CANCEL_POLICIES = frozenset({
    "stop",
    "stop_and_cancel",
    "re_place",
    "ignore",
})


@dataclass
class BrokerDefaults:
    """Cross-broker runtime defaults.

    Loaded from ``workdir/config/brokers.toml`` via
    :func:`load_broker_defaults`. The file is self-healing — fields at
    their default are emitted as commented-out lines, user-edited values
    are preserved across regenerations.
    """

    on_unexpected_cancel: str = "stop"
    """Policy when a bot-owned order disappears without the bot cancelling it.

    ``"stop"`` (default) — graceful stop; surfaced to the runner via the
    event sink so observability can page.
    ``"stop_and_cancel"`` — stop plus a best-effort cancel pass over the
    remaining bot-owned orders.
    ``"re_place"`` — no-op on the cancel; the sync engine re-dispatches
    the protective order on the next diff cycle.
    ``"ignore"`` — silently continue. Only safe when manual external
    cancellations are an expected part of the operational workflow.
    """

    require_one_way_mode: bool = True
    """When True (the default), the startup probe fails closed if the
    account has hedging mode enabled. The base :class:`BrokerPlugin`
    semantics are one-way Pine — hedging mode belongs on a future
    ``HedgeBrokerPlugin`` subclass. Plugins on spot venues without a
    hedging concept (e.g. Coinbase) leave this flag inert."""


def load_broker_defaults(config_dir: Path) -> BrokerDefaults:
    """Load :class:`BrokerDefaults` from ``<config_dir>/brokers.toml``.

    Delegates to :func:`pynecore.core.config.ensure_config`, which
    auto-creates the file with commented defaults on first run, preserves
    user-edited values across regenerations, and caches the result on the
    dataclass. Validation runs after loading — invalid values raise
    :class:`ValueError` with the list of accepted policies so the
    misconfiguration surfaces immediately at startup, not at the first
    reconcile cycle.

    :param config_dir: The ``workdir/config`` directory.
    :return: A populated :class:`BrokerDefaults` instance.
    :raises ValueError: If a loaded value falls outside its allowed set.
    """
    instance = ensure_config(BrokerDefaults, config_dir / 'brokers.toml')
    assert isinstance(instance, BrokerDefaults)
    if instance.on_unexpected_cancel not in VALID_UNEXPECTED_CANCEL_POLICIES:
        raise ValueError(
            f"brokers.toml: on_unexpected_cancel must be one of "
            f"{sorted(VALID_UNEXPECTED_CANCEL_POLICIES)}, got "
            f"{instance.on_unexpected_cancel!r}",
        )
    return instance
