"""
Run identity — logical and physical keys for a broker run.

A *run* is one concrete execution of a Pine strategy against a live broker
account for a specific (symbol, timeframe, optional label) combination.
Two keys exist for different purposes:

- **``run_id``** — human-readable, deterministic stream identifier. The same
  strategy + account + symbol + timeframe + label combination always
  yields the same value. This is the key for historical lookups
  (*"show every past run of this bot"*).
- **``run_instance_id``** — physical autoincrement INTEGER, unique to each
  process invocation. FK column of every storage table. The storage
  populates it; this module is not concerned with it.

``RunIdentity`` is the construction input for both keys: the strategy
runner assembles it from the CLI + plugin auth, hands it to the storage,
and gets a ``RunContext`` back with the ``run_instance_id`` attached.

Why a separate module from ``idempotency.py``: idempotency owns the
bit-level shape of ``client_order_id`` (broker protocol), while
``RunIdentity`` owns run-scoped identity (PyneCore-internal). Two
distinct abstraction layers — merging them in one module would be
opaque.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Final

from pynecore.core.broker.idempotency import RUN_TAG_WIDTH, _to_base36

__all__ = [
    'RunIdentity',
]

# 20 bits → 4 base36 chars, the same range as the original
# ``make_run_tag``. The wider input only improves the collision space;
# the output format (``{run_tag}-{pid}-...``) is unchanged.
_RUN_TAG_BITS: Final[int] = 20
_RUN_TAG_MASK: Final[int] = (1 << _RUN_TAG_BITS) - 1


@dataclass(frozen=True)
class RunIdentity:
    """Identity of a concrete bot run, prior to storage.

    :param strategy_id: Stem of the script file (e.g. ``"ema_cross"``).
        The logical name of the codebase, NOT a version hash —
        source-code changes are captured by ``run_tag`` via the
        ``make_run_tag`` hash input.
    :param symbol: Traded instrument (e.g. ``"EURUSD"``).
    :param timeframe: TradingView format (e.g. ``"60"``, ``"1D"``).
    :param account_id: Plugin-qualified broker-account identifier, e.g.
        ``"capitalcom-demo-1234567"``. Provided by the plugin's sync
        ``.account_id`` property, populated during authentication.
        Defaults to ``"default"`` when no account is available.
    :param label: Optional user override (CLI ``--run-label``). Used to
        distinguish multiple instances of the same
        (strategy_id, symbol, timeframe, account_id) combination.
        Default: ``None``.
    """
    strategy_id: str
    symbol: str
    timeframe: str
    account_id: str
    label: str | None = None

    @property
    def run_id(self) -> str:
        """Human-readable logical key.

        Format: ``"{strategy_id}@{account_id}:{symbol}:{timeframe}"`` or,
        when a label is provided, ``"...#{label}"``. A separate ``#``
        separator is required because ``:`` can appear after the
        timeframe (e.g. ``"1D"``); ``#`` on the other hand is not legal
        even inside a label that comes through the CLI flag (the CLI
        validates this).
        """
        base = f"{self.strategy_id}@{self.account_id}:{self.symbol}:{self.timeframe}"
        return f"{base}#{self.label}" if self.label else base

    def make_run_tag(self, script_source: str) -> str:
        """4-char base36 session tag for client_order_id.

        The idempotency formula (``{run}-{pid}-{bar}-{k}{r}``) reserves 4
        characters for this field, giving 20 bits of capacity
        (~1M slots). Deterministic: the same input always produces the
        same tag.

        Why more inputs than the old ``make_run_tag(script_source)``:
        the same script running on two timeframes simultaneously
        produced identical tags, causing idempotency collisions
        (identical ``client_order_id``s). Including ``strategy_id`` /
        ``symbol`` / ``timeframe`` / ``account`` / ``label`` extends the
        collision space across the realistic run dimensions.
        ``strategy_id`` is also needed on its own: two distinct scripts
        with identical sources (copy-paste) running on the same
        (symbol, tf, account) would otherwise share a tag and emit
        duplicate ``client_order_id``s at the broker.

        The JSON-serialised input gives a deterministic stringification
        free of pipe/quote/unicode edge cases.

        :param script_source: Source code of the Pine script as the
            runner read it.
        :return: Exactly 4 lower-case base36 characters.
        """
        payload = json.dumps(
            [
                self.strategy_id,
                script_source,
                self.symbol,
                self.timeframe,
                self.account_id,
                self.label or "",
            ],
            ensure_ascii=True,
            sort_keys=False,
        )
        digest = hashlib.sha256(payload.encode('ascii')).digest()
        # 20 bit → 4 char base36. A 3-byte slice covers it with margin;
        # we AND-mask afterwards.
        value = int.from_bytes(digest[:3], 'big') & _RUN_TAG_MASK
        return _to_base36(value, width=RUN_TAG_WIDTH)
