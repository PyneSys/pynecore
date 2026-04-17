"""
Canonical ``client_order_id`` formula for BrokerPlugin dispatches.

The broker layer uses a deterministic identifier so that retries, reconnects
and even full process restarts converge on the same exchange-side client
order id — idempotent by construction.

The format is a 30-character budget, chosen as the lowest common denominator
across supported exchanges: Capital.com ``dealReference`` is ≤ 30 chars;
Binance ``newClientOrderId``, Bybit ``orderLinkId``, OKX ``clOrdId``,
Interactive Brokers ``orderRef`` and Deribit ``label`` all accept at least
that many characters. A single string therefore fits every plugin without
per-exchange branching.

Format::

    {run}-{pid}-{bar}-{k}{r}

=====  ==========  ============================================================
Field  Width       Content
=====  ==========  ============================================================
run    4 base36    Session-stable hash of the script source / config.
pid    8 base36    Hash of the Pine-level order id (``pine_id``).
bar    9 base36    Bar open timestamp in milliseconds since the Unix epoch.
k      1           Single-character kind code (entry / TP / SL / close / cancel).
r      1–2 base36  Retry sequence — normally ``0``; bumped only when a prior
                   attempt is deliberately abandoned.
=====  ==========  ============================================================

Every field but ``r`` is fixed-width, keeping the result deterministic across
processes and Python versions. The formula is **pure**: two dispatches for the
same logical order on the same bar always produce identical ids. Exchanges
that enforce client-id uniqueness reject the duplicate outright; exchanges
that do not (Interactive Brokers, Deribit) dedup inside the plugin via a
``get_open_orders`` match on the same id.
"""
from __future__ import annotations

import hashlib
from typing import Final

__all__ = [
    'KIND_ENTRY',
    'KIND_EXIT_TP',
    'KIND_EXIT_SL',
    'KIND_CLOSE',
    'KIND_CANCEL',
    'VALID_KINDS',
    'CLIENT_ORDER_ID_MAX_LEN',
    'RUN_TAG_WIDTH',
    'PINE_ID_HASH_WIDTH',
    'BAR_TS_WIDTH',
    'build_client_order_id',
    'hash_pine_id',
    'make_run_tag',
]

# === Kind codes (single-character) =======================================

KIND_ENTRY: Final[str] = 'e'
KIND_EXIT_TP: Final[str] = 't'
KIND_EXIT_SL: Final[str] = 's'
KIND_CLOSE: Final[str] = 'c'
KIND_CANCEL: Final[str] = 'x'

VALID_KINDS: Final[frozenset[str]] = frozenset({
    KIND_ENTRY, KIND_EXIT_TP, KIND_EXIT_SL, KIND_CLOSE, KIND_CANCEL,
})

# === Width constants =====================================================

CLIENT_ORDER_ID_MAX_LEN: Final[int] = 30
RUN_TAG_WIDTH: Final[int] = 4
PINE_ID_HASH_WIDTH: Final[int] = 8
BAR_TS_WIDTH: Final[int] = 9

# === Base36 encoding =====================================================

_BASE36_DIGITS: Final[str] = '0123456789abcdefghijklmnopqrstuvwxyz'


def _to_base36(value: int, *, width: int | None = None) -> str:
    """Encode a non-negative integer in lower-case base36.

    :param value: Non-negative integer to encode.
    :param width: When given, left-pad with ``'0'`` to this width. The
        function never truncates — an encoded value wider than ``width``
        raises :class:`ValueError`, because silent truncation would break
        determinism on overflow.
    :raises ValueError: On negative input or width overflow.
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if value == 0:
        encoded = '0'
    else:
        digits: list[str] = []
        n = value
        while n:
            digits.append(_BASE36_DIGITS[n % 36])
            n //= 36
        encoded = ''.join(reversed(digits))
    if width is not None:
        if len(encoded) > width:
            raise ValueError(
                f"encoded value {encoded!r} exceeds requested width {width}",
            )
        encoded = encoded.rjust(width, '0')
    return encoded


# === Public helpers ======================================================

def hash_pine_id(pine_id: str) -> str:
    """Return an 8-character base36 hash of a Pine-level order id.

    Pine ids can contain arbitrary characters (spaces, slashes, unicode) and
    arbitrary lengths; hashing fits the budget and neutralises odd input.
    40 bits of sha256 output are encoded, yielding ~1.1e12 distinct slots
    before a birthday collision becomes plausible — orders of magnitude above
    any realistic per-strategy id count.

    :param pine_id: The Pine order identifier (e.g. ``"Long"``, ``"TP/Long"``).
    :return: Exactly 8 lower-case base36 characters.
    """
    digest = hashlib.sha256(pine_id.encode('utf-8')).digest()
    # 40 bits → ceil(log36(2**40)) == 8 chars. A 41st bit would overflow 8.
    value = int.from_bytes(digest[:5], 'big')
    return _to_base36(value, width=PINE_ID_HASH_WIDTH)


def make_run_tag(script_source: str) -> str:
    """Return a 4-character base36 session tag for a script source.

    Two runs of the same script under the same config yield the same tag, so
    a restarted process reconstructs the same ``client_order_id`` for every
    logical intent — the exchange then dedups the retry as a duplicate.

    :param script_source: The Pine script source text (or any string the
        caller wants to participate in session identity — add a config hash
        to the input if config changes should invalidate the tag).
    :return: Exactly 4 lower-case base36 characters.
    """
    digest = hashlib.sha256(script_source.encode('utf-8')).digest()
    # 20 bits → ceil(log36(2**20)) == 4 chars. Keeps ~1M distinct tags.
    value = int.from_bytes(digest[:3], 'big') & 0x0FFFFF
    return _to_base36(value, width=RUN_TAG_WIDTH)


def build_client_order_id(
        *,
        run_tag: str,
        pine_id: str,
        bar_ts_ms: int,
        kind: str,
        retry_seq: int = 0,
) -> str:
    """Build the canonical client-order-id for a broker dispatch.

    :param run_tag: 4-char base36 session tag (see :func:`make_run_tag`).
    :param pine_id: Pine-level order identifier; hashed internally.
    :param bar_ts_ms: Bar open timestamp in milliseconds since the Unix epoch.
        Must be non-negative.
    :param kind: One of :data:`KIND_ENTRY`, :data:`KIND_EXIT_TP`,
        :data:`KIND_EXIT_SL`, :data:`KIND_CLOSE`, :data:`KIND_CANCEL`.
    :param retry_seq: Bumped only when the sync engine deliberately abandons
        a prior attempt (e.g. the exchange never acknowledged the original
        dispatch and the recovery timeout expired). ``0`` by default.
    :raises ValueError: On malformed ``run_tag`` / ``kind``, negative ``bar_ts_ms``
        / ``retry_seq``, or when the formatted id would exceed
        :data:`CLIENT_ORDER_ID_MAX_LEN` (indicates ``retry_seq`` overflow).
    """
    if len(run_tag) != RUN_TAG_WIDTH or not run_tag.isascii() or not run_tag.isalnum():
        raise ValueError(
            f"run_tag must be {RUN_TAG_WIDTH} alphanumeric ASCII chars, "
            f"got {run_tag!r}",
        )
    if kind not in VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(VALID_KINDS)}, got {kind!r}",
        )
    if bar_ts_ms < 0:
        raise ValueError(f"bar_ts_ms must be non-negative, got {bar_ts_ms}")
    if retry_seq < 0:
        raise ValueError(f"retry_seq must be non-negative, got {retry_seq}")

    pid = hash_pine_id(pine_id)
    bar = _to_base36(bar_ts_ms, width=BAR_TS_WIDTH)
    retry = _to_base36(retry_seq)

    result = f"{run_tag}-{pid}-{bar}-{kind}{retry}"
    if len(result) > CLIENT_ORDER_ID_MAX_LEN:
        raise ValueError(
            f"client_order_id exceeds {CLIENT_ORDER_ID_MAX_LEN} chars "
            f"(got {len(result)}); retry_seq={retry_seq} overflows the budget",
        )
    return result
