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

Wire form for short-budget venues
---------------------------------

A venue whose client-id limit is below the canonical width (some FIX
implementations cap ``ClOrdID`` at 20 characters) cannot carry the canonical
id. For those, :func:`encode_wire_client_order_id` derives a fixed-length
**wire form** that exactly fills the venue budget declared by the plugin
(:attr:`~pynecore.core.plugin.broker.BrokerPlugin.client_order_id_max_len`)::

    {run}{bar}{k}{hash}

=====  ==================  ====================================================
Field  Width               Content
=====  ==================  ====================================================
run    4 base36            Same session tag as the canonical form, raw.
bar    9 base36            Same bar open timestamp (ms), raw.
k      1                   Same single-character kind code, raw.
hash   budget - 14 base36  sha256 of the FULL canonical id, base36-encoded.
=====  ==================  ====================================================

The three raw fields keep the restart adoption path cheap: a wire id echoed
by the broker still reveals *whose run*, *which bar* and *which kind* it is,
so recognising a lost anchor only has to forward-hash ``retry_seq``
candidates for a known ``pine_id`` and compare against the opaque tail. The
tail simultaneously carries the hashed ``pid`` / ``retry`` identity and
confirms the full canonical match — an equal wire string is an equal logical
order (modulo a >=31-bit hash collision; :data:`WIRE_CLIENT_ORDER_ID_MIN_LEN`
floors the budget at 20 so the tail never drops below 6 characters).

The mapping is deterministic and applied at every mint site through
:meth:`~pynecore.core.broker.models.DispatchEnvelope.client_order_id`, so
journal rows, broker echoes and rebuilt references all agree on the wire
form. A canonical id that already fits the budget is passed through
unchanged — venues accepting >= 30 characters are byte-for-byte unaffected.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Final

__all__ = [
    'ParsedClientOrderId',
    'ParsedWireClientOrderId',
    'parse_client_order_id',
    'parse_wire_client_order_id',
    'encode_wire_client_order_id',
    'WIRE_CLIENT_ORDER_ID_MIN_LEN',
    'WIRE_RAW_PREFIX_LEN',
    'KIND_ENTRY',
    'KIND_ENTRY_STOP',
    'KIND_ENTRY_STOP_WATCH',
    'KIND_EXIT_TP',
    'KIND_EXIT_SL',
    'KIND_EXIT_TP_PARTIAL',
    'KIND_EXIT_SL_PARTIAL',
    'KIND_EXIT_TRAIL_PARTIAL',
    'KIND_CLOSE',
    'KIND_CANCEL',
    'KIND_MODIFY_ENTRY',
    'KIND_MODIFY_EXIT',
    'VALID_KINDS',
    'CLIENT_ORDER_ID_MAX_LEN',
    'RUN_TAG_WIDTH',
    'PINE_ID_HASH_WIDTH',
    'BAR_TS_WIDTH',
    'build_client_order_id',
    'hash_pine_id',
]

# === Kind codes (single-character) =======================================

KIND_ENTRY: Final[str] = 'e'
# Market entry fired by the software price-watch on the STOP side of a both-set
# Pine entry (``strategy.entry(limit=, stop=)``). Distinct from KIND_ENTRY so
# the stop-fired MARKET and the native LIMIT leg of the same ``pine_id`` get
# different client-order-ids — the engine persists this id before the POST so a
# crash-restart can verify-before-resend and never double-open.
KIND_ENTRY_STOP: Final[str] = 'b'
# Storage-only client-order-id for the engine-internal entry-stop WATCH row
# (no exchange order — the software state machine owns it, mirroring the
# partial-bracket leg rows). Distinct from KIND_ENTRY_STOP ('b'), which is the
# actual stop-fired MARKET order, so the watch row and the market order never
# share an ``orders`` table primary key.
KIND_ENTRY_STOP_WATCH: Final[str] = 'w'
KIND_EXIT_TP: Final[str] = 't'
KIND_EXIT_SL: Final[str] = 's'
# Engine-trigger partial bracket leg kinds. Distinct lowercase codes —
# uppercase variants would collide with the native TP / SL / modify-exit
# codes on exchanges that case-normalise client ids, so the partial-bracket
# legs (which own no exchange-side order while armed) get their own letters.
KIND_EXIT_TP_PARTIAL: Final[str] = 'p'
KIND_EXIT_SL_PARTIAL: Final[str] = 'q'
KIND_EXIT_TRAIL_PARTIAL: Final[str] = 'l'
KIND_CLOSE: Final[str] = 'c'
KIND_CANCEL: Final[str] = 'x'
KIND_MODIFY_ENTRY: Final[str] = 'n'
KIND_MODIFY_EXIT: Final[str] = 'r'

VALID_KINDS: Final[frozenset[str]] = frozenset({
    KIND_ENTRY, KIND_ENTRY_STOP, KIND_ENTRY_STOP_WATCH,
    KIND_EXIT_TP, KIND_EXIT_SL,
    KIND_EXIT_TP_PARTIAL, KIND_EXIT_SL_PARTIAL, KIND_EXIT_TRAIL_PARTIAL,
    KIND_CLOSE, KIND_CANCEL,
    KIND_MODIFY_ENTRY, KIND_MODIFY_EXIT,
})

# === Width constants =====================================================

CLIENT_ORDER_ID_MAX_LEN: Final[int] = 30
RUN_TAG_WIDTH: Final[int] = 4
PINE_ID_HASH_WIDTH: Final[int] = 8
BAR_TS_WIDTH: Final[int] = 9

# Wire form: {run4}{bar9}{kind1} raw prefix + base36 sha256 tail (module
# docstring, "Wire form for short-budget venues").
WIRE_RAW_PREFIX_LEN: Final[int] = RUN_TAG_WIDTH + BAR_TS_WIDTH + 1
# Budget floor: 14 raw + >=6 hash chars (~31 bits). Below that the tail gets
# too weak to confirm identity on the restart adoption path.
WIRE_CLIENT_ORDER_ID_MIN_LEN: Final[int] = 20

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


def build_client_order_id(
        *,
        run_tag: str,
        pine_id: str,
        bar_ts_ms: int,
        kind: str,
        retry_seq: int = 0,
) -> str:
    """Build the canonical client-order-id for a broker dispatch.

    :param run_tag: 4-char base36 session tag (see
        :meth:`~pynecore.core.broker.run_identity.RunIdentity.make_run_tag`).
    :param pine_id: Pine-level order identifier; hashed internally.
    :param bar_ts_ms: Bar open timestamp in milliseconds since the Unix epoch.
        Must be non-negative.
    :param kind: One of the single-character codes in
        :data:`VALID_KINDS` (entry / TP / SL / close / cancel /
        modify-entry / modify-exit).
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


def encode_wire_client_order_id(coid: str, max_len: int) -> str:
    """Encode a canonical client-order-id for a venue's client-id budget.

    Identity when the canonical id fits (``len(coid) <= max_len``) — venues
    accepting the full canonical width are unaffected. Otherwise returns the
    fixed-length wire form ``{run4}{bar9}{kind}{hash}`` of exactly ``max_len``
    characters (module docstring, "Wire form for short-budget venues"). Pure
    and deterministic like :func:`build_client_order_id`, so retries,
    restarts and journal-rebuilt references converge on the same wire id.

    :param coid: A canonical id from :func:`build_client_order_id`.
    :param max_len: The venue's client-id budget (the plugin's
        ``client_order_id_max_len``). Must be at least
        :data:`WIRE_CLIENT_ORDER_ID_MIN_LEN` when shortening is needed.
    :raises ValueError: When ``coid`` is not a well-formed canonical id, or
        the budget is below the wire floor.
    """
    if len(coid) <= max_len:
        return coid
    if max_len < WIRE_CLIENT_ORDER_ID_MIN_LEN:
        raise ValueError(
            f"client-id budget {max_len} is below the wire floor "
            f"{WIRE_CLIENT_ORDER_ID_MIN_LEN}",
        )
    parsed = parse_client_order_id(coid)
    if parsed is None:
        raise ValueError(f"not a canonical client_order_id: {coid!r}")
    hash_width = max_len - WIRE_RAW_PREFIX_LEN
    digest = hashlib.sha256(coid.encode('utf-8')).digest()
    tail_value = int.from_bytes(digest, 'big') % (36 ** hash_width)
    tail = _to_base36(tail_value, width=hash_width)
    bar = _to_base36(parsed.bar_ts_ms, width=BAR_TS_WIDTH)
    return f"{parsed.run_tag}{bar}{parsed.kind}{tail}"


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedClientOrderId:
    """Structural decomposition of a canonical client-order-id.

    The ``pine_id`` is irrecoverable — :func:`hash_pine_id` is one-way — so
    :attr:`pid_hash` carries the 8-char hash instead. A caller that knows a
    candidate ``pine_id`` matches by forward-hashing it
    (``hash_pine_id(candidate) == parsed.pid_hash``).
    """
    run_tag: str
    pid_hash: str
    bar_ts_ms: int
    kind: str
    retry_seq: int


def parse_client_order_id(coid: str) -> ParsedClientOrderId | None:
    """Parse a canonical client-order-id back into its structural fields.

    The inverse of :func:`build_client_order_id`, modulo the one-way
    ``pine_id`` hash (see :class:`ParsedClientOrderId`). Used by the sync
    engine's restart adoption path to recognise the bot's own live broker
    orders from their echoed ``client_order_id`` and recover the
    ``(bar_ts_ms, retry_seq)`` anchor a crash dropped before it was
    journaled.

    Best-effort and total: any input that does not match the
    ``{run4}-{pid8}-{bar9}-{kind}{retry}`` shape — wrong dash count, wrong
    field widths, an unknown ``kind`` code, or a non-base36 ``bar`` /
    ``retry`` — yields ``None`` rather than raising, so a caller can pass an
    externally-owned order's id straight through.

    :param coid: The client-order-id to parse.
    :return: The decomposed fields, or ``None`` when ``coid`` is not a
        well-formed canonical id.
    """
    parts = coid.split('-')
    if len(parts) != 4:
        return None
    run_tag, pid_hash, bar_b36, kind_retry = parts
    if (len(run_tag) != RUN_TAG_WIDTH
            or len(pid_hash) != PINE_ID_HASH_WIDTH
            or len(bar_b36) != BAR_TS_WIDTH
            or len(kind_retry) < 2):
        return None
    kind = kind_retry[0]
    if kind not in VALID_KINDS:
        return None
    retry_b36 = kind_retry[1:]
    base36 = set(_BASE36_DIGITS)
    if not (set(bar_b36) <= base36 and set(retry_b36) <= base36):
        return None
    return ParsedClientOrderId(
        run_tag=run_tag,
        pid_hash=pid_hash,
        bar_ts_ms=int(bar_b36, 36),
        kind=kind,
        retry_seq=int(retry_b36, 36),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedWireClientOrderId:
    """Raw-prefix fields of a wire-form client-order-id.

    Only the fields carried verbatim in the wire prefix are recoverable —
    the ``pid`` hash and ``retry_seq`` live inside the opaque sha256 tail.
    A caller that knows a candidate ``(pine_id, retry_seq)`` matches by
    rebuilding the canonical id and re-encoding it at the echoed id's length
    (``encode_wire_client_order_id(candidate, len(coid)) == coid``).
    """
    run_tag: str
    bar_ts_ms: int
    kind: str


def parse_wire_client_order_id(coid: str) -> ParsedWireClientOrderId | None:
    """Parse the raw prefix of a wire-form client-order-id.

    Best-effort and total like :func:`parse_client_order_id`: anything that
    does not match the ``{run4}{bar9}{kind}{hash>=6}`` all-base36 shape —
    too short, containing a dash (canonical ids always do), an unknown
    ``kind`` code — yields ``None``. Budget-independent: the wire form's
    length always equals the minting venue's budget, so the parser only
    enforces the :data:`WIRE_CLIENT_ORDER_ID_MIN_LEN` floor.

    :param coid: The client-order-id to parse.
    :return: The raw-prefix fields, or ``None`` when ``coid`` is not a
        well-formed wire id.
    """
    if len(coid) < WIRE_CLIENT_ORDER_ID_MIN_LEN:
        return None
    if not (set(coid) <= set(_BASE36_DIGITS)):
        return None
    kind = coid[WIRE_RAW_PREFIX_LEN - 1]
    if kind not in VALID_KINDS:
        return None
    return ParsedWireClientOrderId(
        run_tag=coid[:RUN_TAG_WIDTH],
        bar_ts_ms=int(coid[RUN_TAG_WIDTH:RUN_TAG_WIDTH + BAR_TS_WIDTH], 36),
        kind=kind,
    )
