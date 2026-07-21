"""
Unit tests for :mod:`pynecore.core.broker.idempotency`.

The ``client_order_id`` formula underpins every broker dispatch: retries,
reconnects and full process restarts all rely on the same inputs producing
byte-identical ids. These tests pin down the determinism, the 30-character
budget, collision resistance, and the error paths.
"""
import random
import string

import pytest

from pynecore.core.broker.idempotency import (
    BAR_TS_WIDTH,
    CLIENT_ORDER_ID_MAX_LEN,
    KIND_CANCEL,
    KIND_CLOSE,
    KIND_ENTRY,
    KIND_EXIT_SL,
    KIND_EXIT_TP,
    PINE_ID_HASH_WIDTH,
    ParsedClientOrderId,
    RUN_TAG_WIDTH,
    VALID_KINDS,
    WIRE_CLIENT_ORDER_ID_MIN_LEN,
    WIRE_RAW_PREFIX_LEN,
    build_client_order_id,
    encode_wire_client_order_id,
    hash_pine_id,
    parse_client_order_id,
    parse_wire_client_order_id,
)
from pynecore.core.broker.run_identity import RunIdentity


def _make_run_tag(script_source: str) -> str:
    """Helper that adapts the old 1-arg ``make_run_tag`` signature to the new
    :class:`RunIdentity`-based tag derivation, for test assertions that only
    care about the ``script_source`` dimension."""
    identity = RunIdentity(
        strategy_id="test", symbol="S", timeframe="60",
        account_id="default", label=None,
    )
    return identity.make_run_tag(script_source)

# === Determinism =========================================================


def __test_build_is_deterministic_across_calls__():
    """Same inputs must always produce the same id — across calls and time."""
    kwargs = dict(
        run_tag='ab12',
        pine_id='Long',
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY,
    )
    first = build_client_order_id(**kwargs)
    for _ in range(10):
        assert build_client_order_id(**kwargs) == first


def __test_hash_pine_id_is_deterministic__():
    """``hash_pine_id`` returns the same hash for repeated calls on equal input."""
    assert hash_pine_id('Long') == hash_pine_id('Long')
    assert hash_pine_id('TP/Long') == hash_pine_id('TP/Long')


def __test_make_run_tag_is_deterministic__():
    """The run tag is identical for repeated derivations from the same script source."""
    assert _make_run_tag('strategy("x")\nplot(close)') == _make_run_tag('strategy("x")\nplot(close)')


# === Length budget =======================================================


def __test_result_fits_30_char_budget__():
    """Common-case inputs must stay within the Capital.com ``dealReference`` limit."""
    for kind in VALID_KINDS:
        result = build_client_order_id(
            run_tag='abcd',
            pine_id='SomeVeryLongPineIdentifierThatDoesNotMatter',
            bar_ts_ms=9_999_999_999_999,  # Year 2286
            kind=kind,
            retry_seq=0,
        )
        assert len(result) <= CLIENT_ORDER_ID_MAX_LEN


def __test_hash_pine_id_is_exactly_expected_width__():
    """``hash_pine_id`` output is always ``PINE_ID_HASH_WIDTH`` chars, for any input length."""
    assert len(hash_pine_id('')) == PINE_ID_HASH_WIDTH
    assert len(hash_pine_id('Long')) == PINE_ID_HASH_WIDTH
    assert len(hash_pine_id('x' * 10_000)) == PINE_ID_HASH_WIDTH


def __test_make_run_tag_is_exactly_expected_width__():
    """The run tag is always ``RUN_TAG_WIDTH`` chars, for any script-source length."""
    assert len(_make_run_tag('')) == RUN_TAG_WIDTH
    assert len(_make_run_tag('strategy("x")')) == RUN_TAG_WIDTH
    assert len(_make_run_tag('y' * 100_000)) == RUN_TAG_WIDTH


def __test_bar_ts_is_exactly_expected_width__():
    """The bar segment is 9 chars regardless of ts magnitude."""
    for bar_ts in [0, 1, 1_700_000_000_000]:
        out = build_client_order_id(
            run_tag='abcd', pine_id='L', bar_ts_ms=bar_ts, kind=KIND_ENTRY,
        )
        # Segments: run(4) - pid(8) - bar(N) - k(1) + r(1) -> 4+1+8+1+N+1+1+1
        bar_segment = out.split('-')[2]
        assert len(bar_segment) == BAR_TS_WIDTH


# === Distinctness ========================================================


def __test_different_kind_different_id__():
    """Each ``kind`` yields a distinct id, so all ``VALID_KINDS`` map to unique results."""
    base = dict(run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000)
    ids = {build_client_order_id(**base, kind=k) for k in VALID_KINDS}
    assert len(ids) == len(VALID_KINDS)


def __test_different_pine_id_different_id__():
    """A different ``pine_id`` produces a different id with all other inputs equal."""
    kwargs = dict(run_tag='abcd', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY)
    a = build_client_order_id(**kwargs, pine_id='Long')
    b = build_client_order_id(**kwargs, pine_id='Short')
    assert a != b


def __test_different_bar_ts_different_id__():
    """A different ``bar_ts_ms`` produces a different id with all other inputs equal."""
    kwargs = dict(run_tag='abcd', pine_id='Long', kind=KIND_ENTRY)
    a = build_client_order_id(**kwargs, bar_ts_ms=1_700_000_000_000)
    b = build_client_order_id(**kwargs, bar_ts_ms=1_700_000_060_000)
    assert a != b


def __test_different_run_tag_different_id__():
    """A different ``run_tag`` produces a different id with all other inputs equal."""
    kwargs = dict(pine_id='Long', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY)
    a = build_client_order_id(run_tag='abcd', **kwargs)
    b = build_client_order_id(run_tag='efgh', **kwargs)
    assert a != b


def __test_retry_seq_changes_id__():
    """A different ``retry_seq`` changes the id, surfacing as the trailing base36 digit."""
    kwargs = dict(
        run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY,
    )
    a = build_client_order_id(**kwargs, retry_seq=0)
    b = build_client_order_id(**kwargs, retry_seq=1)
    assert a != b
    assert a.endswith('0')
    assert b.endswith('1')


# === DispatchEnvelope coid identity ======================================


def __test_exit_envelope_folds_from_entry_into_coid__():
    """Two global-exit brackets differing only in ``from_entry`` must get
    distinct coids.

    A pyramided position with one ``strategy.exit`` (no explicit
    ``from_entry``) fans out to one :class:`ExitIntent` per entry, all
    sharing the exit ``pine_id``. If the coid ignored ``from_entry`` both
    brackets would collapse onto the same deterministic id, the venue would
    dedup the second create, and half the position would be left
    unprotected.
    """
    from pynecore.core.broker.models import DispatchEnvelope, ExitIntent

    def _env(from_entry: str) -> DispatchEnvelope:
        return DispatchEnvelope(
            intent=ExitIntent(
                pine_id='EXIT', from_entry=from_entry, symbol='ETHUSDT',
                side='sell', qty=0.01, tp_price=1985.9, sl_price=1885.9,
            ),
            run_tag='abcd',
            bar_ts_ms=1_700_000_000_000,
        )

    env_a = _env('A')
    env_b = _env('B')
    assert env_a.client_order_id(KIND_EXIT_TP) != env_b.client_order_id(KIND_EXIT_TP)
    assert env_a.client_order_id(KIND_EXIT_SL) != env_b.client_order_id(KIND_EXIT_SL)
    # Same intent identity -> deterministic (stable across calls).
    assert env_a.client_order_id(KIND_EXIT_TP) == _env('A').client_order_id(KIND_EXIT_TP)


def __test_entry_envelope_coid_ignores_missing_from_entry__():
    """An :class:`EntryIntent` (no ``from_entry``) hashes the bare pine_id,
    unchanged by the exit-identity fold."""
    from pynecore.core.broker.models import DispatchEnvelope, EntryIntent
    from pynecore.core.broker.models import OrderType

    env = DispatchEnvelope(
        intent=EntryIntent(
            pine_id='Long', symbol='ETHUSDT', side='buy', qty=0.01,
            order_type=OrderType.MARKET,
        ),
        run_tag='abcd',
        bar_ts_ms=1_700_000_000_000,
    )
    expected = encode_wire_client_order_id(
        build_client_order_id(
            run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000,
            kind=KIND_ENTRY,
        ),
        env.coid_max_len,
    )
    assert env.client_order_id(KIND_ENTRY) == expected


# === Collision resistance ================================================


def __test_no_collision_across_1000_random_pine_ids__():
    """A birthday attack on 40 bits would take ~1M ids; 1000 must all be unique."""
    rng = random.Random(0xDEADBEEF)
    charset = string.ascii_letters + string.digits + '/_ '
    pine_ids: set[str] = set()
    while len(pine_ids) < 1000:
        pine_ids.add(
            ''.join(rng.choice(charset) for _ in range(rng.randint(1, 40))),
        )
    seen = {hash_pine_id(pid) for pid in pine_ids}
    assert len(seen) == 1000


def __test_no_collision_across_1000_bar_combinations__():
    """Different bars + kinds produce distinct ids under the same pine_id / run_tag."""
    rng = random.Random(42)
    seen: set[str] = set()
    for _ in range(1000):
        bar_ts = rng.randint(1_000_000_000_000, 2_000_000_000_000)
        kind = rng.choice(list(VALID_KINDS))
        seen.add(
            build_client_order_id(
                run_tag='abcd', pine_id='Long', bar_ts_ms=bar_ts, kind=kind,
            ),
        )
    assert len(seen) == 1000


# === Validation errors ===================================================


@pytest.mark.parametrize('bad_run_tag', [
    '',          # too short
    'ab',        # too short
    'abcde',     # too long
    'ab c',      # contains space
    'ab-d',      # contains hyphen
    'ábcd',      # non-ascii
    'ab_d',      # contains underscore
])
def __test_build_rejects_bad_run_tag__(bad_run_tag):
    """A ``run_tag`` of wrong length or with disallowed characters raises ``ValueError``."""
    with pytest.raises(ValueError, match='run_tag'):
        build_client_order_id(
            run_tag=bad_run_tag,
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=KIND_ENTRY,
        )


@pytest.mark.parametrize('bad_kind', ['', 'entry', 'E', 'exit', 'z'])
def __test_build_rejects_bad_kind__(bad_kind):
    """A ``kind`` outside ``VALID_KINDS`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='kind'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=bad_kind,
        )


def __test_build_rejects_negative_bar_ts__():
    """A negative ``bar_ts_ms`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='bar_ts_ms'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=-1,
            kind=KIND_ENTRY,
        )


def __test_build_rejects_negative_retry_seq__():
    """A negative ``retry_seq`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='retry_seq'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=KIND_ENTRY,
            retry_seq=-1,
        )


def __test_build_rejects_retry_seq_overflowing_30_char_budget__():
    """A retry_seq that would push the id past 30 chars must raise.

    The fixed part is 25 chars (``{4}-{8}-{9}-{1}``), leaving 5 chars for
    ``retry_seq``.  The first 6-char base36 value is ``36**5 == 60_466_176``.
    """
    with pytest.raises(ValueError, match='exceeds'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=KIND_ENTRY,
            retry_seq=36 ** 5,
        )


def __test_build_accepts_maximum_retry_seq_at_30_char_budget__():
    """The largest 5-char base36 retry (``36**5 - 1``) must still fit."""
    out = build_client_order_id(
        run_tag='abcd',
        pine_id='Long',
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY,
        retry_seq=36 ** 5 - 1,
    )
    assert len(out) == CLIENT_ORDER_ID_MAX_LEN


# === Edge cases ==========================================================


def __test_empty_pine_id_is_accepted__():
    """``strategy.close_all()`` has an empty pine_id — must still produce a valid id."""
    out = build_client_order_id(
        run_tag='abcd', pine_id='', bar_ts_ms=1_700_000_000_000, kind=KIND_CLOSE,
    )
    assert len(out) <= CLIENT_ORDER_ID_MAX_LEN


def __test_unicode_pine_id_is_accepted__():
    """The hash layer neutralises whatever bytes the user put in the id."""
    out = build_client_order_id(
        run_tag='abcd',
        pine_id='Belépő stratégia',
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY,
    )
    assert len(out) <= CLIENT_ORDER_ID_MAX_LEN


def __test_bar_ts_zero_is_accepted__():
    """A ``bar_ts_ms`` of 0 is accepted and encodes as nine zeros in the bar segment."""
    out = build_client_order_id(
        run_tag='abcd', pine_id='Long', bar_ts_ms=0, kind=KIND_ENTRY,
    )
    # Zero timestamp encodes as '000000000' (9 zeros).
    assert out == 'abcd-' + hash_pine_id('Long') + '-000000000-e0'


def __test_format_is_lowercase_ascii__():
    """Exchanges routinely lowercase client ids; staying lowercase avoids surprises."""
    out = build_client_order_id(
        run_tag='abcd',
        pine_id='SomeId',
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_EXIT_TP,
    )
    assert out == out.lower()


# === Kind-specific smoke =================================================


@pytest.mark.parametrize('kind,expected_suffix', [
    (KIND_ENTRY, 'e0'),
    (KIND_EXIT_TP, 't0'),
    (KIND_EXIT_SL, 's0'),
    (KIND_CLOSE, 'c0'),
    (KIND_CANCEL, 'x0'),
])
def __test_kind_appears_literally_in_result__(kind, expected_suffix):
    """Each ``kind`` ends the id with its expected one-char code plus the retry digit."""
    out = build_client_order_id(
        run_tag='abcd', pine_id='L', bar_ts_ms=1, kind=kind,
    )
    assert out.endswith(expected_suffix)


# === parse_client_order_id ===============================================


def __test_parse_round_trips_build__():
    """A built id parses back to its structural fields (pine_id is one-way)."""
    coid = build_client_order_id(
        run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY, retry_seq=1,
    )
    parsed = parse_client_order_id(coid)
    assert parsed == ParsedClientOrderId(
        run_tag='abcd',
        pid_hash=hash_pine_id('Long'),
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY,
        retry_seq=1,
    )


def __test_parse_retry_seq_zero_and_high__():
    """Parsing recovers the original ``retry_seq`` across the base36 1-to-2 digit boundary."""
    for retry in (0, 1, 35, 36, 100):
        coid = build_client_order_id(
            run_tag='abcd', pine_id='L', bar_ts_ms=1, kind=KIND_ENTRY,
            retry_seq=retry,
        )
        parsed = parse_client_order_id(coid)
        assert parsed is not None
        assert parsed.retry_seq == retry


def __test_parse_bar_ts_zero__():
    """Parsing an id built with ``bar_ts_ms=0`` recovers a ``bar_ts_ms`` of 0."""
    coid = build_client_order_id(
        run_tag='abcd', pine_id='L', bar_ts_ms=0, kind=KIND_ENTRY,
    )
    parsed = parse_client_order_id(coid)
    assert parsed is not None
    assert parsed.bar_ts_ms == 0


@pytest.mark.parametrize('bad', [
    '',
    'abcd',
    'abcd-1234',
    'abcd-12345678-000000001',          # missing kind+retry segment
    'abc-12345678-000000001-e0',        # run_tag too short
    'abcd-1234567-000000001-e0',        # pid_hash too short
    'abcd-12345678-00000001-e0',        # bar too short (8 not 9)
    'abcd-12345678-000000001-',         # empty kind+retry
    'abcd-12345678-000000001-z0',       # unknown kind code
    'abcd-12345678-00000000!-e0',       # non-base36 bar
    'abcd-12345678-000000001-e!',       # non-base36 retry
])
def __test_parse_malformed_returns_none__(bad):
    """A structurally malformed id (bad segment lengths, codes, or charset) parses to ``None``."""
    assert parse_client_order_id(bad) is None


def __test_parse_externally_owned_id_returns_none__():
    """A foreign exchange id that is not our canonical shape parses to None."""
    assert parse_client_order_id('DEAL-REF-FROM-CAPITAL-1234') is None


# === Wire form (short-budget venues) =====================================


def _canonical(**overrides) -> str:
    kwargs = dict(
        run_tag='ab12',
        pine_id='Long',
        bar_ts_ms=1_700_000_000_000,
        kind=KIND_ENTRY,
    )
    kwargs.update(overrides)
    return build_client_order_id(**kwargs)


def __test_encode_wire_identity_when_budget_fits__():
    """A canonical id within the budget passes through byte-identical."""
    coid = _canonical()
    assert encode_wire_client_order_id(coid, CLIENT_ORDER_ID_MAX_LEN) == coid
    assert encode_wire_client_order_id(coid, len(coid)) == coid


def __test_encode_wire_exact_budget_length_and_charset__():
    """The wire form fills the venue budget exactly, all lowercase base36."""
    for budget in (WIRE_CLIENT_ORDER_ID_MIN_LEN, 22, 25):
        wire = encode_wire_client_order_id(_canonical(), budget)
        assert len(wire) == budget
        assert set(wire) <= set(string.digits + string.ascii_lowercase)
        assert '-' not in wire


def __test_encode_wire_is_deterministic__():
    """Same canonical id + budget must always yield the same wire id."""
    coid = _canonical()
    first = encode_wire_client_order_id(coid, 20)
    for _ in range(10):
        assert encode_wire_client_order_id(coid, 20) == first


def __test_encode_wire_raw_prefix_carries_run_bar_kind__():
    """``{run4}{bar9}{kind}`` are verbatim in the wire prefix."""
    bar_ts_ms = 1_700_000_000_000
    wire = encode_wire_client_order_id(
        _canonical(bar_ts_ms=bar_ts_ms, kind=KIND_EXIT_TP), 20,
    )
    assert wire[:RUN_TAG_WIDTH] == 'ab12'
    assert int(wire[RUN_TAG_WIDTH:RUN_TAG_WIDTH + BAR_TS_WIDTH], 36) == bar_ts_ms
    assert wire[WIRE_RAW_PREFIX_LEN - 1] == KIND_EXIT_TP


def __test_encode_wire_distinct_for_distinct_identity__():
    """pid / retry / kind / bar variations produce distinct wire ids."""
    base = encode_wire_client_order_id(_canonical(), 20)
    for variant in (
        _canonical(pine_id='Short'),
        _canonical(retry_seq=1),
        _canonical(kind=KIND_EXIT_SL),
        _canonical(bar_ts_ms=1_700_000_060_000),
    ):
        assert encode_wire_client_order_id(variant, 20) != base


def __test_encode_wire_no_collision_across_random_pine_ids__():
    """1000 random pine_ids stay collision-free in a 20-char budget."""
    rng = random.Random(42)
    seen = set()
    for _ in range(1000):
        pine_id = ''.join(
            rng.choices(string.ascii_letters + string.digits, k=12),
        )
        seen.add(encode_wire_client_order_id(_canonical(pine_id=pine_id), 20))
    assert len(seen) == 1000


def __test_encode_wire_rejects_budget_below_floor__():
    """A budget below the wire floor must raise, never truncate."""
    with pytest.raises(ValueError):
        encode_wire_client_order_id(
            _canonical(), WIRE_CLIENT_ORDER_ID_MIN_LEN - 1,
        )


def __test_encode_wire_rejects_non_canonical_input__():
    """Only a well-formed canonical id may be wire-encoded."""
    with pytest.raises(ValueError):
        encode_wire_client_order_id('x' * 40, 20)


def __test_parse_wire_round_trips_raw_prefix__():
    """Wire parse recovers run_tag / bar_ts_ms / kind from an encoded id."""
    bar_ts_ms = 1_700_000_000_000
    wire = encode_wire_client_order_id(
        _canonical(bar_ts_ms=bar_ts_ms, kind=KIND_CLOSE), 20,
    )
    parsed = parse_wire_client_order_id(wire)
    assert parsed is not None
    assert parsed.run_tag == 'ab12'
    assert parsed.bar_ts_ms == bar_ts_ms
    assert parsed.kind == KIND_CLOSE


def __test_parse_wire_rejects_canonical_id__():
    """A canonical id (dashes) never parses as a wire id."""
    assert parse_wire_client_order_id(_canonical()) is None


@pytest.mark.parametrize('bad', [
    '',
    'ab12' + '0' * 9 + 'e',              # below the wire floor
    'ab12' + '0' * 9 + 'z' + '0' * 6,    # unknown kind code
    'AB12' + '0' * 9 + 'e' + '0' * 6,    # uppercase (not base36 lowercase)
    'ab1!' + '0' * 9 + 'e' + '0' * 6,    # non-base36 charset
])
def __test_parse_wire_malformed_returns_none__(bad):
    """Malformed / foreign ids parse to None instead of raising."""
    assert parse_wire_client_order_id(bad) is None


def __test_wire_forward_match_recovers_identity__():
    """The restart adoption primitive: rebuilding a candidate canonical id
    and re-encoding at the echoed id's length equals the echoed id exactly
    for the true ``(pine_id, retry_seq)`` and no other candidate."""
    echoed = encode_wire_client_order_id(_canonical(retry_seq=3), 20)
    matches = [
        (pine_id, retry_seq)
        for pine_id in ('Long', 'Short', 'Scalp')
        for retry_seq in range(36)
        if encode_wire_client_order_id(
            _canonical(pine_id=pine_id, retry_seq=retry_seq), len(echoed),
        ) == echoed
    ]
    assert matches == [('Long', 3)]
