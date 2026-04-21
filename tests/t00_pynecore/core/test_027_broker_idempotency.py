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
    RUN_TAG_WIDTH,
    VALID_KINDS,
    build_client_order_id,
    hash_pine_id,
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
    assert hash_pine_id('Long') == hash_pine_id('Long')
    assert hash_pine_id('TP/Long') == hash_pine_id('TP/Long')


def __test_make_run_tag_is_deterministic__():
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
    assert len(hash_pine_id('')) == PINE_ID_HASH_WIDTH
    assert len(hash_pine_id('Long')) == PINE_ID_HASH_WIDTH
    assert len(hash_pine_id('x' * 10_000)) == PINE_ID_HASH_WIDTH


def __test_make_run_tag_is_exactly_expected_width__():
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
    base = dict(run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000)
    ids = {build_client_order_id(**base, kind=k) for k in VALID_KINDS}
    assert len(ids) == len(VALID_KINDS)


def __test_different_pine_id_different_id__():
    kwargs = dict(run_tag='abcd', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY)
    a = build_client_order_id(**kwargs, pine_id='Long')
    b = build_client_order_id(**kwargs, pine_id='Short')
    assert a != b


def __test_different_bar_ts_different_id__():
    kwargs = dict(run_tag='abcd', pine_id='Long', kind=KIND_ENTRY)
    a = build_client_order_id(**kwargs, bar_ts_ms=1_700_000_000_000)
    b = build_client_order_id(**kwargs, bar_ts_ms=1_700_000_060_000)
    assert a != b


def __test_different_run_tag_different_id__():
    kwargs = dict(pine_id='Long', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY)
    a = build_client_order_id(run_tag='abcd', **kwargs)
    b = build_client_order_id(run_tag='efgh', **kwargs)
    assert a != b


def __test_retry_seq_changes_id__():
    kwargs = dict(
        run_tag='abcd', pine_id='Long', bar_ts_ms=1_700_000_000_000, kind=KIND_ENTRY,
    )
    a = build_client_order_id(**kwargs, retry_seq=0)
    b = build_client_order_id(**kwargs, retry_seq=1)
    assert a != b
    assert a.endswith('0')
    assert b.endswith('1')


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
    with pytest.raises(ValueError, match='run_tag'):
        build_client_order_id(
            run_tag=bad_run_tag,
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=KIND_ENTRY,
        )


@pytest.mark.parametrize('bad_kind', ['', 'entry', 'E', 'exit', 'z'])
def __test_build_rejects_bad_kind__(bad_kind):
    with pytest.raises(ValueError, match='kind'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=1_700_000_000_000,
            kind=bad_kind,
        )


def __test_build_rejects_negative_bar_ts__():
    with pytest.raises(ValueError, match='bar_ts_ms'):
        build_client_order_id(
            run_tag='abcd',
            pine_id='Long',
            bar_ts_ms=-1,
            kind=KIND_ENTRY,
        )


def __test_build_rejects_negative_retry_seq__():
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
    out = build_client_order_id(
        run_tag='abcd', pine_id='L', bar_ts_ms=1, kind=kind,
    )
    assert out.endswith(expected_suffix)
