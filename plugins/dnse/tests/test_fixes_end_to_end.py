"""End-to-end verification of the #19 and #20 fixes against a venue that REPRODUCES the
measured DNSE behaviour (``testing/fake_venue.py``).

These are not mocks of the plugin's own methods: a real :class:`DNSEBroker` runs the real
path — ``_place`` -> ``_cancel_one`` -> ``_cancel_took_effect`` -> ``_cancel_dependent_exits``
-> ``get_open_orders`` — against a fake that answers exactly as DNSE did on 2026-08-13
(cancel 200 is an ACK; the venue never cascades). That closes the gap left by the live
session ending: the scenarios can be reconstructed at any hour.

Each fix is verified BOTH ways — the bug is reproduced with the fix disabled, then shown to
be gone with it enabled — so a passing test proves the fix is necessary, not merely present.
"""
import asyncio
import sys
from pathlib import Path

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from fake_venue import FakeDNSEVenue                       # noqa: E402

from pynecore_dnse import broker                            # noqa: E402
from pynecore.core.broker.models import LegType, CancelIntent, DispatchEnvelope  # noqa: E402


def _broker(venue) -> broker.DNSEBroker:
    cfg = broker.DNSEBrokerConfig(api_key="k", api_secret="s",
                                  account_no="ACC1", trading_token="tok")
    b = broker.DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    b._client = venue
    b._cancel_verify_delay = 0.0          # no real sleeping in tests
    return b


def _cancel_env(pine_id: str = "L"):
    """CancelIntent derives intent_key from pine_id, which is the ``_order_ids`` key."""
    return DispatchEnvelope(
        intent=CancelIntent(pine_id=pine_id, symbol="VN30F1M", from_entry=None),
        run_tag="t", bar_ts_ms=1, retry_seq=0, coid_max_len=30)


def _place_entry_with_exit(b, venue):
    """Place an entry + a stop exit bound to it, exactly as TEST 4 did live."""
    env = _cancel_env("L")
    entry = b._place(env, "sell", 1.0, price=2010.6, leg_type=LegType.ENTRY)[0]
    exit_ = b._place(env, "buy", 1.0, price=2020.4, category="STOP",
                     stop_price=2020.2, leg_type=LegType.STOP_LOSS)[0]
    # the identity binding the engine would have recorded
    b._identity[str(entry.id)] = ("L", None, LegType.ENTRY)
    b._identity[str(exit_.id)] = ("X", "L", LegType.STOP_LOSS)
    b._order_ids[env.intent.intent_key] = [str(entry.id)]
    return str(entry.id), str(exit_.id)


# === #20 — a 2xx cancel is only an ACK =======================================

def __test_20_bug_reproduced_when_the_fix_is_disabled__():
    """With the readback bypassed (the pre-902c156 contract), the plugin lies."""
    venue = FakeDNSEVenue(ack_lag=3)          # as measured: still New for 3 reads
    b = _broker(venue)
    env = _cancel_env("L")
    oid = str(b._place(env, "buy", 1.0, price=2020.4, category="STOP",
                       stop_price=2020.2, leg_type=LegType.STOP_LOSS)[0].id)

    b._cancel_took_effect = lambda *_a: True   # <- the old behaviour: trust the 2xx

    assert b._cancel_one(oid) is True, "old contract reports success on the ACK"
    assert venue.orders[oid]["orderStatus"] == "New", \
        "...while the order is still WORKING at the venue — this is bug #20"


def __test_20_fixed_confirms_only_once_the_venue_agrees__():
    venue = FakeDNSEVenue(ack_lag=3)
    b = _broker(venue)
    env = _cancel_env("L")
    oid = str(b._place(env, "buy", 1.0, price=2020.4, category="STOP",
                       stop_price=2020.2, leg_type=LegType.STOP_LOSS)[0].id)
    b._cancel_verify_attempts = 6

    assert b._cancel_one(oid) is True
    assert venue.orders[oid]["orderStatus"] == "Canceled", \
        "the fix must not report success until the venue itself says terminal"
    assert sum(1 for c in venue.calls if c[0] == "get_order_detail") >= 3, \
        "it must actually poll the venue back, not trust the ACK"


def __test_20_reports_failure_when_the_venue_never_applies_the_cancel__():
    """Budget exhausted -> report NOT cancelled so the engine retries."""
    venue = FakeDNSEVenue(ack_lag=99)         # venue never gets around to it
    b = _broker(venue)
    env = _cancel_env("L")
    oid = str(b._place(env, "buy", 1.0, price=2020.4, category="STOP",
                       stop_price=2020.2, leg_type=LegType.STOP_LOSS)[0].id)
    b._cancel_verify_attempts = 3

    assert b._cancel_one(oid) is False, \
        "an unconfirmed cancel must be reported as NOT done, never as success"
    assert venue.orders[oid]["orderStatus"] == "New"


# === #19 — the venue does not cascade, and neither may the PLUGIN =============
# The plugin-side cascade was measured live (T5, 2026-08-14) breaking the engine's
# ownership model: quarantine + a re-placed orphan. Until the engine-side fix lands,
# the correct plugin behaviour is to cancel ONLY the requested intent's ids; the naked
# exit is then the ENGINE's known #19 gap, tracked on the card.

def __test_19_entry_cancel_leaves_the_exit_to_the_engine__():
    """Reverted contract: the exit leg survives at the venue (the engine owns it)."""
    venue = FakeDNSEVenue()
    b = _broker(venue)
    entry_id, exit_id = _place_entry_with_exit(b, venue)

    assert asyncio.run(b.execute_cancel(_cancel_env("L"))) is True
    assert venue.orders[entry_id]["orderStatus"] == "Canceled"
    assert venue.orders[exit_id]["orderStatus"] == "New", (
        "the plugin must NOT unilaterally cancel the engine-owned exit leg — doing so "
        "made the engine re-place it and quarantine (measured live)")
    assert [c for c in venue.calls if c[0] == "cancel_order" and c[1] == exit_id] == []
