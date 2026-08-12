"""``_stop_fill_price`` — the LO price a triggered stop emits.

DNSE has no stop-market order: a conditional order emits an ``LO`` at ``price`` when
``stopPrice`` is crossed. Pricing that LO *at* the trigger makes the stop a
stop-LIMIT that never fills on a gap — triggered, unfilled, still exposed. These
tests pin the fix: the LO is priced **through** the trigger by ``2 x slippage`` ticks,
clamped into the venue band.
"""
import pytest

from pynecore import lib
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig

MINTICK = 0.1
CEILING, FLOOR = 2058.6, 1789.4


@pytest.fixture
def broker(monkeypatch):
    cfg = DNSEBrokerConfig(api_key="k", api_secret="s", account_no="0001672126")
    b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    monkeypatch.setattr(b, "_band", lambda: (CEILING, FLOOR))
    monkeypatch.setattr(b, "_mintick", lambda: MINTICK)
    return b


def _set_slippage(monkeypatch, ticks):
    """Point ``lib._script`` at a stub declaring ``slippage`` (Pine: in ticks)."""
    monkeypatch.setattr(lib, "_script", type("S", (), {"slippage": ticks})(),
                        raising=False)


def __test_buy_stop_prices_above_the_trigger__(broker, monkeypatch):
    _set_slippage(monkeypatch, 3)
    # 2 x 3 ticks x 0.1 = 0.6 THROUGH the trigger, so the LO can lift the offer
    assert broker._stop_fill_price("buy", 1900.0) == pytest.approx(1900.6)


def __test_sell_stop_prices_below_the_trigger__(broker, monkeypatch):
    _set_slippage(monkeypatch, 3)
    assert broker._stop_fill_price("sell", 1900.0) == pytest.approx(1899.4)


def __test_doubles_the_declared_slippage__(broker, monkeypatch):
    _set_slippage(monkeypatch, 5)
    # 2 x 5 x 0.1 = 1.0 — doubling leaves room for the book to move between the
    # trigger printing and the order arriving.
    assert broker._stop_fill_price("buy", 1900.0) == pytest.approx(1901.0)


def __test_zero_slippage_falls_back_to_config_not_the_trigger__(broker, monkeypatch):
    """Pine defaults slippage to 0; using it verbatim would recreate the bug."""
    _set_slippage(monkeypatch, 0)
    price = broker._stop_fill_price("buy", 1900.0)
    assert price > 1900.0, "a 0-slippage script must NOT post the LO at the trigger"
    # default stop_slippage_ticks = 10 -> 10 x 0.1 = 1.0
    assert price == pytest.approx(1901.0)


def __test_missing_script_still_offsets__(broker, monkeypatch):
    """No running script (probe/tooling) must not collapse to trigger price."""
    monkeypatch.setattr(lib, "_script", None, raising=False)
    assert broker._stop_fill_price("sell", 1900.0) == pytest.approx(1899.0)


def __test_clamped_to_the_band__(broker, monkeypatch):
    """A trigger near the band edge must not produce a rejectable price."""
    _set_slippage(monkeypatch, 500)                  # absurd offset: 2*500*0.1 = 100
    assert broker._stop_fill_price("buy", CEILING - 1) == pytest.approx(CEILING)
    assert broker._stop_fill_price("sell", FLOOR + 1) == pytest.approx(FLOOR)


def __test_explicit_stop_limit_is_untouched__(broker, monkeypatch):
    """``entry(stop, limit)`` is the user asking for a stop-LIMIT — honour it.

    Guards the execute_entry branch: only a *bare* stop gets the through-trigger
    price; an explicit limit must reach the venue verbatim.
    """
    import asyncio
    from types import SimpleNamespace

    _set_slippage(monkeypatch, 3)
    sent = {}
    monkeypatch.setattr(broker, "_place",
                        lambda env, side, qty, **kw: sent.update(kw) or [])
    intent = SimpleNamespace(side="buy", qty=1.0, stop=1900.0, limit=1905.0,
                             intent_key="k", pine_id="L", from_entry=None)
    asyncio.run(broker.execute_entry(SimpleNamespace(intent=intent)))

    assert sent["price"] == 1905.0, "an explicit limit must not be overridden"
    assert sent["stop_price"] == 1900.0
    assert sent["category"] == "STOP"


def __test_bare_stop_entry_uses_through_price__(broker, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    _set_slippage(monkeypatch, 3)
    sent = {}
    monkeypatch.setattr(broker, "_place",
                        lambda env, side, qty, **kw: sent.update(kw) or [])
    intent = SimpleNamespace(side="buy", qty=1.0, stop=1900.0, limit=None,
                             intent_key="k", pine_id="L", from_entry=None)
    asyncio.run(broker.execute_entry(SimpleNamespace(intent=intent)))

    assert sent["price"] == pytest.approx(1900.6), "bare stop = stop-MARKET intent"
    assert sent["stop_price"] == 1900.0


def __test_stop_loss_exit_fills_through_the_trigger__(broker, monkeypatch):
    """The dangerous one: an SL priced AT the trigger never fills on a gap."""
    import asyncio
    from types import SimpleNamespace

    _set_slippage(monkeypatch, 3)
    sent = {}
    monkeypatch.setattr(broker, "_place",
                        lambda env, side, qty, **kw: sent.update(kw) or [])
    intent = SimpleNamespace(side="sell", qty=1.0, tp_price=None, sl_price=1900.0,
                             intent_key="k", pine_id="L", from_entry=None)
    asyncio.run(broker.execute_exit(SimpleNamespace(intent=intent)))

    assert sent["stop_price"] == 1900.0, "trigger stays exactly where Pine asked"
    assert sent["price"] == pytest.approx(1899.4), "but the LO must be able to fill"


def __test_oco_tp_keeps_its_limit_sl_leg_fills_through__(broker, monkeypatch):
    """A take-profit is a limit by nature and must NOT slip; only the SL leg does."""
    import asyncio
    from types import SimpleNamespace

    _set_slippage(monkeypatch, 3)
    sent = {}
    monkeypatch.setattr(broker, "_place",
                        lambda env, side, qty, **kw: sent.update(kw) or [])
    intent = SimpleNamespace(side="sell", qty=1.0, tp_price=1950.0, sl_price=1900.0,
                             intent_key="k", pine_id="L", from_entry=None)
    asyncio.run(broker.execute_exit(SimpleNamespace(intent=intent)))

    assert sent["price"] == 1950.0, "TP limit must be exact"
    assert sent["stop_price"] == 1900.0
    assert sent["stop_order_price"] == pytest.approx(1899.4)
