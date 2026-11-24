"""
Catch-up bars in live mode: a closed bar the stream delivers late (an outage, a slow
reconnect) is replayed history. The script computes on it like on a warmup bar, but
it does not trade: a broker run leaves the venue untouched on it, and the first fresh
bar trades again. Paper trading on the history (``live_history_trades``) keeps
simulating on it, as it does on the warmup bars.

The clock is the feed's own: each live bar moves it to the moment the bar is
delivered, so a bar is fresh or catch-up exactly as it would be on the wall clock.
"""
import sys
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import cast

from pynecore.core.broker.models import (
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    EntryIntent,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.live_timing import is_catch_up, is_stale_update
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.script_runner import LIVE_TRANSITION, ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV

__test_helper_MINUTE = 60_000
# A minute boundary in late 2025, so the bars sit on a plausible live timeline
__test_helper_T0 = 29_383_333 * __test_helper_MINUTE


def __test_helper_syminfo(period: str = "1") -> SymInfo:
    """A crypto symbol, one-minute by default: catch-up starts 150 s after a bar's open."""
    return SymInfo(
        prefix="TEST", ticker="BTCUSDT", description="Test", currency="USDT",
        basecurrency="BTC", period=period, type="crypto", mintick=0.01, pricescale=100.0,
        pointvalue=1.0, mincontract=0.00001, timezone="UTC", volumetype="base",
        opening_hours=[], session_starts=[], session_ends=[],
    )


def __test_helper_bar(index: int, closed: bool = True) -> OHLCV:
    """The closed bar ``index`` minutes after :data:`__test_helper_T0`."""
    price = 50_000.0 + index * 10.0
    return OHLCV(timestamp=__test_helper_T0 + index * __test_helper_MINUTE, open=price,
                 high=price + 5.0, low=price - 5.0, close=price, volume=1.0,
                 is_closed=closed)


def __test_helper_open(index: int) -> int:
    return __test_helper_T0 + index * __test_helper_MINUTE


def __test_helper_on_time(index: int) -> int:
    """The moment bar ``index`` is delivered when the stream closes it on time."""
    return __test_helper_open(index + 1) + 1_000


def __test_helper_after_outage(index: int) -> int:
    """The moment a reconnect replays the missed bars: bar ``index`` closed 40 s ago.

    Bar ``index`` is fresh then (a close recovered inside the grace), the two bars
    before it are catch-up bars.
    """
    return __test_helper_open(index + 1) + 40_000


def __test_helper_feed(clock: list[float], history: int,
                       live: list['tuple[OHLCV, int] | Callable[[], None]']) -> Iterator:
    """Yield the history bars, the transition, then the live bars.

    Each live ``(bar, delivered_at)`` pair moves the clock to ``delivered_at``
    before the runner gets the bar; a callable runs once the runner has finished
    with the bar before it.
    """
    for i in range(history):
        yield __test_helper_bar(i)
    yield LIVE_TRANSITION
    for item in live:
        if callable(item):
            item()
            continue
        bar, delivered_at = item
        clock[0] = float(delivered_at)
        yield bar


def __test_helper_write_script(tmp_path: Path, name: str, code: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(code))
    sys.modules.pop(path.stem, None)
    return path


@dataclass
class MockBroker:
    """Duck-typed :class:`BrokerPlugin`: records every venue write."""
    client_order_id_max_len = 30
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    entry_calls: list[EntryIntent] = field(default_factory=list)
    exit_calls: list[ExitIntent] = field(default_factory=list)
    close_calls: list[CloseIntent] = field(default_factory=list)
    cancel_calls: list[CancelIntent] = field(default_factory=list)
    modify_calls: int = 0
    account_id: str = "default"
    position: ExchangePosition | None = None
    _next_id: int = 0

    def writes(self) -> tuple[int, int, int, int, int]:
        return (len(self.entry_calls), len(self.exit_calls), len(self.close_calls),
                len(self.cancel_calls), self.modify_calls)

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _order(self, envelope) -> ExchangeOrder:
        self._next_id += 1
        intent = envelope.intent
        return ExchangeOrder(
            id=f"xchg-{self._next_id}", symbol="BTCUSDT", side=intent.side,
            order_type=OrderType.MARKET, qty=intent.qty, filled_qty=0.0,
            remaining_qty=intent.qty, price=None, stop_price=None, average_fill_price=None,
            status=OrderStatus.OPEN, timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=envelope.client_order_id('e'),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope.intent)
        return [self._order(envelope)]

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope.intent)
        return [self._order(envelope)]

    async def execute_close(self, envelope):
        self.close_calls.append(envelope.intent)
        return self._order(envelope)

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope.intent)
        return True

    async def modify_entry(self, _old, new):
        self.modify_calls += 1
        return [self._order(new)]

    async def modify_exit(self, _old, new):
        self.modify_calls += 1
        return [self._order(new)]

    @staticmethod
    async def get_open_orders(_symbol=None):
        return []

    async def get_position(self, _symbol):
        return self.position

    @staticmethod
    async def get_balance():
        return {"USDT": 1_000_000.0}


def __test_helper_broker_runner(script_path: Path, plugin: MockBroker,
                                feed: Iterator, clock: list[float],
                                period: str = "1") -> ScriptRunner:
    return ScriptRunner(script_path=script_path, ohlcv_iter=feed,
                        syminfo=__test_helper_syminfo(period),
                        broker_plugin=cast(BrokerPlugin, plugin), live=True,
                        live_clock=lambda: clock[0])


def __test_catch_up_bars_leave_the_venue_untouched__(tmp_path):
    """A burst of catch-up bars sends nothing; the first fresh bar trades"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_venue", '''\
        """
        @pyne
        """
        from pynecore.lib import bar_index, barstate, close, plot, script, strategy

        @script.strategy("Catch-up venue")
        def main():
            if barstate.isrealtime:
                if bar_index == 2:
                    strategy.entry("R", strategy.long, qty=1.0, limit=40_000.0)
                else:
                    strategy.cancel("R")
                    strategy.entry("M", strategy.long, qty=1.0)
            plot(close, "close")
    ''')
    plugin = MockBroker()
    clock = [0.0]
    writes: dict[str, tuple] = {}
    runner: ScriptRunner

    def observe(name: str) -> Callable[[], None]:
        def record():
            engine = runner._order_sync_engine
            assert engine is not None
            writes[name] = (plugin.writes(), dict(engine._order_mapping))
        return record

    feed = __test_helper_feed(clock, 2, [
        # The live edge: the resting limit entry goes to the venue
        (__test_helper_bar(2), __test_helper_on_time(2)),
        observe("fresh"),
        # An outage: bars 3 and 4 come back with bar 5, two of them too late
        (__test_helper_bar(3), __test_helper_after_outage(5)),
        observe("catch-up 3"),
        (__test_helper_bar(4), __test_helper_after_outage(5)),
        observe("catch-up 4"),
        (__test_helper_bar(5), __test_helper_after_outage(5)),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)

    closes = [plot_data["close"] for _candle, plot_data, _trades in runner.run_iter()]

    # The script computed on every bar, the catch-up bars included
    assert closes == [__test_helper_bar(i).close for i in range(6)]

    fresh_writes, fresh_mapping = writes["fresh"]
    assert fresh_writes == (1, 0, 0, 0, 0)
    assert [intent.pine_id for intent in plugin.entry_calls[:1]] == ["R"]
    assert "R" in fresh_mapping

    # On the catch-up bars the script cancels R and enters M, yet nothing reaches the
    # venue, and the resting order stays mapped as it was
    assert writes["catch-up 3"] == (fresh_writes, fresh_mapping)
    assert writes["catch-up 4"] == (fresh_writes, fresh_mapping)

    # The close recovered inside the grace is fresh: its orders go out
    assert [intent.pine_id for intent in plugin.entry_calls] == ["R", "M"]
    assert [intent.pine_id for intent in plugin.cancel_calls] == ["R"]


def __test_fills_are_booked_on_catch_up_bars__(tmp_path):
    """A fill that lands during the burst is booked; the script's reaction waits"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_fills", '''\
        """
        @pyne
        """
        from pynecore.lib import bar_index, barstate, plot, script, strategy

        @script.strategy("Catch-up fills")
        def main():
            if barstate.isrealtime:
                if bar_index == 2:
                    strategy.entry("A", strategy.long, qty=1.0)
                elif strategy.position_size > 0:
                    strategy.close("A")
            plot(strategy.position_size, "pos")
    ''')
    plugin = MockBroker(
        capabilities=ExchangeCapabilities(reduce_only=CapabilityLevel.NATIVE))
    clock = [0.0]
    writes: dict[str, tuple] = {}
    runner: ScriptRunner

    def fill_entry():
        engine = runner._order_sync_engine
        assert engine is not None
        exchange_id = engine._order_mapping["A"][0]
        engine.on_order_event(OrderEvent(
            order=ExchangeOrder(
                id=exchange_id, symbol="BTCUSDT", side="buy", order_type=OrderType.MARKET,
                qty=1.0, filled_qty=1.0, remaining_qty=0.0, price=None, stop_price=None,
                average_fill_price=50_030.0, status=OrderStatus.FILLED, timestamp=0.0,
                fee=0.0, fee_currency="",
            ),
            event_type='filled', fill_price=50_030.0, fill_qty=1.0, timestamp=0.0,
            pine_id="A", leg_type=LegType.ENTRY,
        ))
        plugin.position = ExchangePosition(
            symbol="BTCUSDT", side="long", size=1.0, entry_price=50_030.0,
            unrealized_pnl=0.0, liquidation_price=None, leverage=1.0,
            margin_mode="isolated",
        )

    def observe(name: str) -> Callable[[], None]:
        return lambda: writes.__setitem__(name, plugin.writes())

    feed = __test_helper_feed(clock, 2, [
        (__test_helper_bar(2), __test_helper_on_time(2)),
        observe("fresh"),
        fill_entry,
        (__test_helper_bar(3), __test_helper_after_outage(5)),
        (__test_helper_bar(4), __test_helper_after_outage(5)),
        observe("catch-up"),
        (__test_helper_bar(5), __test_helper_after_outage(5)),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)

    sizes = [plot_data["pos"] for _candle, plot_data, _trades in runner.run_iter()]

    # The fill is booked on the first catch-up bar, so the script sees the position
    assert sizes == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    position = runner.script.position
    assert isinstance(position, BrokerPosition)
    assert position.open_trades[0].entry_price == 50_030.0

    # The close the script asks for on the catch-up bars is not sent
    assert writes["fresh"] == (1, 0, 0, 0, 0)
    assert writes["catch-up"] == (1, 0, 0, 0, 0)
    # The first fresh bar sends it
    assert [intent.pine_id for intent in plugin.close_calls] == ["A"]


def __test_a_forming_bar_ends_the_catch_up__(tmp_path):
    """The forming bar is the live edge: its updates trade even after a catch-up burst"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_forming", '''\
        """
        @pyne
        """
        from pynecore.lib import barstate, script, strategy

        @script.strategy("Catch-up forming", calc_on_every_tick=True)
        def main():
            if barstate.isrealtime:
                strategy.entry("L", strategy.long, qty=1.0, limit=40_000.0)
    ''')
    plugin = MockBroker()
    clock = [0.0]
    entries: dict[str, int] = {}

    def observe(name: str) -> Callable[[], None]:
        return lambda: entries.__setitem__(name, len(plugin.entry_calls))

    feed = __test_helper_feed(clock, 2, [
        (__test_helper_bar(2), __test_helper_after_outage(4) + __test_helper_MINUTE),
        (__test_helper_bar(3), __test_helper_after_outage(4) + __test_helper_MINUTE),
        observe("catch-up"),
        # The stream goes on with the bar forming now
        (__test_helper_bar(5, closed=False), __test_helper_open(5) + 10_000),
        observe("first update"),
        (__test_helper_bar(5, closed=False), __test_helper_open(5) + 20_000),
        observe("second update"),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)
    runner.script.calc_on_every_tick = True

    list(runner.run_iter())

    assert entries["catch-up"] == 0
    # The first update places the order, the next one syncs it to the venue
    assert entries["first update"] == 0
    assert entries["second update"] == 1


def __test_stale_forming_updates_do_not_trade__(tmp_path):
    """Updates of a slot past its close deadline (a lagging consumer's backlog) send nothing"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_stale_forming", '''\
        """
        @pyne
        """
        from pynecore.lib import barstate, script, strategy

        @script.strategy("Catch-up stale forming", calc_on_every_tick=True)
        def main():
            if barstate.isrealtime:
                strategy.entry("L", strategy.long, qty=1.0, limit=40_000.0)
    ''')
    plugin = MockBroker()
    clock = [0.0]
    entries: dict[str, int] = {}

    def observe(name: str) -> Callable[[], None]:
        return lambda: entries.__setitem__(name, len(plugin.entry_calls))

    feed = __test_helper_feed(clock, 2, [
        (__test_helper_bar(2), __test_helper_after_outage(4) + __test_helper_MINUTE),
        (__test_helper_bar(3), __test_helper_after_outage(4) + __test_helper_MINUTE),
        # Queued updates of bar 4, processed 100 s and 110 s after its open: past
        # its close deadline (one minute plus the 30 s grace)
        (__test_helper_bar(4, closed=False), __test_helper_open(4) + 100_000),
        (__test_helper_bar(4, closed=False), __test_helper_open(4) + 110_000),
        observe("stale updates"),
        # The bar forming now
        (__test_helper_bar(5, closed=False), __test_helper_open(5) + 10_000),
        observe("first fresh update"),
        (__test_helper_bar(5, closed=False), __test_helper_open(5) + 20_000),
        observe("second fresh update"),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)
    runner.script.calc_on_every_tick = True

    list(runner.run_iter())

    assert entries["stale updates"] == 0
    assert entries["first fresh update"] == 0
    assert entries["second fresh update"] == 1


def __test_a_stale_update_closes_the_latch_again__(tmp_path):
    """A live run whose consumer falls behind stops trading on the stale updates"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_stale_relatch", '''\
        """
        @pyne
        """
        from pynecore.lib import barstate, script, strategy

        @script.strategy("Catch-up stale relatch", calc_on_every_tick=True)
        def main():
            if barstate.isrealtime:
                strategy.entry("L", strategy.long, qty=1.0, limit=40_000.0)
    ''')
    plugin = MockBroker()
    clock = [0.0]
    entries: dict[str, int] = {}

    def observe(name: str) -> Callable[[], None]:
        return lambda: entries.__setitem__(name, len(plugin.entry_calls))

    feed = __test_helper_feed(clock, 2, [
        # A fresh forming bar places the order, which is not synced before the
        # consumer falls behind
        (__test_helper_bar(2, closed=False), __test_helper_open(2) + 10_000),
        observe("fresh"),
        # The next update of it is processed long after its slot closed
        (__test_helper_bar(2, closed=False), __test_helper_open(2) + 100_000),
        observe("stale"),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)
    runner.script.calc_on_every_tick = True

    list(runner.run_iter())

    assert entries["fresh"] == 0
    assert entries["stale"] == 0


def __test_helper_month_open(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)


def __test_helper_month_bar(open_ms: int, closed: bool = True) -> OHLCV:
    return OHLCV(timestamp=open_ms, open=50_000.0, high=50_005.0, low=49_995.0,
                 close=50_000.0, volume=1.0, is_closed=closed)


def __test_monthly_updates_trade_until_the_calendar_close__(tmp_path):
    """The last day of a 31-day month is still the live edge of a monthly bar"""
    script_path = __test_helper_write_script(tmp_path, "catch_up_monthly", '''\
        """
        @pyne
        """
        from pynecore.lib import barstate, script, strategy

        @script.strategy("Catch-up monthly", calc_on_every_tick=True)
        def main():
            if barstate.isrealtime:
                strategy.entry("L", strategy.long, qty=1.0, limit=40_000.0)
    ''')
    plugin = MockBroker()
    clock = [0.0]
    entries: dict[str, int] = {}

    def observe(name: str) -> Callable[[], None]:
        return lambda: entries.__setitem__(name, len(plugin.entry_calls))

    october = __test_helper_month_open(2026, 10)
    october_31 = int(datetime(2026, 10, 31, 12, tzinfo=UTC).timestamp() * 1000)

    def feed() -> Iterator:
        yield __test_helper_month_bar(__test_helper_month_open(2026, 8))
        yield __test_helper_month_bar(__test_helper_month_open(2026, 9))
        yield LIVE_TRANSITION
        # October's forming bar on its last day: past the average month (30.4 days)
        # after its open, yet half a day before its close on November 1
        clock[0] = float(october_31)
        yield __test_helper_month_bar(october, closed=False)
        observe("first update")()
        clock[0] = float(october_31 + 3_600_000)
        yield __test_helper_month_bar(october, closed=False)
        observe("second update")()

    runner = __test_helper_broker_runner(script_path, plugin, feed(), clock, period="1M")
    runner.script.calc_on_every_tick = True

    list(runner.run_iter())

    # The first update places the order, the next one syncs it to the venue
    assert entries["first update"] == 0
    assert entries["second update"] == 1


def __test_monthly_catch_up_follows_the_calendar__():
    """Freshness of a monthly bar is measured from its calendar close, not the average month"""
    october = __test_helper_month_open(2026, 10)
    november = __test_helper_month_open(2026, 11)
    december = __test_helper_month_open(2026, 12)
    grace_ms = 30_000

    assert not is_stale_update(october, "1M", november - 1)
    assert not is_stale_update(october, "1M", november + grace_ms)
    assert is_stale_update(october, "1M", november + grace_ms + 1)

    # A closed October bar is fresh until November could still close on time
    assert not is_catch_up(october, "1M", december + grace_ms)
    assert is_catch_up(october, "1M", december + grace_ms + 1)


_PAPER_SCRIPT = '''\
    """
    @pyne
    """
    from pynecore.lib import barstate, plot, script, strategy

    @script.strategy("Catch-up paper", initial_capital=1_000_000,
                     default_qty_type=strategy.fixed, default_qty_value=1)
    def main():
        if barstate.isrealtime and strategy.position_size == 0:
            strategy.entry("L", strategy.long)
        plot(strategy.position_size, "pos")
'''


def __test_helper_paper_sizes(tmp_path: Path, name: str, history_trades: bool) -> list[float]:
    """Position sizes of a paper run whose first two live bars are catch-up bars."""
    script_path = __test_helper_write_script(tmp_path, name, _PAPER_SCRIPT)
    clock = [0.0]
    feed = __test_helper_feed(clock, 2, [
        (__test_helper_bar(2), __test_helper_after_outage(4)),
        (__test_helper_bar(3), __test_helper_after_outage(4)),
        (__test_helper_bar(4), __test_helper_after_outage(4)),
        (__test_helper_bar(5), __test_helper_on_time(5)),
    ])
    runner = ScriptRunner(script_path, feed, __test_helper_syminfo(), live=True,
                          live_history_trades=history_trades, live_clock=lambda: clock[0])
    return [plot_data["pos"] for _candle, plot_data, _trades in runner.run_iter()]


def __test_paper_history_trades_keep_trading_on_catch_up_bars__(tmp_path):
    """With ``live_history_trades`` the simulator trades on catch-up bars as on the warmup"""
    # Entered at the first catch-up bar's close, filled at the next bar's open
    assert __test_helper_paper_sizes(tmp_path, "paper_history", True) == [0, 0, 0, 1, 1, 1]


def __test_paper_without_history_trades_waits_for_the_live_edge__(tmp_path):
    """A paper run that starts flat at the transition skips the catch-up bars too"""
    # Entered at the first fresh bar's close (bar 4), filled at bar 5's open
    assert __test_helper_paper_sizes(tmp_path, "paper_flat", False) == [0, 0, 0, 0, 0, 1]


def __test_warmup_bars_are_shown_to_the_stop_limit_watches__(tmp_path):
    """Every history bar a live start replays reaches the engine with its range.

    A software stop-limit order armed before a restart judges the bars it did
    not see live by their high / low; the timestamps are the bars' open times
    in milliseconds, the unit of the engine's bar clock.
    """
    script_path = __test_helper_write_script(tmp_path, "warmup_watch", '''\
        """
        @pyne
        """
        from pynecore.lib import close, plot, script

        @script.strategy("Warmup watch")
        def main():
            plot(close, "close")
    ''')
    plugin = MockBroker()
    clock = [0.0]
    feed = __test_helper_feed(clock, 3, [
        (__test_helper_bar(3), __test_helper_on_time(3)),
    ])
    runner = __test_helper_broker_runner(script_path, plugin, feed, clock)
    engine = runner._order_sync_engine
    assert engine is not None
    seen: list[tuple[int, float, float]] = []
    engine.observe_history_bar = (  # type: ignore[method-assign]
        lambda ts, high, low: seen.append((ts, high, low)))

    list(runner.run_iter())

    assert seen == [(bar.timestamp, bar.high, bar.low)
                    for bar in (__test_helper_bar(i) for i in range(3))]
