"""
Replay provider — a deterministic, fixture-fed live provider for tests.

Feeds a security subprocess a fixed, pre-recorded sequence of warmup and live
OHLCV bars from a JSON fixture file, with no network and no sleeps. Drives a
real spawned security child through the streaming path deterministically (used
by the live ``request.security_lower_tf`` window end-to-end test).

Fixture format (UTF-8 JSON)::

    {
      "warmup": [[ts, open, high, low, close, volume], ...],
      "live":   [[ts, open, high, low, close, volume, is_closed], ...]
    }

``warmup`` rows are always closed bars (returned by :meth:`download_ohlcv`).
``live`` rows are streamed by :meth:`watch_ohlcv` in order; the optional 7th
element is the ``is_closed`` flag (default ``True``). Timestamps are Unix
seconds and must lie in the past relative to the run's wall clock so the
child's warmup-horizon drop keeps them.

Determinism contract: this is a *deterministic closed-window replay*, not a
deterministic forming-update replay. :meth:`watch_ohlcv` emits the fixture
bars eagerly (one per call, no pacing); the security child rations closed
intrabars per LTF period from its own buffer, so closed-window output is
deterministic. The developing/forming tail is not lockstep-paced across the
process boundary (the streamer keeps only the latest forming snapshot), so
bar-by-bar developing-tail evolution is covered by the in-process collector
unit test, not by a replay run.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, time
from typing import Callable

from pynecore.core.plugin import override
from pynecore.core.plugin.live_provider import LiveProviderConfig, LiveProviderPlugin
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.types.ohlcv import OHLCV


@dataclass
class ReplayConfig(LiveProviderConfig):
    """Config for :class:`ReplayProvider`.

    :ivar fixture_path: Absolute path to the JSON fixture file (warmup +
                        live bars). Inherits ``symbol_map`` from
                        :class:`LiveProviderConfig`.
    """

    fixture_path: str = ""


class ReplayProvider(LiveProviderPlugin[ReplayConfig]):
    """Fixture-fed live provider that replays pre-recorded bars deterministically."""

    Config = ReplayConfig

    feed_timeout_bars: int | None = None
    """No feed-staleness watchdog: post-fixture silence is the end of the
    recording, not a dead feed, so a reconnect must never be forced."""

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlcv_dir=None, config: ReplayConfig | None = None):
        super().__init__(symbol=symbol, timeframe=timeframe, ohlcv_dir=ohlcv_dir,
                         config=config)
        self._live_bars: list[OHLCV] = []
        self._live_idx: int = 0
        self._exhausted: asyncio.Event | None = None
        self._connected: bool = False

    # --- Timeframe converters (fixtures already use TradingView format) ---

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return timeframe

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        return timeframe

    # --- Symbol metadata ---

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        return [self.symbol] if self.symbol else []

    @override
    def update_symbol_info(self) -> SymInfo:
        """Minimal 24/7 crypto syminfo.

        The e2e test normally passes an explicit ``PluginSymbol.syminfo`` so
        the child never calls this; it is the abstract-method fallback.
        """
        opening_hours = [SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                         for i in range(7)]
        session_starts = [SymInfoSession(day=i, time=time(0, 0)) for i in range(7)]
        session_ends = [SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)]
        return SymInfo(
            prefix="REPLAY",
            description="Replay Symbol",
            ticker=self.symbol or "REPLAY",
            currency="USD",
            basecurrency="REPLAY",
            period=self.timeframe or "1",
            type="crypto",
            mintick=0.01,
            pricescale=100,
            minmove=1,
            pointvalue=1,
            mincontract=0.0001,
            timezone=self.timezone,
            volumetype="base",
            taker_fee=0.0,
            maker_fee=0.0,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
        )

    # --- Fixture loading ---

    def _load_fixture(self) -> dict:
        assert self.config is not None and self.config.fixture_path, \
            "ReplayProvider requires config.fixture_path"
        with open(self.config.fixture_path, encoding="utf-8") as fh:
            return json.load(fh)

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False):
        """Replay the fixture's warmup bars (all closed) into the in-memory
        capture used by the security subprocess. The whole recorded warmup is
        authoritative; ``time_from``/``time_to`` are not used to filter."""
        fixture = self._load_fixture()
        for row in fixture.get("warmup", []):
            ts, o, h, low, c, v = row[:6]
            self.save_ohlcv_data(OHLCV(
                timestamp=int(ts), open=float(o), high=float(h),
                low=float(low), close=float(c), volume=float(v),
            ))
        if on_progress:
            on_progress(time_to)

    # --- Live streaming ---

    @override
    async def connect(self) -> None:
        fixture = self._load_fixture()
        self._live_bars = [
            OHLCV(
                timestamp=int(r[0]), open=float(r[1]), high=float(r[2]),
                low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                is_closed=bool(r[6]) if len(r) > 6 else True,
            )
            for r in fixture.get("live", [])
        ]
        self._live_idx = 0
        self._exhausted = asyncio.Event()
        self._connected = True

    @override
    async def disconnect(self) -> None:
        self._connected = False

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if self._live_idx < len(self._live_bars):
            bar = self._live_bars[self._live_idx]
            self._live_idx += 1
            return bar
        # Fixture exhausted: park on a never-set event (event-driven, no sleep,
        # no busy-wait). The streamer's stop() closes the driving generator,
        # which cancels this await cleanly at shutdown.
        assert self._exhausted is not None
        await self._exhausted.wait()
        return self._live_bars[-1]
