"""Finite offline BrokerPlugin used only by broker-lab subprocess scenarios."""

import asyncio
import os
from dataclasses import dataclass
from datetime import timedelta

from pynecore.core.broker.models import (
    CancelDispositionOutcome,
    CapabilityLevel,
    ExchangeCapabilities,
)
from pynecore.core.plugin import ProviderError, TransientProviderError
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.plugin.live_provider import LiveProviderConfig
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV


@dataclass
class OfflineBrokerConfig(LiveProviderConfig):
    """Credential-free configuration for the offline lifecycle fixture."""


class OfflineBrokerPlugin(BrokerPlugin[OfflineBrokerConfig]):
    """Deterministic finite provider and inert broker with injectable lifecycle faults."""

    Config = OfflineBrokerConfig
    plugin_name = "offline-lab"
    reconnect_delay = 0.01
    max_reconnect_delay = 0.02
    feed_timeout_bars = 1

    def __init__(self, **kwargs):
        kwargs.setdefault("config", OfflineBrokerConfig())
        super().__init__(**kwargs)
        self._connected = False
        self._connect_attempts = 0
        self._live_index = 0
        self._account_id = "offline-lab-account"

    @classmethod
    def to_tradingview_timeframe(cls, timeframe):
        return timeframe

    @classmethod
    def to_exchange_timeframe(cls, timeframe):
        return timeframe

    def get_list_of_symbols(self, *args, **kwargs):
        return ["LABUSD"]

    def update_symbol_info(self):
        return SymInfo(
            prefix="LAB",
            description="Offline broker lab",
            ticker="LABUSD",
            currency="USD",
            basecurrency="LAB",
            period="1",
            type="forex",
            mintick=0.01,
            pricescale=100,
            minmove=1,
            pointvalue=1.0,
            mincontract=1.0,
            opening_hours=[],
            session_starts=[],
            session_ends=[],
            timezone="UTC",
        )

    def download_ohlcv(
        self, time_from, time_to, on_progress=None, limit=None, with_extra=False
    ):
        del limit, with_extra
        if os.environ.get("PYNE_LAB_DOWNLOAD_FAILURE") == "permanent":
            raise ProviderError("offline permanent download failure")
        start = max(time_from, time_to - timedelta(minutes=5))
        bars = [
            OHLCV(
                int((start + timedelta(minutes=index)).timestamp()),
                100,
                101,
                99,
                100,
                1,
            )
            for index in range(5)
        ]
        self.save_ohlcv_data(bars)
        if on_progress is not None:
            on_progress(time_to)

    async def connect(self):
        self._connect_attempts += 1
        permanent = os.environ.get("PYNE_LAB_CONNECT_FAILURE") == "permanent"
        transient_count = int(os.environ.get("PYNE_LAB_CONNECT_FAILURES", "0"))
        if permanent:
            raise ProviderError("offline permanent connect failure")
        if self._connect_attempts <= transient_count:
            raise TransientProviderError("offline transient connect failure")
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    async def watch_ohlcv(self, symbol, timeframe):
        del symbol, timeframe
        if self._live_index == 0:
            self._live_index += 1
            return OHLCV(2_000_000_000, 100, 101, 99, 100, 1, is_closed=True)
        raise asyncio.CancelledError()

    async def can_shutdown(self):
        return os.environ.get("PYNE_LAB_STUCK_SHUTDOWN") != "1"

    async def execute_entry(self, envelope):
        del envelope
        return []

    async def execute_exit(self, envelope):
        del envelope
        return []

    async def execute_close(self, envelope):
        raise AssertionError(f"offline fixture unexpectedly received close: {envelope}")

    async def execute_cancel(self, envelope):
        del envelope
        return True

    async def execute_cancel_with_outcome(self, envelope):
        del envelope
        return CancelDispositionOutcome.CANCEL_CONFIRMED

    async def get_open_orders(self, symbol=None):
        del symbol
        return []

    async def get_position(self, symbol):
        del symbol
        return None

    async def get_balance(self):
        credential = os.environ.get("PYNE_LAB_SECRET", "")
        if os.environ.get("PYNE_LAB_SECRET_FAILURE") == "1":
            if not credential:
                raise AssertionError("secret-redaction fixture requires a credential")
            raise RuntimeError("offline unexpected credential failure")
        if os.environ.get("PYNE_LAB_BALANCE_FAILURE") == "1":
            raise ProviderError("offline startup balance failure")
        return {"USD": 1_000_000.0}

    async def watch_orders(self):
        if False:
            yield None

    def get_capabilities(self):
        return ExchangeCapabilities(
            stop_order=CapabilityLevel.NATIVE,
            trailing_stop=CapabilityLevel.NATIVE,
            tp_sl_bracket=CapabilityLevel.NATIVE,
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
            oca_cancel=CapabilityLevel.SOFTWARE,
            amend_order=CapabilityLevel.SOFTWARE,
            cancel_all=CapabilityLevel.SOFTWARE,
            reduce_only=CapabilityLevel.SOFTWARE,
            watch_orders=CapabilityLevel.SOFTWARE,
            fetch_position=CapabilityLevel.NATIVE,
            idempotency=CapabilityLevel.SOFTWARE,
        )
