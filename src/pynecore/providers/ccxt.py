from typing import Callable
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    def override(func):
        return func
import re
from datetime import datetime, UTC, timedelta, time
from pathlib import Path
import tomllib

from pynecore.core.plugin import ProviderPlugin

from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from ..types.ohlcv import OHLCV

__all__ = ['CCXTProvider']

known_limits = {
    'binance': 1000,
    'bitget': {
        '1w': 12,
        '1d': 300,
        '4h': 1000,
        'default': 200
    },
    'bitmex': 500,
    'bybit': 200,
    'coinbase': 300,
    'kraken': 720,
    'kucoin': 1500,
    'okex': 200,
    'huobi': 2000,
}


def add_space_before_uppercase(s):
    return re.sub(r'(?<!^)([A-Z])', r' \1', s)


@dataclass
class CCXTConfig:
    """CCXT provider configuration"""

    apiKey: str = ""
    """Default API key for all exchanges"""

    secret: str = ""
    """Default API secret for all exchanges"""

    password: str = ""
    """Default API password (required by some exchanges like KuCoin)"""


class CCXTProvider(ProviderPlugin):
    """
    CCXT provider
    """

    plugin_name = "CCXT"
    Config = CCXTConfig

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert CCXT timeframe fmt to TradingView fmt.

        :param timeframe: Timeframe in CCXT fmt (e.g. "1m", "5m", "1h", "1d", "1w", "1M")
        :return: Timeframe in TradingView fmt (e.g. "1", "5", "60", "1D", "1W", "1M")
        :raises ValueError: If timeframe fmt is invalid
        """
        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe fmt: {timeframe}")

        unit = timeframe[-1]
        value = timeframe[:-1]

        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"Invalid timeframe value: {value}")

        if unit == 'm':
            return value
        elif unit == 'h':
            return str(int(value) * 60)
        elif unit == 'd':
            return f"{value}D"
        elif unit == 'w':
            return f"{value}W"
        elif unit == 'M':
            return f"{value}M"
        else:
            raise ValueError(f"Invalid timeframe fmt: {timeframe}")

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """
        Convert TradingView timeframe fmt to CCXT fmt.

        :param timeframe: Timeframe in TradingView fmt (e.g. "1", "5", "60", "1D", "1W", "1M")
        :return: Timeframe in CCXT fmt (e.g. "1m", "5m", "1h", "1d", "1w", "1M")
        :raises ValueError: If timeframe fmt is invalid
        """
        if timeframe.isdigit():
            mins = int(timeframe)
            if mins <= 0:
                raise ValueError(f"Invalid timeframe value: {timeframe}")
            if mins >= 60 and mins % 60 == 0:
                return f"{mins // 60}h"
            return f"{mins}m"

        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe fmt: {timeframe}")

        unit = timeframe[-1].upper()
        value = timeframe[:-1]

        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"Invalid timeframe value: {value}")

        if unit == 'D':
            return f"{value}d"
        elif unit == 'W':
            return f"{value}w"
        elif unit == 'M':
            return f"{value}M"
        else:
            raise ValueError(f"Invalid timeframe fmt: {timeframe}")

    @override
    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlv_dir: Path | None = None, config: object | None = None):
        """
        :param symbol: The symbol to get data for (e.g. "binance:BTC/USDT")
        :param timeframe: The timeframe to get data for in TradingView fmt
        :param ohlv_dir: The directory to save OHLCV data
        :param config: Pre-loaded CCXTConfig instance
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("CCXT is not installed. Please install it using `pip install ccxt`.")

        super().__init__(symbol=symbol, timeframe=timeframe, ohlv_dir=ohlv_dir, config=config)

        # Check symbol fmt
        try:
            if symbol is None:
                raise ValueError("Error: Symbol not provided!")
            xchg, symbol = symbol.split(':', 1)
        except (ValueError, AttributeError):
            xchg = symbol
            symbol = None

        if not xchg:
            raise ValueError("Error: Exchange name not provided! Use 'exchange:symbol' fmt! "
                             "(or simple exchange, if you want to list symbols)")

        self.symbol = symbol
        exchange_name = xchg.lower()

        # Build exchange config from the Config dataclass + optional exchange-specific TOML sections
        exchange_config = {}
        if self.config:
            # Base config from dataclass fields
            exchange_config = {
                k: v for k, v in vars(self.config).items() if v
            }

            # Check for exchange-specific override in the raw TOML
            if self.ohlcv_path:
                config_dir = self.ohlcv_path.parent.parent / 'config'
            else:
                config_dir = None

            if config_dir:
                toml_path = config_dir / 'plugins' / 'ccxt.toml'
                if toml_path.exists():
                    with open(toml_path, 'rb') as f:
                        raw_toml = tomllib.load(f)
                    if exchange_name in raw_toml and isinstance(raw_toml[exchange_name], dict):
                        exchange_config = raw_toml[exchange_name]

        # Create the CCXT client
        self._client: ccxt.Exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            **exchange_config
        })

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """
        Get list of symbols.
        """
        self._client.load_markets()
        return self._client.symbols or []

    @staticmethod
    def _create_24_7_sessions() -> tuple[
        list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]
    ]:
        """
        Create 24/7 opening hours and sessions for crypto markets.

        :return: Tuple of (opening_hours, session_starts, session_ends).
        """
        opening_hours = []
        session_starts = []
        session_ends = []
        for i in range(7):
            opening_hours.append(
                SymInfoInterval(day=i, start=time(hour=0, minute=0),
                                end=time(hour=23, minute=59, second=59)))
            session_starts.append(SymInfoSession(day=i, time=time(hour=0, minute=0)))
            session_ends.append(SymInfoSession(day=i, time=time(hour=23, minute=59, second=59)))
        return opening_hours, session_starts, session_ends

    @override
    def update_symbol_info(self) -> SymInfo:
        """
        Update symbol info from the exchange.
        """
        self._client.load_markets()
        assert self._client.markets
        market_details = self._client.markets[self.symbol]

        opening_hours, session_starts, session_ends = self._create_24_7_sessions()

        # Calculate minmove and pricescale from mintick
        mintick = market_details['precision']['price']
        minmove = mintick
        pricescale = 1
        while minmove < 1.0:
            pricescale *= 10
            minmove *= 10

        try:
            ticker = market_details['info']['symbol']
        except KeyError:
            try:
                ticker = market_details['symbol']
            except KeyError:
                ticker = market_details['id']

        assert self._client.id
        return SymInfo(
            prefix=self._client.id.upper(),
            description=f"{market_details['base']} / {market_details['quote']} "
                        f"{add_space_before_uppercase(market_details['info'].get('contractType', 'Spot'))}",
            ticker=ticker,
            currency=market_details['quote'],
            basecurrency=market_details['base'],
            period=self.timeframe,
            type="crypto",
            mintick=mintick,
            pricescale=pricescale,
            minmove=minmove,
            pointvalue=market_details.get('contractSize') or 1.0,
            timezone=self.timezone,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            taker_fee=market_details.get('taker'),
            maker_fee=market_details.get('maker'),
        )

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None):
        """
        Download OHLCV data.

        :param time_from: The start time.
        :param time_to: The end time.
        :param on_progress: Optional callback to call on progress.
        :param limit: Override the automatic chunk size.
        """
        tf: datetime = time_from.replace(tzinfo=None)
        tt: datetime = (time_to if time_to is not None else datetime.now(UTC)).replace(tzinfo=None)

        if limit is None:
            assert self._client.id
            limit_config = known_limits.get(self._client.id, 100)

            if isinstance(limit_config, dict):
                limit = limit_config.get(self.xchg_timeframe, limit_config.get('default', 100))
            else:
                limit = limit_config

        try:
            while tf < tt:
                if on_progress:
                    on_progress(tf)

                res: list = self._client.fetch_ohlcv(
                    symbol=self.symbol,
                    limit=limit,
                    timeframe=self.xchg_timeframe,
                    since=self._client.parse8601(tf.isoformat())
                )

                if not res:
                    tf += timedelta(days=1)

                for r in res:
                    t = int(r[0] / 1000)
                    dt = datetime.fromtimestamp(t, UTC).replace(tzinfo=None)
                    if dt > tt:
                        raise StopIteration

                    ohlcv = OHLCV(
                        timestamp=t,
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=float(r[4]),
                        volume=float(r[5]),
                    )

                    self.save_ohlcv_data(ohlcv)
                    tf = dt + timedelta(minutes=1)

        except StopIteration:
            pass

        if on_progress:
            on_progress(tt)
