from typing import Callable
import sys

# Python 3.12+
if sys.version_info >= (3, 12):
    from typing import override
else:
    # Python 3.11
    def override(func):
        return func
import re
from datetime import datetime, UTC, timedelta
from pathlib import Path
from datetime import time
import tomllib

from .provider import Provider

from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from ..types.ohlcv import OHLCV

__all__ = ['CCXTProvider']

known_limits = {
    'binance': 1000,
    'bitmex': 500,
    'bybit': 200,
    'coinbase': 300,
    'kraken': 720,
    'kucoin': 1500,
    'okex': 200,
    'huobi': 2000,
}


def add_space_before_uppercase(s):
    # Use regex to add a space before each uppercase letter
    return re.sub(r'(?<!^)([A-Z])', r' \1', s)


class CCXTProvider(Provider):
    """
    CCXT provider
    """

    config_keys = {
        '# Default settings for all exchanges if not specified': '',
        'apiKey': '',
        'secret': '',
        'password': '',
        '# ...anything else your exchange needs': '',
        '# Exchange specific configuration examples:': '',
        '# To set exchange specific configurations use sections like:': '',
        '# [ccxt.binance]': '',
        '# apiKey = "your_binance_api_key"': '',
        '# secret = "your_binance_secret"': '',
        '# ': '',  # noqa
        '# [ccxt.kucoin]': '',
        '# apiKey = "your_kucoin_api_key"': '',
        '# secret = "your_kucoin_secret"': '',
        '# password = "your_kucoin_password"': '',
        '# ': '',  # noqa
        '# [ccxt.okex]': '',
        '# apiKey = "your_okex_api_key"': '',
        '# secret = "your_okex_secret"': '',
        '# password = "your_okex_password"': '',
        '# isTestnet = true    # Add any custom parameter required by the exchange': ''
    }

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert CCXT timeframe fmt to TradingView fmt.

        :param timeframe: Timeframe in CCXT fmt (e.g. "1m", "5m", "1h", "1d", "1w", "1M")
        :type timeframe: str
        :return: Timeframe in TradingView fmt (e.g. "1", "5", "60", "1D", "1W", "1M")
        :rtype: str

        :Examples:

        >>> Provider.to_tradingview_timeframe("1m")  # "1"
        >>> Provider.to_tradingview_timeframe("5m")  # "5"
        >>> Provider.to_tradingview_timeframe("1h")  # "60"
        >>> Provider.to_tradingview_timeframe("1d")  # "1D"

        :raises ValueError: If timeframe fmt is invalid
        """
        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe fmt: {timeframe}")

        unit = timeframe[-1]
        value = timeframe[:-1]

        # Verify that value is a valid number
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
        :type timeframe: str
        :return: Timeframe in CCXT fmt (e.g. "1m", "5m", "1h", "1d", "1w", "1M")
        :rtype: str

        :Examples:

        >>> Provider.to_exchange_timeframe("1")   # "1m"
        >>> Provider.to_exchange_timeframe("5")   # "5m"
        >>> Provider.to_exchange_timeframe("60")  # "1h"
        >>> Provider.to_exchange_timeframe("1D")  # "1d"

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

        # Verify that value is a valid number
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
                 ohlv_dir: Path | None = None, config_dir: Path | None = None):
        """
        :param symbol: The symbol to get data for
        :param timeframe: The timeframe to get data for in TradingView fmt
        :param ohlv_dir: The directory to save OHLV data
        :param config_dir: The directory to read the config file from
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("CCXT is not installed. Please install it using `pip install ccxt`.")

        super().__init__(symbol=symbol, timeframe=timeframe, ohlv_dir=ohlv_dir, config_dir=config_dir)

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
                             "(or symple exchange, if you want to list symbols)")

        self.symbol = symbol
        exchange_name = xchg.lower()

        # Check if there's an exchange-specific configuration
        exchange_config = {}

        # Load configuration from providers.toml
        with open(self.config_dir / 'providers.toml', 'rb') as f:
            data = tomllib.load(f)

            # Look for exchange-specific config
            exchange_section = f'ccxt.{exchange_name}'
            if exchange_section in data:
                exchange_config = data[exchange_section]
            else:
                # Use the default ccxt config
                exchange_config = self.config

        # Create the CCXT client
        self._client: ccxt.Exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            **exchange_config
        })

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """
        Get list of symbols
        """
        self._client.load_markets()
        return self._client.symbols or []

    @classmethod
    def get_opening_hours_and_sessions(cls) \
            -> tuple[list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """
        Process opening hours information
        """
        opening_hours = []
        session_starts = []
        session_ends = []
        for i in range(7):
            opening_hours.append(
                SymInfoInterval(day=i, start=time(hour=0, minute=0), end=time(hour=23, minute=59, second=59)))
            session_starts.append(SymInfoSession(day=i, time=time(hour=0, minute=0)))
            session_ends.append(SymInfoSession(day=i, time=time(hour=23, minute=59, second=59)))

        return opening_hours, session_starts, session_ends

    @override
    def update_symbol_info(self) -> SymInfo:
        """
        Update symbol info from the exchange
        """
        self._client.load_markets()
        assert self._client.markets
        market_details = self._client.markets[self.symbol]

        # Get opening hours and sessions
        opening_hours, session_starts, session_ends = self.get_opening_hours_and_sessions()

        # Calculate minmove and pricescale from mintick  # syminfo.minmove / syminfo.pricescale = syminfo.mintick
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
            type="crypto",  # it could be better, but TV just call everything "crypto"
            mintick=mintick,
            pricescale=pricescale,
            minmove=minmove,
            pointvalue=market_details.get('contractSize') or 1.0,
            timezone=self.timezone,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            # This is not found on TV, but it could be useful
            taker_fee=market_details.get('taker'),
            maker_fee=market_details.get('maker'),
        )

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None):
        """
        Download OHLV data

        :param time_from: The start time
        :param time_to:  The end time
        :param on_progress: Optional callback to call on progress
        """
        # Shortcuts for the time_from and time_to
        tf: datetime = time_from.replace(tzinfo=None)
        tt: datetime = (time_to if time_to is not None else datetime.now(UTC)).replace(tzinfo=None)

        # Get the limit by exchange or use safe default
        assert self._client.id
        limit = known_limits.get(self._client.id, 100)

        try:
            # Loop through the time range
            while tf < tt:
                if on_progress:
                    on_progress(tf)

                # Fetch a part of data
                res: list = self._client.fetch_ohlcv(
                    symbol=self.symbol,
                    limit=limit,
                    timeframe=self.xchg_timeframe,
                    since=self._client.parse8601(tf.isoformat())
                )

                # If no data, skip to the next day, maybe the symbol was not yet traded that day
                if not res:
                    tf += timedelta(days=1)

                # Process the data
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
                    tf = dt + timedelta(minutes=1)  # Move to next time step + 1 minute

        except StopIteration:
            pass

        if on_progress:
            on_progress(tt)
