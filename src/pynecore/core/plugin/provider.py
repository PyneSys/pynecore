from typing import Callable, NamedTuple
from abc import abstractmethod, ABCMeta
from pathlib import Path
from datetime import datetime

from pynecore.types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader

from . import Plugin, ConfigT


class Broker(NamedTuple):
    """A selectable broker / exchange of a :attr:`~ProviderPlugin.multi_broker` provider.

    :ivar id: The canonical selector used in the provider string and the saved
        filename (e.g. ``"pepperstoneuk"``, ``"binance"``). Must be space-free.
    :ivar name: A human-readable display name (e.g. ``"Pepperstone - Europe"``,
        ``"Binance"``), or ``""`` when the provider exposes none.
    """
    id: str
    name: str = ""


class ProviderPlugin(Plugin[ConfigT], metaclass=ABCMeta):
    """
    Base class for all data providers.

    Subclasses must implement the abstract methods and define a ``Config``
    dataclass for configuration (used by :func:`pynecore.core.config.ensure_config`).
    """

    timezone: str = 'UTC'
    """Default timezone of the provider."""

    symbol: str | None = None
    """Symbol of the provider."""

    timeframe: str | None = None
    """Timeframe of the provider."""

    xchg_timeframe: str | None = None
    """Exchange-specific timeframe format."""

    ohlcv_path: Path | None = None
    """Path to the OHLCV data file."""

    fetch_all_by_default: bool = False
    """If True, fetch all available data when no start date is given (instead of 1 year)."""

    multi_broker: bool = False
    """If True, this provider serves many brokers/exchanges and the first segment
    of the provider string after the provider name selects the broker
    (e.g. ``ccxt:BYBIT:BTC/USDT:USDT`` → broker ``BYBIT``). Single-broker
    providers leave this ``False`` and treat the whole string as the symbol."""

    @classmethod
    @abstractmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to TradingView format.

        :param timeframe: Timeframe in exchange format.
        :return: Timeframe in TradingView format.
        """

    @classmethod
    @abstractmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to exchange format.

        :param timeframe: Timeframe in TradingView format.
        :return: Timeframe in exchange format.
        """

    @classmethod
    def get_ohlcv_path(cls, symbol: str, timeframe: str, ohlcv_dir: Path,
                       provider_name: str | None = None) -> Path:
        """
        Get the output path of the OHLCV data file.

        :param symbol: Symbol name.
        :param timeframe: Timeframe in TradingView format.
        :param ohlcv_dir: Directory to save OHLCV data.
        :param provider_name: Override provider name in filename.
        :return: Path to the OHLCV file.
        """
        return ohlcv_dir / (f"{provider_name or cls.__name__.lower().replace('provider', '').replace('plugin', '')}"
                            f"_{symbol.replace('/', '_').replace(':', '_').upper()}"
                            f"_{timeframe}.ohlcv")

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlcv_dir: Path | None = None, config: ConfigT | None = None):
        """
        :param symbol: The symbol to get data for.
        :param timeframe: The timeframe to get data for in TradingView format.
        :param ohlcv_dir: The directory to save OHLCV data.
        :param config: Pre-loaded config dataclass instance.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.xchg_timeframe = self.to_exchange_timeframe(timeframe) if timeframe else None
        if ohlcv_dir:
            assert symbol and timeframe
            self.ohlcv_path = self.get_ohlcv_path(symbol, timeframe, ohlcv_dir)
        else:
            self.ohlcv_path = None
        self.ohlcv_file = OHLCVWriter(self.ohlcv_path) if self.ohlcv_path else None
        self.config: ConfigT | None = config

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a provider-format symbol to the exchange API format.

        Called by the framework before passing ``symbol`` to :meth:`watch_ohlcv`
        in the live runner. For historical methods (:meth:`download_ohlcv`,
        :meth:`update_symbol_info`), providers use ``self.symbol`` directly —
        handle any needed format conversion in ``__init__`` instead.

        Override when the user-configured symbol includes prefixes or formatting
        that the exchange API cannot accept
        (e.g. stripping ``"binance:"`` from ``"binance:BTC/USDT"``).

        :param symbol: Symbol as configured by the user.
        :return: Symbol in the format the exchange API expects.
        """
        return symbol

    def resolve_symbol(self, pine_key: str) -> str:
        """
        Translate a Pine-style symbol key to the plugin-native form.

        Live ``request.security()`` calls hand the framework a TradingView-style
        symbol (e.g. ``"FX:EURUSD"``). This method consults
        ``config.symbol_map`` first (the per-plugin TOML translation table);
        if the key is not mapped the default fallback is the identity, i.e.
        the Pine key is forwarded unchanged on the assumption that the user
        already wrote a plugin-native symbol.

        ``normalize_symbol`` is deliberately **not** used as the fallback:
        provider instances bind ``normalize_symbol`` to the chart's own
        symbol (e.g. CCXT's returns ``self.symbol`` regardless of the
        argument), so consulting it for a cross-symbol key would silently
        resolve to the chart symbol and download wrong data.

        Plugins that need real cross-symbol translation should override
        :meth:`resolve_symbol` directly.

        :param pine_key: Symbol as written in the Pine script.
        :return: Symbol in the format the plugin's exchange API expects.
        """
        sm = getattr(self.config, 'symbol_map', None)
        if sm and pine_key in sm:
            return sm[pine_key]
        return pine_key

    @classmethod
    def construct_pair_symbol(cls, from_cur: str, to_cur: str) -> str:
        """
        Build a Pine-style symbol for a currency pair.

        Used by the auto-spawn rate-source path when a Pine script needs a
        ``(from_cur, to_cur)`` rate that is not already exposed by the chart
        or by an explicit ``request.security()`` context. The default
        concatenation (``"EUR" + "USD" -> "EURUSD"``) matches the most common
        FX symbol convention; plugins whose API expects a different shape
        (e.g. ``"EUR-USD"`` or ``"EUR/USD"``) can override.

        The returned key is fed through :meth:`resolve_symbol`, so users can
        still keep TradingView prefixes (``"FX:EURUSD"``) in their
        ``symbol_map`` instead of relying on the raw concatenation.
        """
        return f"{from_cur}{to_cur}"

    def __enter__(self) -> OHLCVWriter:
        assert self.ohlcv_file is not None
        return self.ohlcv_file.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.ohlcv_file is not None
        self.ohlcv_file.close()

    @classmethod
    def get_list_of_brokers(cls) -> list[Broker]:
        """
        Get the list of brokers/exchanges this provider can serve.

        Only meaningful for :attr:`multi_broker` providers. Optional — the
        default raises :class:`NotImplementedError`, which the ``pyne data``
        CLI catches and reports gracefully. Implemented as a classmethod so it
        can answer ``--list-brokers`` without a symbol-bound instance.

        :return: List of :class:`Broker` records (``id`` selector + optional
            human-readable ``name``).
        :raises NotImplementedError: If the provider does not enumerate brokers.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support listing brokers"
        )

    @abstractmethod
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """
        Get list of available symbols.

        :return: List of symbol names.
        """

    @abstractmethod
    def update_symbol_info(self) -> SymInfo:
        """
        Fetch and return symbol info from the exchange.

        This should include opening hours and session data.

        :return: Symbol information.
        """

    def is_symbol_info_exists(self) -> bool:
        """
        Check if the symbol info TOML file exists.

        :return: True if the file exists.
        """
        assert self.ohlcv_path is not None
        return self.ohlcv_path.with_suffix('.toml').exists()

    def get_symbol_info(self, force_update=False) -> SymInfo:
        """
        Get symbol info, loading from cache or fetching from exchange.

        :param force_update: Force update from exchange even if cached.
        :return: Symbol information.
        """
        assert self.ohlcv_path is not None
        toml_path = self.ohlcv_path.with_suffix('.toml')
        if self.is_symbol_info_exists() and not force_update:
            return SymInfo.load_toml(toml_path)

        sym_info = self.update_symbol_info()
        sym_info.save_toml(toml_path)
        return sym_info

    def save_ohlcv_data(self, data: OHLCV | list[OHLCV]):
        """
        Save OHLCV data to the file.

        :param data: Single OHLCV record or list of records.
        """
        assert self.ohlcv_file is not None
        if isinstance(data, OHLCV):
            self.ohlcv_file.write(data)
        else:
            for candle in data:
                self.ohlcv_file.write(candle)

    @abstractmethod
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False):
        """
        Download OHLCV data from the exchange.

        Use :meth:`save_ohlcv_data` to write records to the data file.

        :param time_from: The start time. Use ``datetime.fromtimestamp(0)`` to fetch all available data.
        :param time_to: The end time.
        :param on_progress: Optional progress callback.
        :param limit: Override the automatic chunk size (number of bars per API request).
        :param with_extra: When ``True``, also fetch and persist the provider's
            extra per-bar fields (e.g. ask/bid/spread) to the ``.extra.csv``
            sidecar. Off by default: the extra fields cost extra requests to
            fetch and slow every later backtest that loads the sidecar, so they
            are only produced on request. Providers without extra fields ignore it.
        """

    def load_ohlcv_data(self) -> OHLCVReader:
        """
        Load OHLCV data from the file.

        :return: An OHLCVReader instance.
        """
        return OHLCVReader(str(self.ohlcv_path))
