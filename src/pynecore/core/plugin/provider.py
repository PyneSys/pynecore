from typing import Callable
from abc import abstractmethod, ABCMeta
from pathlib import Path
from datetime import datetime

from pynecore.types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader

from . import Plugin, ConfigT


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
    def get_ohlcv_path(cls, symbol: str, timeframe: str, ohlv_dir: Path,
                       provider_name: str | None = None) -> Path:
        """
        Get the output path of the OHLCV data file.

        :param symbol: Symbol name.
        :param timeframe: Timeframe in TradingView format.
        :param ohlv_dir: Directory to save OHLCV data.
        :param provider_name: Override provider name in filename.
        :return: Path to the OHLCV file.
        """
        return ohlv_dir / (f"{provider_name or cls.__name__.lower().replace('provider', '').replace('plugin', '')}"
                           f"_{symbol.replace('/', '_').replace(':', '_').upper()}"
                           f"_{timeframe}.ohlcv")

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlv_dir: Path | None = None, config: ConfigT | None = None):
        """
        :param symbol: The symbol to get data for.
        :param timeframe: The timeframe to get data for in TradingView format.
        :param ohlv_dir: The directory to save OHLCV data.
        :param config: Pre-loaded config dataclass instance.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.xchg_timeframe = self.to_exchange_timeframe(timeframe) if timeframe else None
        self.ohlcv_path = self.get_ohlcv_path(symbol, timeframe, ohlv_dir) if ohlv_dir else None
        self.ohlcv_file = OHLCVWriter(self.ohlcv_path) if self.ohlcv_path else None
        self.config: ConfigT | None = config

    def __enter__(self) -> OHLCVWriter:
        assert self.ohlcv_file is not None
        return self.ohlcv_file.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.ohlcv_file is not None
        self.ohlcv_file.close()

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
    def download_ohlcv(self, time_from: datetime | None, time_to: datetime | None,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None):
        """
        Download OHLCV data from the exchange.

        Use :meth:`save_ohlcv_data` to write records to the data file.

        :param time_from: The start time (None to fetch all available data).
        :param time_to: The end time (None to fetch up to the latest).
        :param on_progress: Optional progress callback.
        :param limit: Override the automatic chunk size (number of bars per API request).
        """

    def load_ohlcv_data(self) -> OHLCVReader:
        """
        Load OHLCV data from the file.

        :return: An OHLCVReader instance.
        """
        return OHLCVReader(str(self.ohlcv_path))
