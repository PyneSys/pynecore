from typing import Callable
from abc import abstractmethod, ABCMeta
from pathlib import Path
from datetime import datetime
import tomllib

from ..types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader


class Provider(metaclass=ABCMeta):
    """
    Base class for all providers
    """

    timezone = 'UTC'
    """ Timezone of the provider """

    symbol: str | None = None
    """ Symbol of the provider """

    timeframe: str | None = None
    """ Timeframe of the provider """

    xchg_timeframe: str | None = None
    """ TradingView timeframe """

    ohlcv_path: Path | None = None
    """ Directory to save OHLV data """

    config_keys = {
        '# Settings for the provider': '',
    }
    """ Key-value pairs to put into providers.toml, if key starts with '#' it is a comment. """

    config: dict[str, str] = {}
    """ Config dict for the exchange loaded from providers.toml """

    @classmethod
    @abstractmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to TradingView fmt
        https://www.tradingview.com/pine-script-reference/v6/#var_timeframe.period
        """

    @classmethod
    @abstractmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to exchange fmt
        """

    @classmethod
    def get_ohlcv_path(cls, symbol: str, timeframe: str, ohlv_dir: Path, provider_name: str | None = None) -> Path:
        """
        Get the output path of the OHLV data
        """
        return ohlv_dir / (f"{provider_name or cls.__name__.lower().replace('provider', '')}"
                           f"_{symbol.replace('/', '_').replace(':', '_').upper()}"
                           f"_{timeframe}.ohlcv")

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlv_dir: Path | None = None, config_dir: Path | None = None):
        """
        :param symbol: The symbol to get data for
        :param timeframe: The timeframe to get data for in TradingView fmt
        :param ohlv_dir: The directory to save OHLV data
        :param config_dir: The directory to read the config file from
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.xchg_timeframe = self.to_exchange_timeframe(timeframe) if timeframe else None
        self.ohlcv_path = self.get_ohlcv_path(symbol, timeframe, ohlv_dir) if ohlv_dir else None
        self.ohlcv_file = OHLCVWriter(self.ohlcv_path) if self.ohlcv_path else None

        if not config_dir:  # Default config dir from the parent of the ohlcv_dir
            assert self.ohlcv_path is not None
            config_dir = self.ohlcv_path.parent.parent / 'config'
        self.config_dir = config_dir

        self.load_config()

    def __enter__(self) -> OHLCVWriter:
        assert self.ohlcv_file is not None
        return self.ohlcv_file.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.ohlcv_file is not None
        self.ohlcv_file.close()

    @abstractmethod
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """
        Get list of symbols
        """

    def load_config(self):
        """
        Load config from providers.toml
        """
        with open(self.config_dir / 'providers.toml', 'rb') as f:
            data = tomllib.load(f)
            self.config = data[self.__class__.__name__.replace('Provider', '').lower()]

    @abstractmethod
    def update_symbol_info(self) -> SymInfo:
        """
        Update symbol info from the exchange
        """

    def is_symbol_info_exists(self) -> bool:
        """
        Check if symbol info file exists
        """
        assert self.ohlcv_path is not None
        return self.ohlcv_path.with_suffix('.toml').exists()

    def get_symbol_info(self, force_update=False) -> SymInfo:
        """
        Get market details of a symbol

        :param force_update: Force update the symbol info
        """
        assert self.ohlcv_path is not None
        toml_path = self.ohlcv_path.with_suffix('.toml')
        # Check if file already exists
        if self.is_symbol_info_exists() and not force_update:
            return SymInfo.load_toml(toml_path)

        sym_info = self.update_symbol_info()
        sym_info.save_toml(toml_path)
        return sym_info

    @abstractmethod
    def get_opening_hours_and_sessions(self) \
            -> tuple[list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """
        Get opening hours and sessions of a symbol
        """

    def save_ohlcv_data(self, data: OHLCV | list[OHLCV]):
        """
        Save OHLV data to a file

        :param data: OHLV data
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
                       limit: int | None = None):
        """
        Download OHLV data

        In the user code you can call `self.save_ohlcv_data()` to save the data into the data file

        :param time_from: The start time
        :param time_to: The end time
        :param on_progress: Optional callback to call on progress
        :param limit: Override the automatic chunk size (number of bars per API request)
        """

    def load_ohlcv_data(self) -> OHLCVReader:
        """
        Load OHLV data from the file
        """
        return OHLCVReader(str(self.ohlcv_path))
