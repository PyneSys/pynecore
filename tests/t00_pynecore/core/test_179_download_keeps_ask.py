"""A continued download keeps fetching the ask side of a file that already stores it."""
from datetime import datetime
from pathlib import Path

from pynecore.core.download_runner import download_to_file
from pynecore.core.ohlcv import OHLCVReader
from pynecore.core.plugin.provider import ExtraData, ProviderPlugin
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV, AskOHLC

_START_MS = 1_700_000_000_000


class _AskProvider(ProviderPlugin):
    """Serves three new one-minute bars per call, with their ask side only on request."""

    extra_data = ExtraData(label="Ask side", help="Rebuilt from ticks.")
    calls: list[bool]
    next_index = 0

    @classmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return timeframe

    @classmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        return timeframe

    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        return []

    def update_symbol_info(self) -> SymInfo:
        raise NotImplementedError

    def download_ohlcv(self, time_from, time_to, on_progress=None, limit=None,
                       with_extra=False):
        self.calls.append(with_extra)
        for index in range(self.next_index, self.next_index + 3):
            ask = AskOHLC(1.25, 2.25, 0.75, 1.75) if with_extra else None
            self.save_ohlcv_data(OHLCV(_START_MS + index * 60_000, 1.0, 2.0, 0.5, 1.5, 1.0,
                                       ask=ask))
        self.next_index += 3


def __test_helper_download(tmp_path: Path, calls: list[bool], next_index: int, **kwargs):
    provider = _AskProvider(symbol="TEST", timeframe="1", ohlcv_dir=tmp_path)
    provider.calls = calls
    provider.next_index = next_index
    download_to_file(provider, time_to=datetime(2023, 11, 15), syminfo=None, **kwargs)
    assert provider.ohlcv_path is not None
    return provider.ohlcv_path


def __test_continuation_keeps_the_ask_side__(tmp_path: Path):
    """Without the opt-in, continuing an ask file still asks the provider for the ask."""
    calls: list[bool] = []
    __test_helper_download(tmp_path, calls, 0, time_from=datetime(2023, 11, 14),
                           extra_data=True)
    path = __test_helper_download(tmp_path, calls, 3, time_from="continue")

    assert calls == [True, True]
    with OHLCVReader(path) as reader:
        asks = [bar.ask for bar in reader]
    assert asks == [AskOHLC(1.25, 2.25, 0.75, 1.75)] * 6


def __test_conflict_truncation_does_not_opt_in__(tmp_path: Path):
    """Replacing an ask file through a conflict keeps no bars, so it asks for no ask."""
    calls: list[bool] = []
    __test_helper_download(tmp_path, calls, 3, time_from=datetime(2023, 11, 14, 12),
                           extra_data=True)
    __test_helper_download(tmp_path, calls, 0, time_from=datetime(2023, 11, 14),
                           on_conflict='truncate')

    assert calls == [True, False]


def __test_plain_file_and_truncation_stay_without_ask__(tmp_path: Path):
    """A file without ask columns gets no opt-in, and a truncated download starts over."""
    calls: list[bool] = []
    __test_helper_download(tmp_path, calls, 0, time_from=datetime(2023, 11, 14))
    __test_helper_download(tmp_path, calls, 3, time_from="continue")
    path = __test_helper_download(tmp_path, calls, 0, time_from=datetime(2023, 11, 14),
                                  extra_data=True, on_conflict='truncate')
    with OHLCVReader(path) as reader:
        assert reader.has_ask
    path = __test_helper_download(tmp_path, calls, 0, time_from=datetime(2023, 11, 14),
                                  truncate=True)

    assert calls == [False, False, True, False]
    with OHLCVReader(path) as reader:
        assert not reader.has_ask
