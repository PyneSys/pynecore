"""
A download writes its bars in batches, whatever granularity the provider saves at.

A provider that saves bar by bar would otherwise pay two fsyncs per bar: on an
ext4 disk that is about 16 s for 1000 bars, while APFS hides the cost entirely.
"""
import os
from datetime import datetime
from pathlib import Path

import pynecore.core.ohlcv as ohlcv
from pynecore.core.download_runner import download_to_file
from pynecore.core.ohlcv import OHLCVReader
from pynecore.core.plugin.provider import ProviderPlugin
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV

_BARS = 1000
_START_MS = 1_700_000_000_000


class _BarByBarProvider(ProviderPlugin):
    """Serves ``_BARS`` one-minute bars and saves each one on its own."""

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
        for i in range(_BARS):
            self.save_ohlcv_data(OHLCV(_START_MS + i * 60_000, 1.0, 2.0, 0.5, 1.5, 1.0))


def __test_bar_by_bar_download_is_published_in_slices__(tmp_path: Path, monkeypatch):
    """A whole download costs a few fsyncs, and every bar ends up published."""
    provider = _BarByBarProvider(symbol="TEST", timeframe="1", ohlcv_dir=tmp_path)
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        fsyncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(ohlcv.os, "fsync", counting_fsync)
    result = download_to_file(
        provider,
        time_from=datetime(2023, 11, 14),
        time_to=datetime(2023, 11, 15),
        syminfo=None,
    )

    assert result.bars_written == _BARS
    # Creating the empty file, then one publication at the end of the download
    # (1000 bars stay below one slice); bar by bar it would be 2000.
    assert len(fsyncs) <= 4
    assert provider.ohlcv_path is not None
    with OHLCVReader(provider.ohlcv_path) as reader:
        assert reader.size == _BARS
