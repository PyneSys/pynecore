"""
Currency rate provider for request.currency_rate().

Resolves exchange rates from OHLCV data files by scanning their TOML metadata
for basecurrency/currency pairs. Supports lazy-loading and timestamp-based
binary search for efficient per-bar rate lookups.
"""
from __future__ import annotations

import struct
import bisect
from math import isnan
from pathlib import Path
from typing import TYPE_CHECKING

from .ohlcv_file import RECORD_SIZE

if TYPE_CHECKING:
    from .syminfo import SymInfo


class CurrencyRateProvider:
    """
    Provides currency exchange rates from OHLCV close prices.

    At construction time, scans TOML metadata for all provided OHLCV paths
    to build a (basecurrency, currency) → ohlcv_path mapping.
    At runtime, lazy-loads OHLCV data and uses binary search to find
    the closest prior bar for a given timestamp.

    The chart's own OHLCV can also serve as a rate source — when the chart
    symbol is itself a currency pair (e.g. EURUSD), the provider returns
    ``lib.close`` directly instead of loading from file.
    """

    __slots__ = ('_pair_map', '_rate_cache', '_chart_pair')

    def __init__(
            self,
            security_data: dict[str, str | Path],
            chart_syminfo: SymInfo | None = None,
    ):
        """
        :param security_data: Maps user keys to OHLCV file paths
        :param chart_syminfo: Chart's SymInfo — if it has basecurrency,
                              the chart itself becomes a rate source
        """
        self._pair_map: dict[tuple[str, str], str] = {}
        self._rate_cache: dict[str, tuple[list[int], list[float]]] = {}
        self._chart_pair: tuple[str, str] | None = None

        self._build_pair_map(security_data, chart_syminfo)

    def _build_pair_map(
            self,
            security_data: dict[str, str | Path],
            chart_syminfo: SymInfo | None,
    ) -> None:
        """
        Scan TOML files and build (basecurrency, currency) → ohlcv_path mapping.
        If multiple files provide the same pair, the one with the most bars wins.
        """
        from .syminfo import SymInfo

        if chart_syminfo and chart_syminfo.basecurrency:
            self._chart_pair = (chart_syminfo.basecurrency, chart_syminfo.currency)

        for _key, path in security_data.items():
            ohlcv_path = self._resolve_ohlcv_path(path)
            if ohlcv_path is None:
                continue

            toml_path = Path(ohlcv_path).with_suffix('.toml')
            if not toml_path.exists():
                continue

            try:
                syminfo = SymInfo.load_toml(toml_path)
            except (ValueError, KeyError):
                continue

            if not syminfo.basecurrency:
                continue

            pair = (syminfo.basecurrency, syminfo.currency)
            if pair in self._pair_map:
                existing_size = self._get_ohlcv_bar_count(self._pair_map[pair])
                new_size = self._get_ohlcv_bar_count(ohlcv_path)
                if new_size <= existing_size:
                    continue

            self._pair_map[pair] = ohlcv_path

    def get_rate(self, from_cur: str, to_cur: str, timestamp: int) -> float:
        """
        Get exchange rate for a currency pair at a given timestamp.

        :param from_cur: Source currency code (e.g. "EUR")
        :param to_cur: Target currency code (e.g. "USD")
        :param timestamp: UNIX timestamp (seconds)
        :return: Exchange rate, or NaN if unavailable
        """
        if from_cur == "NONE" or to_cur == "NONE":
            return float('nan')
        if from_cur == to_cur:
            return 1.0

        # Chart pair — return lib.close directly
        if self._chart_pair == (from_cur, to_cur):
            from .. import lib
            return float(lib.close)
        if self._chart_pair == (to_cur, from_cur):
            from .. import lib
            close = float(lib.close)
            return 1.0 / close if close and not isnan(close) and close != 0.0 else float('nan')

        # Direct pair from OHLCV
        if (from_cur, to_cur) in self._pair_map:
            return self._lookup(self._pair_map[(from_cur, to_cur)], timestamp)

        # Inverse pair
        if (to_cur, from_cur) in self._pair_map:
            rate = self._lookup(self._pair_map[(to_cur, from_cur)], timestamp)
            return 1.0 / rate if rate and not isnan(rate) and rate != 0.0 else float('nan')

        return float('nan')

    def _lookup(self, ohlcv_path: str, timestamp: int) -> float:
        """
        Look up the close price at or before the given timestamp.
        Lazy-loads and caches the OHLCV data on first access.
        """
        if ohlcv_path not in self._rate_cache:
            self._load_ohlcv(ohlcv_path)

        timestamps, closes = self._rate_cache[ohlcv_path]
        if not timestamps:
            return float('nan')

        # Binary search: find rightmost timestamp <= target
        idx = bisect.bisect_right(timestamps, timestamp) - 1
        if idx < 0:
            return float('nan')

        return closes[idx]

    def _load_ohlcv(self, ohlcv_path: str) -> None:
        """
        Load timestamps and close prices from binary OHLCV file.
        Reads directly from the binary format for efficiency — only extracts
        the timestamp (uint32 at offset 0) and close (float32 at offset 16)
        from each 24-byte record.
        """
        timestamps: list[int] = []
        closes: list[float] = []

        path = Path(ohlcv_path)
        if not path.exists():
            self._rate_cache[ohlcv_path] = (timestamps, closes)
            return

        file_size = path.stat().st_size
        bar_count = file_size // RECORD_SIZE

        with open(path, 'rb') as f:
            data = f.read()

        for i in range(bar_count):
            offset = i * RECORD_SIZE
            ts = struct.unpack_from('I', data, offset)[0]
            close = struct.unpack_from('f', data, offset + 16)[0]
            timestamps.append(ts)
            closes.append(close)

        self._rate_cache[ohlcv_path] = (timestamps, closes)

    @staticmethod
    def _resolve_ohlcv_path(path: str | Path) -> str | None:
        """Resolve OHLCV file path, adding .ohlcv extension if needed."""
        p = Path(path)
        if p.suffix == '.ohlcv':
            return str(p) if p.exists() else None
        ohlcv_p = p.with_suffix('.ohlcv')
        if ohlcv_p.exists():
            return str(ohlcv_p)
        return str(p) if p.exists() else None

    @staticmethod
    def _get_ohlcv_bar_count(path: str) -> int:
        """Get the number of bars in an OHLCV file without opening it."""
        try:
            return Path(path).stat().st_size // RECORD_SIZE
        except OSError:
            return 0
