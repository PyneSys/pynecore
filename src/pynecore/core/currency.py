"""
Currency rate provider for request.currency_rate().

Sources of exchange-rate data, in priority order:

* the chart symbol itself — when the chart is a currency pair (e.g. EURUSD
  chart), ``lib.close`` is the up-to-the-second rate.
* a security context already attached to the run — backtest or live —
  whose ``(basecurrency, currency)`` syminfo matches the requested pair.
  The provider reads the security's latest ``close`` from its
  :class:`ResultBlock` (the same channel ``__sec_read__`` uses).
* legacy file path: a sibling ``.toml`` describing a static
  ``.ohlcv`` file. Only relevant when older callers still pass plain
  paths; live runs never carry these.
"""
from __future__ import annotations

import struct
import bisect
from math import isnan
from pathlib import Path
from typing import TYPE_CHECKING

from .ohlcv_file import RECORD_SIZE
from .security_shm import ResultReader

if TYPE_CHECKING:
    from .syminfo import SymInfo
    from .security_shm import SyncBlock


class CurrencyRateProvider:
    """
    Provides currency exchange rates from running security contexts.

    Walks two input sources at construction time:
    * ``sec_syminfos`` — per-sec_id :class:`SymInfo` for active security
      contexts (the chart-side prefetch in live mode, the backtest
      ``.toml`` cache otherwise).
    * ``security_data`` — legacy file paths kept for backward compatibility
      with callers that have not been migrated yet.

    The chart's own OHLCV is also recognized — when the chart symbol is
    itself a currency pair (e.g. EURUSD), the provider returns
    ``lib.close`` directly instead of consulting any security context.
    """

    __slots__ = (
        '_pair_map', '_chart_pair',
        '_sync_block', '_result_readers',
        '_file_rate_cache',
    )

    def __init__(
            self,
            security_data: 'dict[str, str | Path] | None' = None,
            chart_syminfo: 'SymInfo | None' = None,
            sec_syminfos: 'dict[str, SymInfo] | None' = None,
            sync_block: 'SyncBlock | None' = None,
    ):
        """
        :param security_data: Legacy mapping of user keys to OHLCV file paths.
                              Used only when no matching ``sec_syminfo`` is
                              available (backtest fallback for callers that
                              have not provided pre-loaded syminfos).
        :param chart_syminfo: Chart's SymInfo — if it carries basecurrency,
                              the chart itself becomes a rate source.
        :param sec_syminfos: Per-sec_id :class:`SymInfo` for security
                             contexts active in this run. The provider
                             builds its ``(basecurrency, currency) →
                             sec_id`` lookup primarily from these.
        :param sync_block: The shared :class:`SyncBlock` used by the
                           security infrastructure. Required when any rate
                           source is a sec_id (so :class:`ResultReader`
                           can fetch the latest pickled close from the
                           security's :class:`ResultBlock`).
        """
        # Tagged source: ``('sec', sec_id)`` for in-process security
        # contexts, ``('file', path)`` for legacy file-backed sources.
        self._pair_map: dict[tuple[str, str], tuple[str, str]] = {}
        self._chart_pair: tuple[str, str] | None = None
        self._sync_block: 'SyncBlock | None' = sync_block
        self._result_readers: dict[str, ResultReader] = {}
        self._file_rate_cache: dict[str, tuple[list[int], list[float]]] = {}

        self._build_pair_map(
            security_data or {},
            chart_syminfo,
            sec_syminfos or {},
        )

    def _build_pair_map(
            self,
            security_data: 'dict[str, str | Path]',
            chart_syminfo: 'SymInfo | None',
            sec_syminfos: 'dict[str, SymInfo]',
    ) -> None:
        """
        Index every available rate source by ``(basecurrency, currency)``.

        Priority: chart pair > running security context > static file.
        On collisions within a priority class the first entry wins; this
        keeps the mapping deterministic across runs.
        """
        if chart_syminfo and chart_syminfo.basecurrency:
            self._chart_pair = (chart_syminfo.basecurrency, chart_syminfo.currency)

        for sec_id, syminfo in sec_syminfos.items():
            if not syminfo or not syminfo.basecurrency:
                continue
            pair = (syminfo.basecurrency, syminfo.currency)
            self._pair_map.setdefault(pair, ('sec', sec_id))

        from .syminfo import SymInfo
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
            existing = self._pair_map.get(pair)
            if existing is not None:
                # Sec-backed entries (live or in-process backtest contexts)
                # outrank static files; among files, the larger one wins to
                # match historical behaviour.
                if existing[0] == 'sec':
                    continue
                if self._get_ohlcv_bar_count(ohlcv_path) <= \
                        self._get_ohlcv_bar_count(existing[1]):
                    continue
            self._pair_map[pair] = ('file', ohlcv_path)

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

        if self._chart_pair == (from_cur, to_cur):
            from .. import lib
            return float(lib.close)
        if self._chart_pair == (to_cur, from_cur):
            from .. import lib
            close = float(lib.close)
            return 1.0 / close if close and not isnan(close) and close != 0.0 else float('nan')

        if (from_cur, to_cur) in self._pair_map:
            return self._lookup(self._pair_map[(from_cur, to_cur)], timestamp)

        if (to_cur, from_cur) in self._pair_map:
            rate = self._lookup(self._pair_map[(to_cur, from_cur)], timestamp)
            return 1.0 / rate if rate and not isnan(rate) and rate != 0.0 else float('nan')

        return float('nan')

    def _lookup(self, source: tuple[str, str], timestamp: int) -> float:
        """Dispatch on the tagged source type."""
        kind, value = source
        if kind == 'sec':
            return self._lookup_sec(value)
        return self._lookup_file(value, timestamp)

    def _lookup_sec(self, sec_id: str) -> float:
        """Read the latest pickled close from the security's result block."""
        if self._sync_block is None:
            return float('nan')
        reader = self._result_readers.get(sec_id)
        if reader is None:
            reader = ResultReader(sec_id)
            self._result_readers[sec_id] = reader
        try:
            value = reader.read(self._sync_block, default=None)
        except (FileNotFoundError, OSError):
            # Result block not attached yet — subprocess hasn't written
            # its first bar. Caller treats NaN as "not ready" and emits
            # NA for this bar.
            return float('nan')
        if value is None:
            return float('nan')
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, tuple) and value and isinstance(value[0], (int, float)):
            return float(value[0])
        return float('nan')

    def _lookup_file(self, ohlcv_path: str, timestamp: int) -> float:
        """Static-file path (backtest only)."""
        if ohlcv_path not in self._file_rate_cache:
            self._load_ohlcv(ohlcv_path)
        timestamps, closes = self._file_rate_cache[ohlcv_path]
        if not timestamps:
            return float('nan')
        idx = bisect.bisect_right(timestamps, timestamp) - 1
        if idx < 0:
            return float('nan')
        return closes[idx]

    def _load_ohlcv(self, ohlcv_path: str) -> None:
        """Load timestamps and close prices from binary OHLCV file."""
        timestamps: list[int] = []
        closes: list[float] = []
        path = Path(ohlcv_path)
        if not path.exists():
            self._file_rate_cache[ohlcv_path] = (timestamps, closes)
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
        self._file_rate_cache[ohlcv_path] = (timestamps, closes)

    def close(self) -> None:
        """Close every attached :class:`ResultReader`."""
        for r in self._result_readers.values():
            r.close()
        self._result_readers.clear()

    @staticmethod
    def _get_ohlcv_bar_count(path: str) -> int:
        """Get the number of bars in an OHLCV file without opening it."""
        try:
            return Path(path).stat().st_size // RECORD_SIZE
        except OSError:
            return 0

    @staticmethod
    def _resolve_ohlcv_path(path: str | Path) -> str | None:
        """Resolve OHLCV file path, adding the ``.ohlcv`` extension if needed.

        A dot inside the name belongs to the symbol (e.g. a perpetual
        ``BTCUSDT.P``), so append by name rather than ``with_suffix`` which
        would clobber the symbol's own dotted tail.
        """
        p = Path(path)
        if p.name.endswith('.ohlcv'):
            return str(p) if p.exists() else None
        ohlcv_p = p.with_name(p.name + '.ohlcv')
        if ohlcv_p.exists():
            return str(ohlcv_p)
        return str(p) if p.exists() else None
