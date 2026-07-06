"""
@pyne
"""
import struct
import tempfile
from pathlib import Path

from pynecore.core.currency import CurrencyRateProvider
from pynecore.core.script_runner import ScriptRunner


def _make_ohlcv(dir_path: Path, stem: str) -> str:
    """Create a minimal ``<stem>.ohlcv`` (+ empty ``.toml``) in ``dir_path``.

    ``stem`` may itself contain a dot (a perpetual symbol like ``BTCUSDT.P``),
    so the extension is appended by name — never via ``Path.with_suffix``,
    which would clobber the symbol's own dotted tail (the very bug under test).

    :return: The suffix-less stem path, as passed to ``--security``.
    """
    base = dir_path / stem
    ohlcv_path = base.with_name(base.name + ".ohlcv")
    with open(ohlcv_path, "wb") as f:
        f.write(struct.pack("Ifffff", 1000, 1.0, 1.0, 1.0, 1.0, 100.0))
    with open(base.with_name(base.name + ".toml"), "w") as f:
        f.write("")
    return str(base)


def __test_ensure_ohlcv_ext_keeps_dotted_symbol__(log):
    """_ensure_ohlcv_ext resolves a stem whose symbol carries a dot (BTCUSDT.P)"""
    with tempfile.TemporaryDirectory() as d:
        stem = _make_ohlcv(Path(d), "sec_BINANCE_BTCUSDT.P_60")
        resolved = ScriptRunner._ensure_ohlcv_ext(stem)
        # The .ohlcv extension is appended to the FULL stem, not substituted for
        # the ".P_60" tail (which with_suffix would misread as an extension).
        assert resolved == stem + ".ohlcv"
        assert Path(resolved).exists()
        # Regression guard: never collapse "BTCUSDT.P_60" down to "BTCUSDT".
        assert not resolved.endswith("sec_BINANCE_BTCUSDT.ohlcv")


def __test_ensure_ohlcv_ext_accepts_full_dotted_path__(log):
    """_ensure_ohlcv_ext leaves an already-suffixed dotted path untouched"""
    with tempfile.TemporaryDirectory() as d:
        stem = _make_ohlcv(Path(d), "sec_BINANCE_BTCUSDT.P_60")
        full = stem + ".ohlcv"
        assert ScriptRunner._ensure_ohlcv_ext(full) == full


def __test_resolve_ohlcv_path_keeps_dotted_symbol__(log):
    """CurrencyRateProvider._resolve_ohlcv_path resolves a dotted-symbol stem"""
    with tempfile.TemporaryDirectory() as d:
        stem = _make_ohlcv(Path(d), "sec_BINANCE_BTCUSDT.P_60")
        resolved = CurrencyRateProvider._resolve_ohlcv_path(stem)
        assert resolved == stem + ".ohlcv"
        # The mangled sibling never exists and must not be returned.
        assert not str(resolved).endswith("sec_BINANCE_BTCUSDT.ohlcv")


def __test_resolve_ohlcv_path_missing_returns_none__(log):
    """_resolve_ohlcv_path returns None when neither stem nor .ohlcv exists"""
    with tempfile.TemporaryDirectory() as d:
        missing = str(Path(d) / "sec_BINANCE_ETHUSDT.P_60")
        assert CurrencyRateProvider._resolve_ohlcv_path(missing) is None
