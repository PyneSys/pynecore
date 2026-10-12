"""SPDX-License-Identifier: Apache-2.0
Standalone public synthetic checks; no data files, services or strategy code.
"""
from pathlib import Path
import math

from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.na import NA
from pynecore.types.ohlcv import OHLCV

ROOT = Path(__file__).resolve().parent


def __test_helper_info__() -> SymInfo:
    """Return synthetic symbol metadata with no exchange calendar.

    :return: Metadata for closed one-minute synthetic bars
    """
    return SymInfo(
        prefix="SYNTH", description="Synthetic", ticker="SYNTH", currency="USD",
        period="1", type="index", mintick=1, pricescale=1, minmove=1,
        pointvalue=1, mincontract=1, opening_hours=[], session_starts=[],
        session_ends=[], timezone="UTC", session="regular",
    )


def __test_helper_values__(script: str, bars: tuple, inputs: dict, copy: bool) -> list:
    """Copy scalar results immediately, including the missing-value mask.

    :param script: Synthetic script filename
    :param bars: Closed, ordered synthetic candles
    :param inputs: Script parameters
    :param copy: Whether the runner copies each result container
    :return: Independent values with missing values represented by None
    """
    runner = ScriptRunner(ROOT / script, bars, __test_helper_info__(), inputs=inputs,
                          lossless_prices=True, last_bar_index=len(bars) - 1)
    values = []
    for _, plots in runner.run_iter(copy_results=copy):
        value = plots["value"]
        missing = isinstance(value, NA) or (isinstance(value, float) and math.isnan(value))
        values.append(None if missing else value)
    return values


def __test_synthetic_feature_reuse__() -> None:
    """Verify feature reuse, warmup, fallback and causal prefix equivalence."""
    bars = tuple(
        OHLCV(946684800000 + index * 60000, 100 + index % 17,
              101 + index % 17, 99 + index % 17, 100 + index % 17, 1)
        for index in range(96)
    )
    runner = ScriptRunner(ROOT / "feature_source.py", bars, __test_helper_info__(),
                          lossless_prices=True, last_bar_index=len(bars) - 1)
    features = [plots["feature"] for _, plots in runner.run_iter(copy_results=False)]
    assert len(features) == len(bars)
    cached = tuple(bar._replace(extra_fields={"feature": value})
                   for bar, value in zip(bars, features))
    assert all(bar.extra_fields is None for bar in bars)
    for offset in (-2.0, 0.0, 3.0):
        direct = __test_helper_values__("feature_direct.py", bars, {"offset": offset}, True)
        fast = __test_helper_values__("feature_direct.py", bars, {"offset": offset}, False)
        reused = __test_helper_values__("feature_cached.py", cached, {"offset": offset}, False)
        assert direct == fast == reused
        assert direct[:6] == [None] * 6
        assert all(value is not None for value in direct[6:])
        prefix = __test_helper_values__("feature_direct.py", bars[:48], {"offset": offset}, False)
        assert reused[:48] == prefix
    # Changing a feature parameter invalidates reuse; the full calculation is needed.
    changed = __test_helper_values__("feature_direct.py", bars, {"length": 11}, False)
    stale = __test_helper_values__("feature_cached.py", cached, {}, False)
    assert changed != stale


if __name__ == "__main__":
    __test_synthetic_feature_reuse__()
    print("PASS: 3 offsets x96bars; warmup, direct/fast/cache, causal prefix and stale-window rejection")
