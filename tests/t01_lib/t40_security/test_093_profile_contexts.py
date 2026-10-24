"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Profile Contexts", shorttitle="PC")
def main():
    daily: Series[float] = request.security(syminfo.tickerid, "D", close * 2.0)
    plot(daily, "daily")


# Hourly 24/7 feed on the day grid; every timestamp is Unix MILLISECONDS
__test_helper_T0 = 1_735_689_600_000
__test_helper_HOUR = 3_600_000
__test_helper_DAYS = 6


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for hour in range(__test_helper_DAYS * 24):
        price = 100.0 + hour
        bars.append(OHLCV(timestamp=__test_helper_T0 + hour * __test_helper_HOUR,
                          open=price, high=price, low=price, close=price, volume=1.0))
    return bars


def __test_helper_write_daily(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as writer:
        for bar in __test_helper_bars():
            writer.write(bar)
    SymInfo(
        prefix="EXCH", description="Profile contexts", ticker="PC",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_line(text):
    from pathlib import Path
    lines = Path(__file__).read_text(encoding='utf-8').splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


def __test_profile_holds_the_security_context__(runner, tmp_path):
    """ A profiled run profiles its ``request.security`` child too, as a context """
    import json
    import os
    import sys
    from pathlib import Path

    import pytest

    from pynecore.core import profiler

    if sys.version_info < (3, 13):
        pytest.skip("profiling needs Python 3.13+")
    if profiler._native_profiler is None:  # noqa
        if os.environ.get('PYNE_REQUIRE_NATIVE_PROFILER'):
            pytest.fail("the compiled profiler extension is missing from this build")
        pytest.skip("the compiled profiler extension is not built")

    sys.modules.pop(Path(__file__).stem, None)
    feed = __test_helper_write_daily(tmp_path)
    path = tmp_path / 'profile.json'
    runner(__test_helper_bars(), syminfo_override={"period": "60"},
           security_data={"D": feed}, profile_path=path).run()
    profile = json.loads(path.read_text(encoding='utf-8'))

    assert profile['categories']['security'] > 0
    context, = profile['contexts']
    assert context['timeframe'] == 'D'
    assert context['bars'] > 0 and context['script_ns'] > 0

    file, = [file for file in context['files'] if Path(file['path']) == Path(__file__).resolve()]
    hits = {int(line): counts[1] for line, counts in file['lines'].items()}
    # The child runs main()'s slice for this context: the security line on every one
    # of its bars, and nothing of the plot
    assert hits[__test_helper_line('request.security(syminfo.tickerid, "D"')] >= context['bars']
    assert __test_helper_line('plot(daily, "daily")') not in hits
