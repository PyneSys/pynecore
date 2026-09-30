"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo


@script.indicator(title="Protocol Released", shorttitle="PR")
def main():
    own = request.security(syminfo.tickerid, "5", close)
    plot(own, "own")


__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, Unix MILLISECONDS
__test_helper_step = 300_000  # the chart's 5 minutes


def __test_helper_chart_bars():
    """One hour of 5-minute chart bars."""
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(12):
        c = 50.0 + i
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_step,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    return bars


def __test_finished_run_releases_its_security_states__(runner, log):
    """A finished run leaves no security protocol and no security state behind

    The protocol functions are closures over the run's ``SecurityState``
    objects, each holding OS semaphores. Left in the script module's globals
    they would keep every state of the last run alive for as long as the module
    is imported, and a process running many scripts runs out of semaphores.
    """
    import gc
    import os
    import sys
    from pathlib import Path

    from pynecore.core.security import SecurityState

    sys.modules.pop(Path(__file__).stem, None)
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    r = runner(__test_helper_chart_bars())
    rows = [dict(plots) for _candle, plots in r.run_iter()]
    assert len(rows) == 12
    module = r.script_module
    for name in ('__sec_signal__', '__sec_write__', '__sec_read__', '__sec_wait__',
                 '__active_security__', '__same_context__'):
        assert name not in vars(module), name

    del r
    gc.collect()
    assert not any(isinstance(obj, SecurityState) for obj in gc.get_objects())
    log.info("the finished run's security states are collectable")
