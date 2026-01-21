"""
@pyne
"""
from pynecore.lib import script, ta, plot, timeframe
from pynecore.types.na import NA


@script.indicator(title="Pivot Point Levels Test", shorttitle="PPL", overlay=False)
def main():
    # Test all pivot types with daily anchor
    daily_change = timeframe.change("D")

    # Traditional
    trad = ta.pivot_point_levels("Traditional", daily_change, False)
    plot(trad[0], "P_trad")
    plot(trad[1], "R1_trad")
    plot(trad[2], "S1_trad")
    plot(trad[3], "R2_trad")
    plot(trad[4], "S2_trad")
    plot(trad[5], "R3_trad")
    plot(trad[6], "S3_trad")

    # Fibonacci
    fib = ta.pivot_point_levels("Fibonacci", daily_change, False)
    plot(fib[0], "P_fib")
    plot(fib[1], "R1_fib")
    plot(fib[2], "S1_fib")

    # DM (DeMark) - only has P, R1, S1
    dm = ta.pivot_point_levels("DM", daily_change, False)
    plot(dm[0], "P_dm")
    plot(dm[1], "R1_dm")
    plot(dm[2], "S1_dm")

    # Camarilla - has R4, S4
    cam = ta.pivot_point_levels("Camarilla", daily_change, False)
    plot(cam[0], "P_cam")
    plot(cam[7], "R4_cam")
    plot(cam[8], "S4_cam")

    # Woodie
    woodie = ta.pivot_point_levels("Woodie", daily_change, False)
    plot(woodie[0], "P_woodie")


# noinspection PyShadowingNames
def __test_pivot_point_levels__(csv_reader, runner, dict_comparator, log):
    """Pivot Point Levels"""
    from pathlib import Path
    syminfo_path = Path(__file__).parent / "data" / "pivot_point_levels.toml"
    with csv_reader('pivot_point_levels.csv', subdir="data") as cr:
        for i, (candle, plot_data) in enumerate(runner(cr, syminfo_path=syminfo_path).run_iter()):
            # Convert TradingView's 1e+100 NA representation to NA
            expected = {k: NA(float) if v == 1e+100 else v for k, v in candle.extra_fields.items()}
            dict_comparator(plot_data, expected)
            if i > 900:
                break
