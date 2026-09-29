"""
@pyne

Regression test: while the ``math.sum`` window fills, each bar is a single
compensated add, not the fused evict-and-add step with a zero eviction.

The two forms only differ when rounding ``s - c`` alone moves the sum, so the
entry stored on such a bar comes out an ulp off and every later eviction of it
carries the error. Volume-scaled and ratio sources hit it all the time, price
sources practically never. Measured on CAPITALCOM:EURUSD 30m (probe ws: 42
warmups of ``close * volume``, ``(close - open) * volume`` and
``(high - low) / close`` at lengths 10..60, 841k displayed bars bit-exact with
the single add, none with the fused form). The bars and expected sums below are
TradingView's own, from bars 148.. and 1480.. of that chart.
"""
from pynecore.lib import close, high, low, math, plot, script, volume


@script.indicator("Math Sum Warmup Single Add")
def main():
    plot(math.sum(close * volume, 11), 'cv')
    plot(math.sum((high - low) / close, 11), 'rng')


# (open, high, low, close, volume)
__test_helper_bars_148 = [
    (1.03832, 1.03835, 1.03771, 1.03771, 836.0),
    (1.03772, 1.03852, 1.03771, 1.03841, 987.0),
    (1.0384, 1.03846, 1.03773, 1.03798, 969.0),
    (1.03797, 1.03831, 1.0376, 1.0381, 997.0),
    (1.03809, 1.03852, 1.03808, 1.03841, 826.0),
    (1.0384, 1.03841, 1.03798, 1.03821, 784.0),
    (1.03822, 1.03854, 1.038, 1.03854, 630.0),
    (1.03856, 1.03928, 1.03856, 1.03927, 579.0),
    (1.03925, 1.03933, 1.03903, 1.03925, 529.0),
    (1.03924, 1.03941, 1.0391, 1.03916, 515.0),
    (1.03915, 1.0396, 1.03907, 1.03952, 644.0),
    (1.03953, 1.04029, 1.03953, 1.03965, 1037.0),
    (1.03964, 1.04039, 1.03963, 1.04002, 1005.0),
    (1.04003, 1.04023, 1.03962, 1.03973, 892.0),
    (1.03974, 1.04158, 1.03974, 1.04126, 1799.0),
    (1.04127, 1.04243, 1.04092, 1.04145, 2927.0),
    (1.04143, 1.04208, 1.0403, 1.04063, 2845.0),
    (1.04062, 1.04251, 1.04051, 1.04249, 2180.0),
    (1.04248, 1.04341, 1.0424, 1.04251, 2392.0),
    (1.04252, 1.04298, 1.0424, 1.04267, 1657.0),
    (1.04266, 1.04342, 1.04165, 1.04228, 2480.0),
    (1.04229, 1.04341, 1.04207, 1.04277, 1750.0),
    (1.04276, 1.04282, 1.04091, 1.04092, 1756.0),
    (1.04091, 1.04145, 1.04048, 1.04132, 1665.0),
    (1.04133, 1.04168, 1.04037, 1.04052, 2009.0),
    (1.04053, 1.0406, 1.0387, 1.03891, 2998.0),
    (1.03892, 1.04036, 1.03883, 1.04012, 2165.0),
    (1.04013, 1.0405, 1.03864, 1.03963, 2588.0),
    (1.03964, 1.03967, 1.03891, 1.03952, 2334.0),
    (1.03953, 1.03975, 1.03908, 1.03908, 2493.0),
]
__test_helper_bars_1480 = [
    (1.04253, 1.04275, 1.04166, 1.04258, 2640.0),
    (1.0426, 1.04316, 1.03728, 1.04091, 6951.0),
    (1.04093, 1.0441, 1.04082, 1.04392, 3258.0),
    (1.04394, 1.04425, 1.04302, 1.0437, 2050.0),
    (1.04369, 1.04602, 1.04354, 1.04576, 1739.0),
    (1.04575, 1.04657, 1.04556, 1.04615, 1445.0),
    (1.04616, 1.04628, 1.0457, 1.04595, 625.0),
    (1.04594, 1.04668, 1.04591, 1.04647, 800.0),
    (1.04628, 1.04669, 1.04619, 1.04662, 175.0),
    (1.04667, 1.0468, 1.0463, 1.04655, 340.0),
    (1.04654, 1.04672, 1.04625, 1.04636, 978.0),
    (1.04637, 1.04642, 1.0456, 1.04615, 905.0),
    (1.04614, 1.04668, 1.04577, 1.04644, 1209.0),
    (1.04642, 1.04642, 1.04589, 1.04606, 1247.0),
    (1.04607, 1.04667, 1.04587, 1.0463, 1368.0),
    (1.04629, 1.04648, 1.04581, 1.04591, 1204.0),
    (1.0459, 1.04609, 1.04567, 1.04571, 1061.0),
    (1.04572, 1.04588, 1.04551, 1.04566, 1059.0),
    (1.04565, 1.04583, 1.04536, 1.04571, 860.0),
    (1.0457, 1.04614, 1.04569, 1.04587, 754.0),
    (1.04588, 1.04637, 1.04586, 1.04596, 583.0),
    (1.04595, 1.04611, 1.04546, 1.04547, 711.0),
    (1.04548, 1.04557, 1.04504, 1.04507, 717.0),
    (1.04509, 1.04529, 1.0447, 1.04525, 816.0),
    (1.04524, 1.04583, 1.04516, 1.04555, 1084.0),
    (1.04556, 1.04608, 1.04554, 1.04584, 1078.0),
    (1.04585, 1.0472, 1.04583, 1.0468, 2110.0),
    (1.04679, 1.04697, 1.04633, 1.04646, 1915.0),
    (1.04647, 1.04803, 1.04635, 1.04782, 2616.0),
    (1.04781, 1.04861, 1.04737, 1.04766, 2529.0),
]
# TradingView's plotted sums from the 11th bar on (the first ten are na)
__test_helper_tv_cv = [
    8615.30691, 8825.8984, 8846.20783, 8767.84437, 9606.08541, 11796.6829, 13943.31861,
    15561.666609999998, 17453.6132, 18631.554139999997, 20681.24114, 21836.637759999998,
    22586.376229999998, 23274.953929999996, 24437.919449999998, 25679.344889999997,
    24882.88054, 24612.850629999997, 24766.462109999997, 24863.204629999997,
]
__test_helper_tv_rng = [
    0.01704680684423804, 0.01678514993513952, 0.012005861647612512, 0.0093705215266941,
    0.008956621021218693, 0.0072257304836446275, 0.0066619267480767415,
    0.00646125043909119, 0.006174898786486918, 0.006127434279645612,
    0.00613726436736733, 0.006309818115261105, 0.006033134766322121,
    0.005727977954019244, 0.005862125912682021, 0.005613858219315178,
    0.006282018206383469, 0.006491962943169739, 0.0077414482536507226,
    0.008475582955731522,
]


def __test_helper_make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="forex", mintick=0.00001, pricescale=100000,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=1.0,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def __test_helper_run(script_path, module_key, rows, name):
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c, volume=v)
        for i, (o, h, lo, c, v) in enumerate(rows)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    return [plot_data[name] for _candle, plot_data in runner.run_iter()]


# noinspection PyShadowingNames
def __test_math_sum_warmup_single_add__(script_path, module_key):
    """
    Both sums reproduce TradingView bit for bit from the first full window on.
    """
    cv = __test_helper_run(script_path, module_key, __test_helper_bars_148, 'cv')
    assert cv[10:] == __test_helper_tv_cv, cv[10:]
    rng = __test_helper_run(script_path, module_key, __test_helper_bars_1480, 'rng')
    assert rng[10:] == __test_helper_tv_rng, rng[10:]
