"""
@pyne
"""
from pynecore.lib import (
    bar_index, barmerge, close, high, low, na, nz, open, plot, request, script,
    syminfo, ta
)
from pynecore.types import IBPersistent, Series


def tick_count():
    """Count every execution of this expression, re-ticks included.

    A ``varip`` slot is the one piece of child state a developing re-tick does
    NOT roll back, so this counter fingerprints the ROUND SEQUENCE its own
    context ran. It stands in a context of its OWN — a ``varip`` anywhere in a
    context's slice clears its ``closed_shift`` flag, so a skipping context can
    never carry one.

    :return: The number of executions so far in this context.
    """
    n: IBPersistent[int] = 0
    n += 1
    return n


@script.indicator(title="Dev Skip Equivalence", shorttitle="DSE")
def main():
    # The expression of a security call runs in the CHILD, so these three series
    # are the child's own — and a history reference to one of them is a shape a
    # hand-written Pyne script can spell (``ta.sma(close, 5)[1]`` is what the
    # Pine compiler emits for the same idea, and the AST tests cover that).
    sma: Series[float] = ta.sma(close, 5)
    summ: Series[float] = close + open
    counted: Series[float] = close + tick_count()
    # (1) The PineCoders non-repainting idiom: the whole expression is a history
    # reference, so its value is fixed for the period and only the period's
    # first chart bar needs a developing round.
    prev_close: Series[float] = request.security(
        syminfo.tickerid, "60", close[1], lookahead=barmerge.lookahead_on)
    # (2) The same shape over a stateful series, on the same feed — so (1) and
    # (2) are ONE compile-time group and one child serves both.
    prev_sma: Series[float] = request.security(
        syminfo.tickerid, "60", sma[1], lookahead=barmerge.lookahead_on)
    # (3) A tuple whose every element is shifted.
    prev_high, prev_low = request.security(
        syminfo.tickerid, "D", [high[1], low[2]], lookahead=barmerge.lookahead_on)
    # (4) A shifted arithmetic series.
    prev_sum: Series[float] = request.security(
        syminfo.tickerid, "D", summ[1], lookahead=barmerge.lookahead_on)
    # (5) Lazy start: nothing reads this context until the chart is well past
    # its first periods, so its rounds are queued and released at that first
    # read — the skip has to hold for the queue too. On a feed of its own, so
    # the group of (1) and (2) does not start it along with them.
    late: Series[float] = request.security(
        syminfo.tickerid, "240", close[2], lookahead=barmerge.lookahead_on)
    gated: Series[float] = late if bar_index > 100 else nz(gated[1])
    # (6) NOT skippable: the second term reads the developing bar.
    mixed: Series[float] = request.security(
        syminfo.tickerid, "60", close[1] + close, lookahead=barmerge.lookahead_on)
    # (7) NOT skippable: the round sequence is observable through the ``varip``
    # counter this context's slice carries — the slicer clears the flag the
    # shifted expression earned. On a feed of its own: a group shares ONE clone,
    # so a ``varip`` anywhere in the union clears it for every member.
    ticks: Series[float] = request.security(
        syminfo.tickerid, "120", counted[1], lookahead=barmerge.lookahead_on)
    # (8) NOT skippable: ``gaps_on`` answers na between the closes.
    gapped: Series[float] = request.security(
        syminfo.tickerid, "60", close[1], gaps=barmerge.gaps_on,
        lookahead=barmerge.lookahead_on)
    # (9) NOT skippable: a consumer pairs with the round the chart is on, and
    # the consumer itself has a dependency.
    produced: Series[float] = request.security(
        syminfo.tickerid, "60", close[1], lookahead=barmerge.lookahead_on)
    consumed: Series[float] = request.security(
        syminfo.tickerid, "60", produced[1], lookahead=barmerge.lookahead_on)
    # (10) NOT skippable: the chart's own timeframe, where every chart bar is a
    # period of its own.
    same_tf: Series[float] = request.security(
        syminfo.tickerid, "5", close[1], lookahead=barmerge.lookahead_on)
    plot(prev_close, "prev_close")
    plot(prev_sma, "prev_sma")
    plot(prev_high, "prev_high")
    plot(prev_low, "prev_low")
    plot(prev_sum, "prev_sum")
    plot(gated, "gated")
    plot(mixed, "mixed")
    plot(ticks, "ticks")
    plot(gapped, "gapped")
    plot(produced, "produced")
    plot(consumed, "consumed")
    plot(same_tf, "same_tf")
    plot(na if na(prev_close) else prev_close - close, "spread")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 5m/1h/1D grid
__test_helper_step = 300_000  # 5 minutes
__test_helper_day_ms = 86_400_000
__test_helper_days = 1
__test_helper_bars = __test_helper_days * 288


def __test_helper_write_feed(tmp_dir, timeframe, span_ms):
    """Write one security feed at ``timeframe`` over the chart's range.

    :param tmp_dir: Directory to write into.
    :param timeframe: The feed's timeframe string.
    :param span_ms: That timeframe's period length in ms.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"DSE{timeframe}.ohlcv"
    total = (__test_helper_days * __test_helper_day_ms) // span_ms
    with OHLCVWriter(path, timeframe) as w:
        for i in range(total):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c,
                          high=c + 1.0, low=c - 1.0, close=c, volume=10.0))
    SymInfo(
        prefix="PYTEST", description="Dev Skip", ticker="TEST",
        currency="USD", period=timeframe, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 50.0 + (i % 37) * 0.25
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_step,
                          open=c - 0.1, high=c + 0.5, low=c - 0.5, close=c,
                          volume=1.0 + i))
    return bars


def __test_helper_is_na(value):
    """Whether a read answered Pine ``na``, in either of its two shapes.

    :param value: The read value.
    :return: Whether it is ``na``.
    """
    from pynecore.types.na import NA
    return value is None or isinstance(value, NA) or (
        isinstance(value, float) and value != value)


def __test_helper_same(a, b):
    """Pine-style equality for two read values: ``na`` matches ``na``.

    :param a: One value.
    :param b: The other.
    :return: Whether the two reads answered the same.
    """
    if __test_helper_is_na(a) or __test_helper_is_na(b):
        return __test_helper_is_na(a) and __test_helper_is_na(b)
    return a == b


def __test_helper_run(runner, no_skip):
    """Run the script on the per-bar round path, with the skip on or off.

    ``PYNE_NO_SECURITY_DEV_SKIP`` is a RUNTIME switch — the ``closed_shift`` flag
    is emitted either way and only the chart's step building consults it — so the
    two runs share the very same bytecode and need no module eviction beyond the
    one every test in this directory does.

    Two measurements come back with the values. The DECISION per context, by
    wrapping :func:`security.dev_skip_active`, which the chart calls once per
    context as its feed resolves; and the number of developing rounds the chart
    launched, by wrapping :meth:`SyncBlock.set_flags` — every step writes its own
    flags right before it launches, so a write leaving ``FLAG_IS_DEVELOPING`` set
    IS a developing round. The flags are written on the slot of a merged group's
    PRIMARY, so the round count is per child, not per context.

    :param runner: The ``runner`` fixture.
    :param no_skip: Whether to force a developing round per chart bar.
    :return: ``(rows, decisions, dev_rounds)`` — the plot values per chart bar,
        the skip decision per sid, and the developing rounds launched per sid.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    import pynecore.core.security as security_module
    from pynecore.core.security_shm import FLAG_IS_DEVELOPING, SyncBlock

    sys.modules.pop(Path(__file__).stem, None)

    decisions: dict[str, bool] = {}
    dev_rounds: dict[str, int] = {}
    original_set_flags = SyncBlock.set_flags
    original_decide = security_module.dev_skip_active

    def _recording_decide(state, sec_id, **kwargs):
        answer = original_decide(state, sec_id, **kwargs)
        decisions[sec_id] = answer
        return answer

    def _counting_set_flags(self, sec_id, flags):
        if flags & FLAG_IS_DEVELOPING:
            dev_rounds[sec_id] = dev_rounds.get(sec_id, 0) + 1
        return original_set_flags(self, sec_id, flags)

    rows: list[dict] = []
    previous = security_module.NO_BATCH
    # The per-bar path for every context: what the batch replays of it is
    # compared against the same switch in test_086.
    security_module.NO_BATCH = True
    security_module.dev_skip_active = _recording_decide
    SyncBlock.set_flags = _counting_set_flags
    if no_skip:
        os.environ['PYNE_NO_SECURITY_DEV_SKIP'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {
                "5": __test_helper_write_feed(tmp, "5", __test_helper_step),
                "60": __test_helper_write_feed(tmp, "60", 3_600_000),
                "120": __test_helper_write_feed(tmp, "120", 7_200_000),
                "240": __test_helper_write_feed(tmp, "240", 14_400_000),
                "D": __test_helper_write_feed(tmp, "1D", __test_helper_day_ms),
            }
            bars = __test_helper_chart_bars()
            r = runner(bars, security_data=feeds,
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        SyncBlock.set_flags = original_set_flags
        security_module.dev_skip_active = original_decide
        security_module.NO_BATCH = previous
        if no_skip:
            os.environ.pop('PYNE_NO_SECURITY_DEV_SKIP', None)
    return rows, decisions, dev_rounds


def __test_helper_by_index(per_sid):
    """Re-key a per-sid map by the context's declaration index.

    A sec id ends in the position of its ``request.security()`` call, which is
    what the script below can be read against.

    :param per_sid: Map keyed by sec id.
    :return: The same values keyed by index.
    """
    out = {}
    for sec_id, value in per_sid.items():
        out[int(sec_id.rsplit('\xb7', 1)[1])] = value
    return out


def __test_helper_run_with_timeout(fn, seconds=600):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging.

    :param fn: The callable to run.
    :param seconds: How long to wait for it.
    :return: Whatever ``fn`` returned.
    """
    import threading
    box = {}

    def target():
        try:
            box['result'] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box['error'] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"deadlock: the run did not finish within {seconds}s")
    if 'error' in box:
        raise box['error']
    return box.get('result')


def __test_helper_both_runs(runner):
    """Both halves of the A/B, skip first.

    :param runner: The ``runner`` fixture.
    :return: ``(skip result, no-skip result)``, each a
        ``(rows, decisions, dev_rounds)`` triple.
    """
    skip = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_skip=False))
    plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_skip=True))
    return skip, plain


def __test_dev_skip_answers_exactly_like_a_round_per_bar__(runner, log):
    """Every plot is identical with the developing-round skip on and off.

    The skip removes rounds, nothing else: a ``[k>=1]`` expression computes the
    same value on every chart bar of one HTF period, because a developing
    re-tick re-runs the period's first tick from the child's rolled-back
    baseline. Covered here: the plain idiom, a stateful call under the shift, a
    shifted tuple, a shifted sum, two of them merged into one child, a lazily
    started context — and the five shapes the skip must NOT apply to, compared
    value for value on the same run.
    """
    (skip_rows, _sd, _sr), (plain_rows, _pd, _pr) = __test_helper_both_runs(runner)

    assert len(skip_rows) == len(plain_rows) == __test_helper_bars
    for i in range(len(skip_rows)):
        got = skip_rows[i]
        want = plain_rows[i]
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': skip={got[key]!r} per-bar={want[key]!r}")

    log.info("%d bars x %d plots identical with the skip on and off",
             len(skip_rows), len(skip_rows[0]))


def __test_only_the_provable_shapes_skip__(runner, log):
    """The decision per context, and the rounds it actually removed

    Indexes follow the ``request.security()`` calls of the script above: 0-4 are
    the shifted shapes that may skip, 5-10 the ones that may not — a term
    reading the developing bar, a ``varip`` in the slice, ``gaps_on``, a context
    with a consumer, that consumer's own dependency, and the chart's own
    timeframe.

    The last one is DECIDED BY NOT BEING ASKED: the chart's own timeframe is a
    same-context, served inline with no child and no preparation, and the
    decision is taken where a context is prepared (immediately before its
    developing batch is armed). Nothing is asked for it, and its ``dev_skip``
    stays at the default ``False``.
    """
    (_skip_rows, decisions, skipped), (_plain_rows, plain_decisions, plain) = (
        __test_helper_both_runs(runner))

    by_index = __test_helper_by_index(decisions)
    assert by_index == {0: True, 1: True, 2: True, 3: True, 4: True,
                        5: False, 6: False, 7: False, 8: False,
                        9: False}, f"decisions: {by_index}"
    assert 10 not in by_index, \
        "the chart's own timeframe was asked, though it is never prepared"
    assert not any(__test_helper_by_index(plain_decisions).values()), \
        "the switch left a context skipping"

    total_skip = sum(skipped.values())
    total_plain = sum(plain.values())
    assert total_plain - total_skip > 2 * __test_helper_bars, (
        f"developing rounds barely dropped: {total_skip} against {total_plain}")
    # Per child, and only for the contexts that decided to skip: a merged group
    # is scheduled on its primary's slot, so a context that is not one has no
    # count of its own. Whoever does skip is down to one round per HTF period —
    # 24 hourly periods, 6 four-hourly ones and one daily one for this chart,
    # nothing near the chart's own 5-minute scale. A round can write its flags
    # twice (once for the step, once when ``_launch`` marks it as not the bar's
    # last publication), so the bound is per period and not per write.
    hours = __test_helper_bars // 12
    counted = 0
    for sec_id, skips in decisions.items():
        if not skips or sec_id not in skipped:
            continue
        counted += 1
        assert skipped[sec_id] <= 2 * hours + 4, (
            f"'{sec_id}' kept {skipped[sec_id]} developing rounds")
    assert counted >= 3, f"only {counted} skipping children launched rounds"

    log.info("%d developing rounds instead of %d over %d chart bars",
             total_skip, total_plain, __test_helper_bars)
