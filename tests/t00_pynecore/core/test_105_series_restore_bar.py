"""
@pyne

The child-subtree rollback restores a series with the O(changed slots) same-bar
undo (``SeriesImpl._restore_bar``) instead of the full buffer copy. Every user of
that path re-runs the SAME bar the snapshot was taken on, so the undo must be
bit-identical to the full restore for every shape a same-bar re-run can produce
-- and must fall back to the full copy for the shapes it cannot prove.
"""
import random

from pynecore import lib
from pynecore.core import instance_state
from pynecore.core.series import SeriesImpl
from pynecore.types.na import na_float

MAX_BARS_BACK_CHOICES = (2, 3, 5, 8)


def _state_of(series: SeriesImpl) -> tuple:
    """Every field a snapshot carries, as a comparable tuple."""
    return (list(series._buffer), series._size, series._write_pos,
            series._last_bar_index, series._max_bars_back,
            series._max_bars_back_set, series._capacity)


def _new_series(max_bars_back: int, compacted: bool) -> SeriesImpl:
    """A float series with the given capacity and fill mode."""
    return SeriesImpl(max_bars_back, na_float, compacted, False)


def _apply(series: SeriesImpl, op: tuple) -> None:
    """Replay one recorded operation on a series."""
    if op[0] == 'add':
        lib.bar_index = op[1]
        series.add(op[2])
    else:
        series.set(op[1])


def _random_ops(rng: random.Random, bar: int, count: int,
               same_bar_only: bool) -> tuple[list[tuple], int]:
    """Record a random operation sequence, returning it with the final bar index.

    :param rng: Seeded generator.
    :param bar: Bar index the sequence starts on.
    :param count: Number of operations to record.
    :param same_bar_only: Keep every operation on the starting bar (the shape a
        developing re-tick produces) instead of also advancing bars.
    :return: The operations and the bar index they end on.
    """
    ops: list[tuple] = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.35:
            ops.append(('set', rng.uniform(-100.0, 100.0)))
        elif same_bar_only or kind < 0.6:
            ops.append(('add', bar, rng.uniform(-100.0, 100.0)))
        elif kind < 0.85:
            bar += 1
            ops.append(('add', bar, rng.uniform(-100.0, 100.0)))
        else:
            bar += rng.randint(2, 4)  # gap: forward fill (or nothing, compacted)
            ops.append(('add', bar, rng.uniform(-100.0, 100.0)))
    return ops, bar


def __test_restore_bar_matches_restore_fuzz__():
    """`_restore_bar` reproduces `_restore` on every same-bar re-run shape."""
    saved_bar_index = lib.bar_index
    try:
        rng = random.Random(20260918)
        for trial in range(400):
            max_bars_back = rng.choice(MAX_BARS_BACK_CHOICES)
            compacted = rng.random() < 0.3
            history, bar = _random_ops(rng, 0, rng.randint(0, 14), False)
            # A same-bar re-run may also append: a series in a conditionally
            # taken branch forward-fills the bars its branch skipped.
            retick, _ = _random_ops(rng, bar, rng.randint(0, 5),
                                    rng.random() < 0.5)
            new_max_bars_back = None
            if rng.random() < 0.1:
                new_max_bars_back = rng.choice(MAX_BARS_BACK_CHOICES)

            incremental = _new_series(max_bars_back, compacted)
            full = _new_series(max_bars_back, compacted)
            for op in history:
                _apply(incremental, op)
                _apply(full, op)
            baseline = _state_of(incremental)
            snapshot_incremental = incremental._snapshot()
            snapshot_full = full._snapshot()
            for op in retick:
                _apply(incremental, op)
                _apply(full, op)
            if new_max_bars_back is not None:
                incremental.max_bars_back = new_max_bars_back
                full.max_bars_back = new_max_bars_back

            incremental._restore_bar(snapshot_incremental)
            full._restore(snapshot_full)
            assert _state_of(incremental) == _state_of(full), \
                f"trial {trial}: {history} / {retick}"
            assert _state_of(full) == baseline, f"trial {trial}: full restore drifted"
    finally:
        lib.bar_index = saved_bar_index


def __test_restore_bar_directed_shapes__():
    """The named shapes of the undo: same-bar set, appended tail, gap fill,
    at-capacity ring overwrite and the capacity-change fallback."""
    saved_bar_index = lib.bar_index
    try:
        def prepared(bars: int, max_bars_back: int = 3) -> SeriesImpl:
            series = _new_series(max_bars_back, False)
            for i in range(bars):
                lib.bar_index = i
                series.add(float(i))
            return series

        # Same-bar set (no structural change)
        series = prepared(3)
        snapshot = series._snapshot()
        baseline = _state_of(series)
        series.set(99.0)
        series._restore_bar(snapshot)
        assert _state_of(series) == baseline

        # Appended tail while the buffer is still filling
        series = prepared(2)
        snapshot = series._snapshot()
        baseline = _state_of(series)
        lib.bar_index = 2
        series.add(7.0)
        series._restore_bar(snapshot)
        assert _state_of(series) == baseline

        # Gap fill (more than one appended element)
        series = prepared(1)
        snapshot = series._snapshot()
        baseline = _state_of(series)
        lib.bar_index = 3
        series.add(7.0)
        series._restore_bar(snapshot)
        assert _state_of(series) == baseline

        # The single ring overwrite of an at-capacity add
        series = prepared(4)  # capacity is max_bars_back + 1 == 4
        assert series._size == series._capacity
        snapshot = series._snapshot()
        baseline = _state_of(series)
        lib.bar_index = 4
        series.add(7.0)
        assert _state_of(series) != baseline
        series._restore_bar(snapshot)
        assert _state_of(series) == baseline

        # A capacity change between snapshot and restore falls back
        series = prepared(3)
        snapshot = series._snapshot()
        baseline = _state_of(series)
        series.max_bars_back = 8
        series._restore_bar(snapshot)
        assert _state_of(series) == baseline
    finally:
        lib.bar_index = saved_bar_index


def __test_child_snapshot_same_bar_retick__():
    """A `RootChildSnapshot` re-tick restores a child's series and var slots."""
    saved_bar_index = lib.bar_index
    root_key = 'test_105_root'
    try:
        grandchild_layout = {'init': (None, None), 'series': ((1, 3, 'float'),),
                             'varip': (), 'children': ()}
        child_layout = {'init': (None, None, None), 'series': ((1, 3, 'float'),),
                        'varip': (), 'children': ((2, 'grandchild', False),)}
        root_layout = {'init': (None, None), 'series': ((0, 3, 'float'),),
                       'varip': (), 'children': ((1, 'child', False),)}

        instance_state.discard_root(root_key)
        root = instance_state.create_root(root_key, root_layout)
        child = instance_state._make_state(child_layout)
        grandchild = instance_state._make_state(grandchild_layout)
        root[1] = child
        child[2] = grandchild

        child[0] = [1.0, 2.0]
        grandchild[0] = {'a': 1}
        for bar in range(6):
            lib.bar_index = bar
            child[1].add(float(bar))
            grandchild[1].add(float(bar) * 2.0)

        snapshot = instance_state.RootChildSnapshot(keys=[root_key])
        snapshot.save()
        baseline = (_state_of(child[1]), _state_of(grandchild[1]),
                    list(child[0]), dict(grandchild[0]))

        # The re-tick runs on the SAME bar: every add degrades to a set
        child[1].add(-1.0)
        grandchild[1].add(-2.0)
        child[0].append(3.0)
        grandchild[0]['b'] = 2
        assert _state_of(child[1]) != baseline[0]

        snapshot.restore()
        restored_child, restored_grandchild = root[1], root[1][2]
        assert _state_of(restored_child[1]) == baseline[0]
        assert _state_of(restored_grandchild[1]) == baseline[1]
        assert restored_child[0] == baseline[2]
        assert restored_grandchild[0] == baseline[3]
    finally:
        instance_state.discard_root(root_key)
        lib.bar_index = saved_bar_index
