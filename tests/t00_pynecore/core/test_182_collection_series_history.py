"""
@pyne

The history of an array, a matrix or a map is the object as it was when the
previous bar closed, a shallow copy whose elements stay shared; the history of a
user-defined object is the object itself. Measured on TradingView
(CAPITALCOM:EURUSD 1D) with one push per bar:

- ``(a[1]).size()``, ``(m[1]).size()``, ``(mx[1]).rows()``: ``bar_index``
- an alias recorded BEFORE the bar's push, ``(b[1]).size()``: ``bar_index``
- an array read before the bar's push: ``bar_index``
- an array of a UDT holding a growing array, ``(h[1]).get(0).arr.size()``:
  ``bar_index + 1`` (the element is shared)
- a UDT counter, ``(t[1]).v``: ``bar_index + 1`` (a reference)
- the history of a call returning the same growing array, ``(f()[1]).size()``
  (``inline_series``): ``bar_index``, also when the same helper serves a scalar
  call site as well
- a declared series parameter, ``hist(x) => x[1]`` called with the array and
  with ``close``: ``bar_index`` for the array, also through a local series
  alias of the parameter
"""
from pynecore import lib
from pynecore.core.instance_state import _configure, _make_state, _rewind_vector
from pynecore.core.pine_udt import udt
from pynecore.core.series import CollectionSeriesImpl, close_collection_bar, inline_series
from pynecore.lib import array, bar_index, close, map, matrix, na, script
from pynecore.types import Persistent, PersistentSeries, Series
from pynecore.types.matrix import Matrix


@udt
class Holder:
    arr: list[float] = na


@udt
class Counter:
    v: float = 0.0


def grow():
    kept: Persistent[list[float]] = array.new_float(0)
    array.push(kept, close)
    return kept


def ident(x):
    return x


def lag(x):
    return inline_series(ident(x), 1)


def hist(x: Series):
    return x[1]


def hist_alias(x):
    alias: Series = x
    return alias[1]


@script.indicator(title="collection series history")
def main():
    early = -1
    a: PersistentSeries[list[float]] = array.new_float(0)
    if bar_index > 0:
        early = array.size(a[1])
    array.push(a, close)

    m: PersistentSeries[dict[int, float]] = map.new()
    map.put(m, int(bar_index), close)

    mx: PersistentSeries[Matrix[float]] = matrix.new(0, 1, na)
    matrix.add_row(mx, matrix.rows(mx), array.from_items(close))

    base: Persistent[list[float]] = array.new_float(0)
    alias: Series = base
    array.push(base, close)

    inner: Persistent[Holder] = Holder.new(array.new_float(0))
    holders: PersistentSeries[list[Holder]] = array.new(0, na)
    if bar_index == 0:
        array.push(holders, inner)
    array.push(inner.arr, close)

    counter: PersistentSeries[Counter] = Counter.new()
    counter.v += 1

    grown = inline_series(grow(), 1)
    lagged = lag(a)
    lagged_close = lag(close)
    declared = hist(a)
    declared_close = hist(close)
    aliased = hist_alias(a)
    aliased_close = hist_alias(close)

    first = bar_index == 0
    return {
        "bi": bar_index,
        "early": early,
        "array": -1 if first else array.size(a[1]),
        "map": -1 if first else map.size(m[1]),
        "matrix": -1 if first else matrix.rows(mx[1]),
        "alias": -1 if first else array.size(alias[1]),
        "shared": -1 if first else array.size(array.get(holders[1], 0).arr),
        "udt": -1 if first else counter[1].v,
        "inline": -1 if first else array.size(grown),
        "poly": -1 if first else array.size(lagged),
        "poly_scalar": -1 if first else lagged_close,
        "declared": -1 if first else array.size(declared),
        "declared_scalar": -1 if first else declared_close,
        "aliased": -1 if first else array.size(aliased),
        "aliased_scalar": -1 if first else aliased_close,
    }


def __test_collection_history_is_the_closed_bar_copy__(csv_reader, runner):
    """Arrays, matrices and maps read the previous bar's closing state, their
    elements and user-defined objects stay references (TradingView parity)"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        bars = 0
        previous_close = None
        for candle, plot in runner(cr).run_iter():
            bi = plot["bi"]
            if bi == 0:
                for key in ("early", "array", "map", "matrix", "alias", "shared", "udt",
                            "inline", "poly", "declared", "aliased"):
                    assert plot[key] == -1, key
            else:
                for key in ("early", "array", "map", "matrix", "alias", "inline", "poly",
                            "declared", "aliased"):
                    assert plot[key] == bi, f"{key} on bar {bi}: {plot[key]}"
                assert plot["shared"] == bi + 1
                assert plot["udt"] == bi + 1
                assert plot["poly_scalar"] == previous_close
                assert plot["declared_scalar"] == previous_close
                assert plot["aliased_scalar"] == previous_close
            previous_close = candle.close
            bars += 1
            if bars == 6:
                break
        assert bars == 6


def __test_rolled_back_first_write_is_not_frozen__():
    """A collection series whose first write a discarded pass rolled back is empty
    when the bar closes: closing it must skip it, not index the empty buffer"""
    saved = lib.bar_index
    try:
        series = CollectionSeriesImpl(5, None, False)
        snapshot = series._snapshot()
        lib.bar_index = 3
        series.add([1.0])
        series._restore(snapshot)
        close_collection_bar()
        assert series._size == 0
    finally:
        lib.bar_index = saved


def __test_vector_selected_collection_series_survives_a_rewind__():
    """A series parameter the instance vector makes a collection series stays one
    when a discarded pass rewinds the instance, and a plain instance keeps a plain one"""
    layout = {'init': (None, (None,)), 'series': ((0, None, None),), 'varip': (),
              'children': (), 'names': ('x', '·pins'), 'pin': 1, 'series_pins': ((0, 0),)}
    collection = _configure(_make_state(layout), layout, ('collection',))
    plain = _configure(_make_state(layout), layout, (None,))
    assert type(collection[0]) is CollectionSeriesImpl
    assert type(plain[0]) is not CollectionSeriesImpl
    _rewind_vector(collection, layout)
    assert type(collection[0]) is CollectionSeriesImpl
