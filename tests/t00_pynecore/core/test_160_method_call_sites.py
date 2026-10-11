"""
@pyne
"""
from pynecore.core.overload import overload
from pynecore.core.pine_method import method, method_call
from pynecore.lib import array, plot, script
from pynecore.types import Persistent


class Obj:
    k: float = 0.0


def get_obj() -> Obj:
    return Obj()


@method
@overload
def kind(this: int) -> float:
    return 1.0


@method
@overload
def kind(this: float) -> float:
    return 2.0


@script.indicator(title="Method call sites", shorttitle="method_call_sites")
def main():
    @method
    def m(self: Obj, x: float) -> float:
        cnt: Persistent[int] = 0
        first: Persistent[float] = x
        cnt += 1
        return cnt * 1000 + first

    captured = array.new_float()
    array.push(captured, 1.0)
    array.push(captured, 2.0)

    @method
    def size(this: float, x: float) -> float:
        return this + array.size(captured) + x

    index: int = 1

    @method
    def get(this: list[float], i: int) -> float:
        return array.get(this, i + index)

    r: int = 14
    plot(method_call(m, get_obj(), 1.0), "site_a")
    plot(method_call(m, get_obj(), 2.0), "site_b")
    plot(kind(r / 8), "direct_kind")
    plot(method_call(kind, r / 8), "method_kind")
    plot(method_call(size, 3.0, 10.0), "builtin_named")
    plot(method_call(get, captured, 0), "builtin_wins")
    plot(method_call(get, captured, index=0), "builtin_keyword")


def __test_helper_bars(count: int):
    """Hourly bars with increasing timestamps."""
    from pynecore.types.ohlcv import OHLCV
    start = 1735689600000  # 2025-01-01T00:00:00Z
    return iter([OHLCV(timestamp=start + i * 3600000, open=1.0, high=2.0, low=0.5,
                       close=1.1, volume=10.0) for i in range(count)])


def __test_method_call_sites__(runner):
    """ Every ``method_call`` site keeps its own method state, an overloaded
    method selects by the static argument type like a direct call, a closure
    capture never stands in for the receiver of a builtin-named method, and a
    builtin that wins over a closure-converted method takes none of its captures,
    not even when a visible keyword shares a capture's name """
    rows = [p for _, p in runner(__test_helper_bars(3)).run_iter()]
    assert [(p['site_a'], p['site_b']) for p in rows] == \
        [(1001.0, 1002.0), (2001.0, 2002.0), (3001.0, 3002.0)]
    # ``r / 8`` is int-typed with a fractional value: the int implementation
    assert all(p['direct_kind'] == 1.0 and p['method_kind'] == 1.0 for p in rows)
    assert all(p['builtin_named'] == 15.0 for p in rows)
    assert all(p['builtin_wins'] == 1.0 for p in rows)
    assert all(p['builtin_keyword'] == 1.0 for p in rows)
