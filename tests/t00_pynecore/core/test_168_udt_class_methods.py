"""
@pyne edge
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import close, na, plot, script, ta
from pynecore.types import Persistent


@udt
class Pair:
    a: float = na(float)
    b: float = na(float)
    step: int = 1

    def crossed(self) -> bool:
        return ta.crossover(self.a, self.b)

    def fast(self) -> float:
        return ta.ema(self.a, 5)

    def chained(self) -> float:
        return self.fast() + ta.sma(self.b, 3)

    def count(self) -> int:
        n: Persistent[int] = 0
        n += self.step
        return n

    def smooth(self, x: float) -> float:
        return ta.ema(x, 3)


@script.indicator(title="UDT class methods", shorttitle="udt_class_methods")
def main():
    p: Pair = Pair(ta.ema(close, 9), ta.ema(close, 21))
    plot(1 if p.crossed() else 0, "crossed")
    plot(1 if ta.crossover(ta.ema(close, 9), ta.ema(close, 21)) else 0, "crossed_main")
    plot(p.fast(), "fast")
    plot(Pair.fast(p), "fast_class")
    plot(ta.ema(ta.ema(close, 9), 5), "fast_main")
    plot(p.chained(), "chained")
    plot(ta.ema(ta.ema(close, 9), 5) + ta.sma(ta.ema(close, 21), 3), "chained_main")
    plot(p.count(), "count_a")
    plot(p.count(), "count_b")
    total = 0
    for i in range(3):
        total += p.count()
    plot(total, "loop_total")
    via_instance: float = 0.0
    via_class: float = 0.0
    via_main: float = 0.0
    for i in range(3):
        via_instance = p.smooth(close + i)
        via_class = Pair.smooth(p, close + i)
        via_main = ta.ema(close + i, 3)
    plot(via_instance, "loop_instance")
    plot(via_class, "loop_class")
    plot(via_main, "loop_main")


def __test_class_body_methods__(csv_reader, runner):
    """ A method in the class body keeps its ``ta`` and ``var`` state per call
    site, exactly like the same code written in ``main`` """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        rows = [p for _, p in runner(cr).run_iter()]
    assert sum(p['crossed'] for p in rows) > 0
    for bar, p in enumerate(rows):
        assert p['crossed'] == p['crossed_main'], bar
        for name in ('fast', 'fast_class', 'chained'):
            main_value = p[name.split('_')[0] + '_main']
            assert (p[name] != p[name]) == (main_value != main_value), (bar, name)
            if p[name] == p[name]:
                assert p[name] == main_value, (bar, name)
        assert p['count_a'] == p['count_b'] == bar + 1
        # One instance for the loop site: three steps a bar
        assert p['loop_total'] == 9 * bar + 6
        # A loop site rolls the method's builtins back to bar start on every
        # iteration, however the method is spelled
        for name in ('loop_instance', 'loop_class'):
            assert (p[name] != p[name]) == (p['loop_main'] != p['loop_main']), (bar, name)
            if p[name] == p[name]:
                assert p[name] == p['loop_main'], (bar, name)
