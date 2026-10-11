"""
Per-call-site method state through ``method_call``.

TradingView gives every call site of a method its own ``var`` instance, and a
call path through a function or another method its own one too; a loop site is
ONE instance shared by its iterations. Measured on CAPITALCOM:EURUSD 60 with
methods called on variable receivers, on call results (``getObj().m(x)``,
``arr.get(0).m(x)``) and through a wrapper function: every probe plot matched
bar for bar once the ``method_call`` sites were anchored, and none matched
while they shared one module-lifetime instance.

These run the slot mini pipeline on sources shaped like PyneComp's output (a
``@method`` nested in ``main``, reached through ``method_call``), so the anchor
emission and the runtime dispatch are exercised together.
"""
import ast

from pynecore.core.instance_state import (RootChildSnapshot, _make_state, create_root,
                                          discard_root)
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


def __test_helper_transform(source: str) -> tuple[dict, str]:
    """Run the slot mini pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    # A module name: an overload group registers under the defining module's
    ns: dict = {'__name__': 'test_131_method_call_site_state_src'}
    exec(compile(tree, '<method-call-site-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


__test_helper_METHOD = '''
from pynecore.core.pine_method import method, method_call
from pynecore.types import Persistent


class Obj:
    pass


def get_obj():
    return Obj()

'''

# The counter and the first-argument latch of the TradingView probe: a shared
# instance shows up in both (the count runs on, the latch keeps another site's x)
__test_helper_COUNTER = '''
    @method
    def m(self, x: float):
        cnt: Persistent[int] = 0
        first: Persistent[float] = x
        cnt += 1
        return cnt * 1000 + first
'''


def __test_sites_keep_their_own_state__():
    """ Two ``method_call`` sites of one method count independently, across bars """
    ns, dump = __test_helper_transform(__test_helper_METHOD + '''
def main():
''' + __test_helper_COUNTER + '''
    a = method_call(m, get_obj(), 1.0)
    b = method_call(m, get_obj(), 2.0)
    return a, b
''')
    assert '__bind_any·__(__state·main__, 0, method_call)' in dump
    assert '__bind_any·__(__state·main__, 1, method_call)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1001.0, 1002.0)
    assert ns['main'](state) == (2001.0, 2002.0)  # m is a new object every bar
    assert ns['main'](state) == (3001.0, 3002.0)


def __test_call_path_through_a_function__():
    """ One site inside a function, reached from two call sites of that
    function, is two instances: the anchor lives in the function's instance """
    ns, _ = __test_helper_transform(__test_helper_METHOD + '''
def main():
''' + __test_helper_COUNTER + '''
    def wrap(x: float):
        return method_call(m, get_obj(), x)
    return wrap(5.0), wrap(6.0)
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1005.0, 1006.0)
    assert ns['main'](state) == (2005.0, 2006.0)


def __test_loop_site_is_one_instance__():
    """ A site in a loop is ONE instance for all of its iterations """
    ns, dump = __test_helper_transform(__test_helper_METHOD + '''
def main():
''' + __test_helper_COUNTER + '''
    last = 0.0
    for i in range(3):
        last = method_call(m, get_obj(), 11.0 + i)
    return last
''')
    assert '__bind_loop·__(__state·main__, 0, method_call)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 3011.0  # the latch keeps the FIRST iteration's x
    assert ns['main'](state) == 6011.0


def __test_discarded_pass_rolls_method_state_back__():
    """ A discarded re-execution (calc_on_order_fills, a live tick) rolls the
    method instances of the anchored sites back to bar start -- they must not
    advance, and must not be dropped either """
    ns, _ = __test_helper_transform(__test_helper_METHOD + '''
def main():
''' + __test_helper_COUNTER + '''
    return method_call(m, get_obj(), 1.0), method_call(m, get_obj(), 2.0)
''')
    key = 'test_131_discarded_pass'
    root = create_root(key, ns['__pyne_slot_layout__']['main'])
    try:
        snapshot = RootChildSnapshot([key])
        assert ns['main'](root) == (1001.0, 1002.0)
        snapshot.save()
        assert ns['main'](root) == (2001.0, 2002.0)  # the discarded pass
        snapshot.restore()
        assert ns['main'](root) == (2001.0, 2002.0)  # the committed pass
        assert ns['main'](root) == (3001.0, 3002.0)
    finally:
        discard_root(key)


def __test_overloaded_method_per_site_and_rollback__():
    """ An overloaded method bound through ``method_call`` is its group's own
    dispatcher inside the site's cache: per site, and rolled back through it """
    ns, _ = __test_helper_transform(__test_helper_METHOD + '''
from pynecore.core.overload import overload


def main():
    @method
    @overload
    def m(self, x: float):
        cnt: Persistent[int] = 0
        cnt += 1
        return cnt * 1000 + x

    @method
    @overload
    def m(self, x: str):
        cnt: Persistent[int] = 0
        cnt += 1
        return cnt * 1000

    return method_call(m, get_obj(), 1.0), method_call(m, get_obj(), 2.0)
''')
    key = 'test_131_overloaded'
    root = create_root(key, ns['__pyne_slot_layout__']['main'])
    try:
        snapshot = RootChildSnapshot([key])
        assert ns['main'](root) == (1001.0, 1002.0)
        snapshot.save()
        assert ns['main'](root) == (2001.0, 2002.0)
        snapshot.restore()
        assert ns['main'](root) == (2001.0, 2002.0)
    finally:
        discard_root(key)


def __test_instance_born_in_discarded_pass_keeps_varip__():
    """ An instance the discarded pass CREATES rolls back to its creation state,
    but keeps its varip advance -- TradingView, calc_on_order_fills, a function
    first called in the fill pass: varip 2 and var 1 on the committed pass.
    Covers a fast-path child, a ``method_call`` anchor and an anchored callee """
    ns, _ = __test_helper_transform(__test_helper_METHOD + '''
from pynecore.types import IBPersistent


def main(active):
    def fvip():
        n: IBPersistent[int] = 0
        n += 1
        return n

    def fvar():
        n: Persistent[int] = 0
        n += 1
        return n

    @method
    def mvip(self):
        n: IBPersistent[int] = 0
        n += 1
        return n

    if not active:
        return None
    return fvip(), fvar(), method_call(mvip, get_obj())
''')
    key = 'test_131_born_in_discarded_pass'
    root = create_root(key, ns['__pyne_slot_layout__']['main'])
    try:
        snapshot = RootChildSnapshot([key])
        assert ns['main'](root, False) is None
        snapshot.save()
        assert ns['main'](root, True) == (1, 1, 1)  # the discarded pass
        snapshot.restore()
        assert ns['main'](root, True) == (2, 1, 2)  # varip kept, var rolled back
        assert ns['main'](root, True) == (3, 2, 3)  # the next bar
    finally:
        discard_root(key)


def __test_method_first_reached_in_discarded_pass_keeps_varip__():
    """ A method a ``method_call`` site reaches for the first time in the
    discarded pass is a new entry of the site's existing cache: rewound to its
    creation state like any new instance, its varip kept """
    ns, _ = __test_helper_transform(__test_helper_METHOD + '''
from pynecore.types import IBPersistent


def main(second):
    @method
    def first_method(self):
        n: IBPersistent[int] = 0
        n += 1
        return n

    @method
    def second_method(self):
        n: IBPersistent[int] = 0
        v: Persistent[int] = 0
        n += 1
        v += 1
        return n * 10 + v

    picked = second_method if second else first_method
    return method_call(picked, get_obj())
''')
    key = 'test_131_first_reached'
    root = create_root(key, ns['__pyne_slot_layout__']['main'])
    try:
        snapshot = RootChildSnapshot([key])
        assert ns['main'](root, False) == 1
        snapshot.save()
        assert ns['main'](root, True) == 11  # the discarded pass
        snapshot.restore()
        assert ns['main'](root, True) == 21  # varip 2, var 1
    finally:
        discard_root(key)
