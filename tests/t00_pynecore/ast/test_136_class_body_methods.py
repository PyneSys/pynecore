"""
State of methods declared in a class body.

A method in the class body is Pine's ``method m(Type this)`` spelled in the
class, so its state belongs to the CALL SITE like any function's: every site
its own instance, a loop site one instance for all of its iterations. The
hidden state parameter stands first, as on every definition, and an instance
reaches the method through the ``__state_method__`` descriptor, which puts the
receiver behind the state.

These run the slot mini pipeline, so the routing and the runtime binding are
exercised together.
"""
import ast

import pytest

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
    ns: dict = {'__name__': 'test_136_class_body_methods_src'}
    exec(compile(tree, '<class-body-method-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


# A counter and a first-argument latch: a shared instance shows up in both
__test_helper_COUNTER = '''
from pynecore.core.pine_method import method_call
from pynecore.types import Persistent


def latch(x: float):
    cnt: Persistent[int] = 0
    first: Persistent[float] = x
    cnt += 1
    return cnt * 1000 + first


class Obj:
    def m(self, x: float):
        cnt: Persistent[int] = 0
        first: Persistent[float] = x
        cnt += 1
        return cnt * 1000 + first

    def through(self, x: float):
        return latch(x) + self.m(x)

    def plain(self, x: float):
        return x * 2

'''


def __test_instance_sites_keep_their_own_state__():
    """ Two sites of one method count independently, across bars """
    ns, dump = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    return obj.m(1.0), obj.m(2.0)
''')
    assert '@__state_method__' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1001.0, 1002.0)
    assert ns['main'](state) == (2001.0, 2002.0)
    assert ns['main'](state) == (3001.0, 3002.0)


def __test_a_site_keeps_its_state_whatever_the_receiver__():
    """ The state is the site's, not the receiver's: a new object every bar
    still counts on """
    ns, _ = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    return Obj().m(7.0)
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 1007.0
    assert ns['main'](state) == 2007.0


def __test_a_call_through_the_class_routes_to_the_method__():
    """ ``Obj.m(obj, x)`` reaches the function itself, which takes the state
    first: the fast route, no anchor """
    ns, dump = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    return Obj.m(obj, 3.0)
''')
    assert '__resolve_slot·__(__state__, 0, Obj.m)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 1003.0
    assert ns['main'](state) == 2003.0


def __test_a_method_calls_stateful_code_of_its_own__():
    """ A method's own sites live in the method's instance, a function and
    another method of the receiver alike """
    ns, _ = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    return obj.through(4.0), obj.through(5.0)
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (2008.0, 2010.0)
    assert ns['main'](state) == (4008.0, 4010.0)


def __test_loop_site_is_one_instance__():
    """ A site in a loop is ONE instance for all of its iterations """
    ns, dump = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    last = 0.0
    for i in range(3):
        last = obj.m(11.0 + i)
    return last
''')
    assert '__bind_loop·__(__state__, 0, obj.m)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 3011.0  # the latch keeps the FIRST iteration's x
    assert ns['main'](state) == 6011.0


def __test_method_call_by_name_keeps_per_site_state__():
    """ ``method_call('m', obj)`` finds the method on the receiver, and the
    state is still the site's """
    ns, _ = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    return method_call('m', obj, 1.0), method_call('m', obj, 2.0)
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1001.0, 1002.0)
    assert ns['main'](state) == (2001.0, 2002.0)


def __test_a_stateless_method_stays_a_plain_method__():
    """ Only a state-carrying method gets the descriptor """
    ns, dump = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    return Obj().plain(4.0)
''')
    assert 'def plain(self, x: float)' in dump
    assert ns['main'](_make_state(ns['__pyne_slot_layout__']['main'])) == 8.0


def __test_discarded_pass_rolls_method_state_back__():
    """ A discarded re-execution rolls the method instances back to bar start """
    ns, _ = __test_helper_transform(__test_helper_COUNTER + '''
def main():
    obj = Obj()
    return obj.m(1.0), obj.m(2.0)
''')
    key = 'test_136_discarded_pass'
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


def __test_a_static_method_keeps_state_without_a_receiver__():
    """ A static method has no receiver: the layout alone wires it """
    ns, _ = __test_helper_transform(__test_helper_COUNTER + '''
class Tool:
    @staticmethod
    def count(x: float):
        return latch(x)


def main():
    return Tool.count(1.0), Tool().count(2.0)
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1001.0, 1002.0)
    assert ns['main'](state) == (2001.0, 2002.0)


def __test_any_other_decorator_on_a_stateful_method_is_refused__():
    """ A decorator decides what the attribute holds, so the state could not
    be routed: refused at transform time, never miscalled at run time """
    with pytest.raises(SyntaxError, match="method 'count' keeps state"):
        __test_helper_transform(__test_helper_COUNTER + '''
class Tool:
    @classmethod
    def count(cls, x: float):
        return latch(x)
''')


def __test_positional_only_parameters_follow_the_state__():
    """ The hidden state stands first in the signature, ahead of the
    positional-only parameters too, a method's receiver and a function's
    arguments alike """
    ns, dump = __test_helper_transform(__test_helper_COUNTER + '''
def count(x: float, /, y: float = 0.0):
    return latch(x) + y


class Strict:
    def m(self, /, x: float):
        return latch(x)


def main():
    return count(1.0), Strict().m(2.0), Strict.m(Strict(), 3.0)
''')
    assert 'def count(__state__, x: float, /, y: float=0.0)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == (1001.0, 1002.0, 1003.0)
    assert ns['main'](state) == (2001.0, 2002.0, 2003.0)
