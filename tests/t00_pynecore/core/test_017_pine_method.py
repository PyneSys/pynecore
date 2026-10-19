"""
Unit tests for the slot-based pine method dispatch (core/pine_method.py).

A transformed ``method_call`` site is anchored and gets its own dispatch entry
from ``method_call.__pyne_bind__``, with its own per-method cache: one instance
per method per CALL SITE. These tests call ``method_call`` itself, which is the
anchorless fallback: one module-lifetime cache shared by every direct call,
surviving function-object re-creation (state from the cache, closure from the
passed object) and cleared by ``instance_state.reset()``. The per-site entries
are exercised directly through the factory at the end.
"""
import sys
import types

import pytest

from pynecore.core.instance_state import reset
from pynecore.core.pine_export import Exported
from pynecore.core.pine_method import method_call
from pynecore.types.na import NA

LAYOUT = {
    'init': (0.0,),
    'series': (),
    'varip': (),
    'children': (),
    'names': ('total',),
}


class _Receiver:
    """Plain receiver object — not a Pine builtin type, has no methods."""


def _make_method(qualname: str, tag: str = ''):
    """Create a state-carrying method the way the transformer would emit it.

    :param qualname: Qualified name controlling the dispatch cache key.
    :param tag: Closure value, to observe WHICH function object ran.
    """
    def acc(__state__, var, x):
        __state__[0] += x
        return (__state__[0], tag) if tag else __state__[0]
    acc.__module__ = 'fake_t017'
    acc.__qualname__ = qualname
    acc.__pyne_layout__ = LAYOUT
    return acc


def __test_builtin_string_method__():
    """ A string method on a builtin type dispatches to the lib module """
    xs = [1.0]
    method_call('push', xs, 2.0)
    assert xs == [1.0, 2.0]


def __test_local_method_shared_state__():
    """ Without an anchor a local method keeps ONE shared instance: state
    persists across direct calls and across receivers """
    acc = _make_method('shared_state.m')
    assert method_call(acc, _Receiver(), 1.0) == 1.0
    assert method_call(acc, _Receiver(), 2.0) == 3.0  # other receiver, same instance


def __test_stateless_method_raw__():
    """ A method without a layout is called raw, no hidden parameter """
    def plain(var, x):
        return x * 2
    plain.__module__ = 'fake_t017'
    plain.__qualname__ = 'stateless.m'
    assert method_call(plain, _Receiver(), 5.0) == 10.0


def __test_reset_clears_method_state__():
    """ instance_state.reset() drops the per-method instances """
    acc = _make_method('reset_test.m')
    assert method_call(acc, _Receiver(), 1.0) == 1.0
    reset()
    assert method_call(acc, _Receiver(), 5.0) == 5.0  # fresh state


def __test_recreated_method_keeps_state__():
    """ A re-created function object (a method defined inside main) keeps
    the cached state but runs with its own fresh closure """
    first = _make_method('recreated.m', tag='first')
    assert method_call(first, _Receiver(), 1.0) == (1.0, 'first')
    second = _make_method('recreated.m', tag='second')
    # identity miss, same qualified name: state survives, the new closure runs
    assert method_call(second, _Receiver(), 2.0) == (3.0, 'second')


def __test_exported_method_via_receiver_module__():
    """ A string method resolves to an Exported proxy in the module that
    defines the receiver's UDT class; rebinding the proxy keeps the state """
    mod = types.ModuleType('fake_udt_t017')
    exec('class Point:\n    pass', mod.__dict__)
    exported = Exported()
    exported.set(_make_method('exported.m'))
    mod.mymethod = exported
    sys.modules['fake_udt_t017'] = mod
    try:
        point = mod.Point()
        assert method_call('mymethod', point, 2.0) == 2.0
        assert method_call('mymethod', point, 3.0) == 5.0
        # a re-exported (re-created) function continues the same state
        exported.set(_make_method('exported.m'))
        assert method_call('mymethod', point, 1.0) == 6.0
    finally:
        del sys.modules['fake_udt_t017']


def __test_exported_method_on_na_receiver__(monkeypatch):
    """ A string method on an na of a UDT reaches the export of the module that
    defines the DECLARED type, instead of na answering the call with itself """
    mod = types.ModuleType('fake_na_udt_t017')
    exec('class Point:\n    pass', mod.__dict__)
    exported = Exported()

    def tag(this):
        return 'handled', this
    tag.__module__ = 'fake_na_udt_t017'
    tag.__qualname__ = 'tag'
    exported.set(tag)
    mod.tag = exported
    monkeypatch.setitem(sys.modules, 'fake_na_udt_t017', mod)
    receiver = NA(mod.Point)
    assert method_call('tag', receiver) == ('handled', receiver)


@pytest.mark.parametrize('receiver', [NA(None), NA(_Receiver)])
def __test_unanswered_method_on_na_receiver_is_noop__(receiver):
    """ A string method no export answers on an untyped na, or on an na whose
    declared type has no such method, is a no-op returning the na """
    assert method_call('set_top', receiver, 1.0) is receiver
    assert method_call('delete', receiver) is receiver


@pytest.mark.parametrize('name', ['replace', 'format', 'split'])
def __test_exported_string_method_wins_over_python_str__(name, monkeypatch):
    """ A library method on a ``string`` receiver named like a Python ``str``
    method reaches the export, never ``str.replace`` / ``str.format`` / ... """
    lib_mod = types.ModuleType('lib.fake_strings_t017')
    exported = Exported()

    def impl(this, mapping):
        return this + '|' + ','.join(sorted(mapping))
    impl.__module__ = 'lib.fake_strings_t017'
    impl.__qualname__ = name
    exported.set(impl)
    setattr(lib_mod, name, exported)
    monkeypatch.setitem(globals(), '_fake_strings_t017', lib_mod)
    assert method_call(name, 'a b', {'x': 1.0, 'y': 2.0}) == 'a b|x,y'


def __test_closure_converted_method__():
    """ A closure-converted method needs no reordering: the closure transform
    declares the closure parameters right after the receiver and passes the
    arguments there, so the positional order lines up as-is """
    def m(var, c1, c2, x):
        return type(var).__name__, c1, c2, x
    m.__module__ = 'fake_t017'
    m.__qualname__ = 'closure_conv.m'
    result = method_call(m, _Receiver(), 'c1', 'c2', 7.0)
    assert result == ('_Receiver', 'c1', 'c2', 7.0)


def __test_closure_capture_is_not_the_receiver__():
    """ A builtin-named method whose first capture is a builtin container still
    runs the user method: the builtin lookup sees the RECEIVER, which the
    closure transform keeps in front of the captures """
    def size(var, captured, x):
        return var + len(captured) + x
    size.__module__ = 'fake_t017'
    size.__qualname__ = 'closure_builtin_name.size'
    assert method_call(size, 3.0, [1.0, 2.0], 10.0) == 15.0


def __test_builtin_winning_drops_closure_captures__():
    """ A builtin that wins the dispatch of a closure-converted method takes the
    visible arguments only: the captures the method publishes as
    ``__pyne_closure__`` are dropped after the receiver, a visible keyword named
    like a capture stays the builtin's, and the captures of an unpacked-receiver
    site (``__pyne_captures__``) never reach the builtin """
    def get(var, index, i):
        return var[i + index]
    get.__module__ = 'fake_t017'
    get.__qualname__ = 'closure_builtin_wins.get'
    setattr(get, '__pyne_closure__', ('index',))
    values = [1.0, 2.0]
    assert method_call(get, values, 1, 0) == 1.0  # array.get(values, 0)
    assert method_call(get, values, 1, index=0) == 1.0  # array.get(values, index=0)
    assert method_call(get, *[values], 0, __pyne_captures__=(1,)) == 1.0


def __test_unpacked_receiver_captures_follow_the_receiver__():
    """ The captures of an unpacked-receiver site reach the method right after
    its receiver, where the closure transform declares them """
    receiver = _Receiver()

    def shift(var, offset, x):
        assert var is receiver
        return x + offset
    shift.__module__ = 'fake_t017'
    shift.__qualname__ = 'closure_unpacked.shift'
    setattr(shift, '__pyne_closure__', ('offset',))
    assert method_call(shift, *[receiver], 1.0, __pyne_captures__=(10.0,)) == 11.0


def __test_dispatcher_method__():
    """ A method publishing __pyne_bind__ (an overload dispatcher) gets one
    shared anchored entry from its factory """
    calls = []

    def make_bound(pin=None):
        calls.append(pin)
        def bound(var, x):
            return x + 1
        return bound

    def dispatcher(var, x):
        raise AssertionError('the dispatch must call the factory binding')
    dispatcher.__module__ = 'fake_t017'
    dispatcher.__qualname__ = 'dispatcher.m'
    dispatcher.__pyne_bind__ = make_bound

    assert method_call(dispatcher, _Receiver(), 1.0) == 2.0
    assert method_call(dispatcher, _Receiver(), 2.0) == 3.0
    assert len(calls) == 1  # bound once, cached for the module lifetime


def __test_builtin_array_method__():
    """ A string method on a plain list dispatches to the array namespace """
    xs = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert method_call('max', xs) == 5.0
    assert method_call('min', xs) == 1.0


def __test_sequence_view_method_dispatch__():
    """ array.slice() returns a SequenceView; method calls on the view
    (slice(...).max() / .min() / .size()) dispatch to the array namespace,
    not just plain lists """
    from pynecore.lib import array
    view = array.slice([3.0, 1.0, 4.0, 1.0, 5.0], 1, 4)  # -> [1.0, 4.0, 1.0]
    assert method_call('max', view) == 4.0
    assert method_call('min', view) == 1.0
    assert method_call('size', view) == 3


def __test_no_such_method_asserts__():
    """ An unresolvable string method fails loudly """
    with pytest.raises(AssertionError, match='No such method'):
        method_call('definitely_missing', _Receiver())


def __test_anchored_entries_keep_state_per_site__():
    """ Every anchor gets its own dispatch entry with its own method instances:
    two sites calling the same method count independently, and each keeps its
    state across a re-created function object """
    site_a = method_call.__pyne_bind__(None)
    site_b = method_call.__pyne_bind__(None)
    assert site_a is not site_b and site_a is not method_call
    acc = _make_method('per_site.m')
    assert site_a(acc, _Receiver(), 1.0) == 1.0
    assert site_b(acc, _Receiver(), 10.0) == 10.0  # its own instance
    assert site_a(acc, _Receiver(), 2.0) == 3.0
    again = _make_method('per_site.m')  # main() re-ran: a new function object
    assert site_b(again, _Receiver(), 5.0) == 15.0
    assert site_a(again, _Receiver(), 1.0) == 4.0
    assert method_call(acc, _Receiver(), 7.0) == 7.0  # the fallback is a third instance


def __test_anchored_entry_exposes_its_cache__():
    """ An anchored entry publishes its method states as ``__pyne_cache__``,
    the shape the loop-site rollback and the re-execution snapshot walk """
    site = method_call.__pyne_bind__(None)
    acc = _make_method('cache_shape.m')
    site(acc, _Receiver(), 2.0)
    entries = list(site.__pyne_cache__.values())
    assert len(entries) == 1
    target, state, _bound = entries[0]
    assert target is acc and state[0] == 2.0


def __test_anchored_entries_exported_method_per_site__():
    """ A string-dispatched library method is bound per anchor too """
    mod = types.ModuleType('fake_udt_t017_sites')
    exec('class Point:\n    pass', mod.__dict__)
    exported = Exported()
    exported.set(_make_method('exported_sites.m'))
    mod.mymethod = exported
    sys.modules['fake_udt_t017_sites'] = mod
    try:
        point = mod.Point()
        site_a = method_call.__pyne_bind__(None)
        site_b = method_call.__pyne_bind__(None)
        assert site_a('mymethod', point, 2.0) == 2.0
        assert site_b('mymethod', point, 3.0) == 3.0
        assert site_a('mymethod', point, 1.0) == 3.0
    finally:
        del sys.modules['fake_udt_t017_sites']
