"""
Unit tests for the slot-based pine method dispatch (core/pine_method.py).

``method_call`` sites are not transformed, so the dispatch binds the
runtime-selected method through a module-lifetime per-method cache: one
shared instance per method function, surviving function-object re-creation
(state from the cache, closure from the passed object) and cleared by
``instance_state.reset()`` — the formalized legacy empty-scope semantics.
"""
import sys
import types

import pytest

from pynecore.core.instance_state import reset
from pynecore.core.pine_export import Exported
from pynecore.core.pine_method import method_call

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
    """ A local method keeps ONE shared instance: state persists across
    calls and across receivers (the legacy empty-scope semantics) """
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


def __test_closure_converted_method__():
    """ A closure-converted method needs no reordering: the closure transform
    prepends the closure parameters and inserts the arguments before the
    receiver, so the positional order lines up as-is """
    def m(c1, c2, var, x):
        return (c1, c2, type(var).__name__, x)
    m.__module__ = 'fake_t017'
    m.__qualname__ = 'closure_conv.m'
    result = method_call(m, 'c1', 'c2', _Receiver(), 7.0)
    assert result == ('c1', 'c2', '_Receiver', 7.0)


def __test_dispatcher_method__():
    """ A method publishing __pyne_bind__ (an overload dispatcher) gets one
    shared anchored entry from its factory """
    calls = []

    def make_bound():
        calls.append(True)
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
