"""
Pine method dispatch on the slot-based instance-state scheme.

A method's state belongs to the CALL SITE, like any other function's:
TradingView gives every call site of a method its own ``var`` instance, and a
call path through a function or another method its own one too (measured on
receivers that are variables, call results and array elements alike). A
``method_call`` site is therefore anchored like an overload dispatcher site:
``method_call`` publishes a ``__pyne_bind__`` factory, the isolation transform
routes its call sites uniform, and every anchor receives its own dispatch
entry with its own per-method bound cache. The method is still selected at run
time, from the receiver; what that entry adds is where its state lives.

Calling ``method_call`` itself (no anchor: module level, untransformed code)
falls back to one module-lifetime cache shared by every such call, cleared by
``instance_state.reset()`` -- the same fallback an overload dispatcher has.
"""
import sys
from types import ModuleType
from typing import Any, Callable
from functools import partial

from ..lib import array, matrix, box, line, label, table, linefill, polyline
from ..lib import map as map_lib
from ..utils.sequence_view import SequenceView
from ..types import matrix as matrix_types
from ..types import line as line_types
from ..types import box as box_types
from ..types import label as label_types
from ..types import table as table_types
from ..types import linefill as linefill_types
from ..types import polyline as polyline_types
from ..types.na import NA

from .instance_state import _configure, _make_state, register_shared_cache  # noqa: internal API
from .pine_export import Exported, in_module_bool_mode


#: Runtime classes of Pine scalar receivers
_PINE_SCALAR_TYPES: frozenset[type] = frozenset({str, int, float, bool})


def method(func: Callable) -> Callable:
    """
    Decorator to mark a function as a Pine method.
    This is used to indicate that the function should be treated as a method in Pine Script.
    """
    setattr(func, '__pine_method__', True)
    return func


def _get_builtin_method(method_name: str, var: Any) -> Callable | None:
    """
    Get the built-in method for a Pine Script object.
    :param method_name: The name of the method
    :param var: The object on which the method is being called
    :return: The built-in method, or None if not found
    """
    try:
        var_type = type(var)
        if var_type is NA:
            # An na value carries its DECLARED type. A builtin method invoked on
            # an na builtin object (e.g. ``naLine.delete()`` on a line that was
            # never drawn) must dispatch to that builtin namespace, which treats
            # na as a no-op the way Pine does. An na of a UDT (or untyped) type
            # matches nothing here and falls through to user-method dispatch.
            declared = var.type
            if declared is None:
                return None
            var_type = declared if isinstance(declared, type) else type(declared)
        match var_type:
            case _ if var_type is list:
                return getattr(array, method_name)
            case _ if var_type is SequenceView:
                # array.slice() returns a view; method calls on it (e.g.
                # slice(...).max()) dispatch to the array namespace too.
                return getattr(array, method_name)
            case _ if var_type is dict:
                return getattr(map_lib, method_name)
            case _ if var_type is matrix_types.Matrix:
                return getattr(matrix, method_name)
            case _ if var_type is line_types.Line:
                return getattr(line, method_name)
            case _ if var_type is box_types.Box:
                return getattr(box, method_name)
            case _ if var_type is label_types.Label:
                return getattr(label, method_name)
            case _ if var_type is table_types.Table:
                return getattr(table, method_name)
            case _ if var_type is linefill_types.LineFill:
                return getattr(linefill, method_name)
            case _ if var_type is polyline_types.Polyline:
                return getattr(polyline, method_name)
            case _:
                return None
    except AttributeError:
        pass

    return None


# The bound cache of the anchorless fallback (see the module docstring). Method
# key -> (function identity, state vector or None, bound callable). A
# per-function key is essential: a constant key would make every method share
# one cache entry, running one method with another method's state. Registered
# so instance_state.reset() clears it between runs.
_method_anchors: dict[str, tuple[Any, list | None, Callable]] = register_shared_cache({})


# noinspection PyShadowingNames
def _bound_method(method: Any, cache: dict[str, tuple[Any, list | None, Callable]],
                  pin: str | None = None, vector: tuple | None = None) -> Callable:
    """Bound instance of a runtime-dispatched method, kept in a dispatch cache.

    The cache belongs to one call site (an anchored entry, see
    :func:`_dispatcher`), or to every anchorless call at once. Its key is the
    method's qualified name and the entry survives function-object
    re-creation: a method defined inside ``main()`` is a fresh object every
    invocation, but its persistent state must live on — on an identity miss
    the entry's state vector is rebound to the new object (whose closure cells
    are the live ones). State from the cache, closure from the passed object.

    :param method: The method callable, or an ``Exported`` proxy of one.
    :param cache: The dispatch cache the bound instance lives in.
    :param pin: The call site's overload pin, for a method that is an overload
        group: the group's anchored entry selects its implementation with it.
    :param vector: The call site's instance vector, written into the state of a
        method created here (see ``instance_state._configure``).
    :return: The callable to invoke with the visible arguments.
    :raises ValueError: If an ``Exported`` proxy is not initialized yet.
    """
    target: Callable = method
    na_bool: bool | None = None
    if isinstance(method, Exported):
        na_bool = method.__na_bool__
        unwrapped: Callable | None = method.__fn__
        if unwrapped is None:
            raise ValueError("Exported proxy has not been initialized with a function yet")
        target = unwrapped
    key = f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', '?')}"
    entry = cache.get(key)
    if entry is not None and entry[0] is target:
        return entry[2]
    # A method of another module runs that module's bool semantics
    # (see pine_export.in_module_bool_mode)
    if isinstance(target, type) or (
            hasattr(target, '__self__') and isinstance(target.__self__, type)):
        return target  # types and classmethods are called as-is (legacy guard)
    state: list | None = None
    if hasattr(target, '__pyne_bind__'):
        bound: Callable = target.__pyne_bind__(pin)  # overload dispatcher: its own anchored entry
    else:
        layout: dict[str, Any] | None = getattr(target, '__pyne_layout__', None)
        if layout is not None:
            state = entry[1] if entry is not None and entry[1] is not None \
                else _configure(_make_state(layout), layout, vector)
            bound = partial(target, state)
        else:
            bound = target  # stateless — called raw
    if na_bool is not None:
        bound = in_module_bool_mode(bound, na_bool)
    cache[key] = (target, state, bound)
    return bound


def _adapt_exported_kwargs(exported: Exported, kwargs: dict) -> dict:
    """
    Map keyword arguments onto a compiled library's canonically renamed
    parameter names when needed.

    A compiler cannot always know at emission time that a string-dispatched
    method call targets a library export (the receiver may be untyped), so a
    keyword argument can arrive under its original Pine spelling while the
    library ``def`` declares ``name + '__ren__'``. The rename is applied only
    when the raw name is absent from the target signature and the suffixed
    one is present — a correct call is never altered.

    :param exported: The resolved Exported proxy
    :param kwargs: Keyword arguments as emitted at the call site
    :return: Keyword arguments matching the target's parameter names
    """
    if not kwargs:
        return kwargs
    code: Any = getattr(exported.__fn__, '__code__', None)
    if code is None:
        return kwargs
    params = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    if all(k in params for k in kwargs):
        return kwargs
    return {(k + '__ren__' if k not in params and k + '__ren__' in params else k): v
            for k, v in kwargs.items()}


def _dispatcher(cache: dict[str, tuple[Any, list | None, Callable]],
                pin: str | None = None, vector: tuple | None = None) -> Callable[..., Any]:
    """Create a ``method_call`` dispatch entry bound to one dispatch cache.

    The module-level :func:`method_call` is the entry of the anchorless
    fallback cache; ``method_call.__pyne_bind__`` hands out a fresh entry with
    a cache of its own to every anchor, which is what gives each call site its
    own method instances. The cache is exposed as ``__pyne_cache__`` (the same
    shape an overload dispatcher's has), so the loop-site same-bar rollback and
    the discarded re-execution snapshot walk the method states in it.

    :param cache: The dispatch cache the entry binds methods in.
    :param pin: The call site's overload pin, which the type pass stamps where it
        resolved the method to an overload group of the calling module: it
        selects from the static argument types, like a direct call's pin does.
        It reaches only the method handed over as a function -- the one the
        type pass resolved; a builtin the receiver reaches first ignores it.
    :param vector: The call site's instance vector, which the type pass stamps
        where it resolved the method to a generic function of the calling module
        whose instance varies with its caller (an implicit conversion, a pin of
        a call inside it). Like the pin it configures only the method handed
        over as a function.
    :return: The dispatch entry.
    """

    # noinspection PyShadowingNames
    def method_call(method: str | Callable, var: Any, *args,
                    __pyne_captures__: tuple | None = None, **kwargs) -> Any:
        """
        Dispatch a method call on a Pine Script variable to the appropriate handler.

        This function serves as the central dispatcher for Pine Script method calls, handling
        both built-in type methods (like array and matrix operations) and user-defined local
        methods. It provides the Pine Script-like method calling syntax by routing calls to the
        correct implementation based on the variable type and method name.

        Closure-converted methods: the closure transform declares a method's
        closure parameters right after its receiver and passes the matching
        arguments right after the receiver at the call site, so ``var`` is always
        the receiver the builtin lookup needs, and the plain positional order
        lines up for the method itself. The method publishes their names as
        ``__pyne_closure__``, and a builtin that wins the lookup gets the
        arguments without that many leading ones. An unpacked receiver leaves no
        position to follow: such a call site hands the captures over in
        ``__pyne_captures__`` instead, which only the method receives -- after
        its receiver -- so no visible argument is ever mistaken for one.

        :param method: The method to call, either as a string name (for built-in methods) or
                       a callable (for local methods)
        :param var: The object/variable on which the method is being called (e.g., array,
                    matrix, or custom object)
        :param args: Positional arguments to pass to the method
        :param __pyne_captures__: A closure-converted method's captures, from a call
                                  site whose receiver is unpacked
        :param kwargs: Keyword arguments to pass to the method
        :return: The result of the method call, or None if the method cannot be dispatched
        :raises AssertionError: If a string method name is provided but no matching method is
                                found for the variable type
        """
        # If method is a string
        if isinstance(method, str):
            # Support for builtin methods
            _method = _get_builtin_method(method, var)
            if _method is not None:
                return _method(var, *args, **kwargs)

            # Modules. A Pine scalar has no builtin methods, so a Python ``str`` /
            # number attribute (``str.replace``, ``str.format``, ...) must never
            # answer a library method of the same name. An na receiver answers every
            # attribute with itself, so it would swallow the call: its exported method
            # is looked up by its DECLARED type instead.
            receiver_type = type(var)
            if receiver_type is NA:
                declared = var.type
                receiver_type = declared if isinstance(declared, type) else type(declared)
            elif receiver_type not in _PINE_SCALAR_TYPES:
                try:
                    return getattr(var, method)(*args, **kwargs)
                except AttributeError:
                    pass

            # Methods exported by a compiled library are module-level Exported proxies.
            # Try the module that defines the receiver's UDT class first, then the
            # caller's own module, then every library module the caller imports -- a
            # library can export methods on another library's UDT, so the defining
            # module is not always the UDT's own.
            # If the raw name misses everywhere, retry once with PyneComp's canonical
            # rename suffix: a compiled library emits a method whose Pine name cannot
            # live verbatim in Python (``lambda``, ``method_call``, ...) under
            # ``name + '__ren__'``. A stub with that shape can only be the image of
            # this Pine name (a Pine ``x__ren__`` compiles to ``x__ren____ren__``),
            # so the retry can never hit a wrong export.
            caller_globals = sys._getframe(1).f_globals
            mod = sys.modules.get(receiver_type.__module__)
            for lookup_name in (method, method + '__ren__'):
                _method = getattr(mod, lookup_name, None) if mod is not None else None
                if not isinstance(_method, Exported):
                    _method = caller_globals.get(lookup_name)
                if not isinstance(_method, Exported):
                    for _gval in caller_globals.values():
                        if isinstance(_gval, ModuleType) and _gval.__name__.startswith('lib.'):
                            _method = getattr(_gval, lookup_name, None)
                            if isinstance(_method, Exported):
                                break
                if isinstance(_method, Exported):
                    return _bound_method(_method, cache)(
                        var, *args, **_adapt_exported_kwargs(_method, kwargs))

            assert False, f'No such method: {var}->{method}'

        # It is a local method, it should be a local function
        elif callable(method):
            # It may not detected well the type and there may be a user with the same method
            # name. So we 1st trt if it is a built-in object and has that method, because it
            # has priority
            _method = _get_builtin_method(method.__name__, var)
            if _method:
                if __pyne_captures__ is None:
                    captures: tuple[str, ...] | None = getattr(method, '__pyne_closure__', None)
                    if captures is not None:
                        args = args[len(captures):]
                return _method(var, *args, **kwargs)

            if __pyne_captures__ is not None:
                args = __pyne_captures__ + args
            return _bound_method(method, cache, pin, vector)(var, *args, **kwargs)

        return None

    setattr(method_call, '__pyne_cache__', cache)
    return method_call


method_call = _dispatcher(_method_anchors)
# Per-anchor entries for the call sites the isolation transform anchors
setattr(method_call, '__pyne_bind__', lambda pin=None, vector=None: _dispatcher({}, pin, vector))
