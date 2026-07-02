"""
Pine method dispatch on the slot-based instance-state scheme.

``method_call`` is on the transformer's NON_TRANSFORMABLE list, so its call
sites stay raw and there is no caller anchor — the method is selected at
runtime and bound through a module-lifetime per-method cache instead: ONE
shared instance per method function, cleared between runs. This formalizes
what the legacy runtime did implicitly (its ``__scope_id__`` here was forever
``''``, so every isolation it requested landed on the same empty-scope cache
key, shared by all call sites of a method and dropped on ``reset()``).
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

from .instance_state import _make_state, register_shared_cache
from .pine_export import Exported


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


# Method key -> (function identity, state vector or None, bound callable).
# A per-function key is essential: a constant key would make every method
# share one cache entry, running one method with another method's state.
# Registered so instance_state.reset() clears it between runs.
_method_anchors: dict[str, tuple[Any, list | None, Callable]] = register_shared_cache({})


# noinspection PyShadowingNames
def _bound_method(method: Any) -> Callable:
    """Module-lifetime bound instance of a runtime-dispatched method.

    The cache key is the method's qualified name and the entry survives
    function-object re-creation: a method defined inside ``main()`` is a
    fresh object every invocation, but its persistent state must live on —
    on an identity miss the entry's state vector is rebound to the new
    object (whose closure cells are the live ones). State from the cache,
    closure from the passed object: the exact legacy split.

    :param method: The method callable, or an ``Exported`` proxy of one.
    :return: The callable to invoke with the visible arguments.
    :raises ValueError: If an ``Exported`` proxy is not initialized yet.
    """
    target = method
    if isinstance(target, Exported):
        target = target.__fn__
        if target is None:
            raise ValueError("Exported proxy has not been initialized with a function yet")
    key = f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', '?')}"
    entry = _method_anchors.get(key)
    if entry is not None and entry[0] is target:
        return entry[2]
    if isinstance(target, type) or (
            hasattr(target, '__self__') and isinstance(target.__self__, type)):
        return target  # types and classmethods are called as-is (legacy guard)
    state: list | None = None
    bind = getattr(target, '__pyne_bind__', None)
    if bind is not None:
        bound: Callable = bind()  # overload dispatcher: one shared anchored entry
    else:
        layout = getattr(target, '__pyne_layout__', None)
        if layout is not None:
            state = entry[1] if entry is not None and entry[1] is not None \
                else _make_state(layout)
            bound = partial(target, state)
        else:
            bound = target  # stateless — called raw
    _method_anchors[key] = (target, state, bound)
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


# noinspection PyShadowingNames
def method_call(method: str | Callable, var: Any, *args, **kwargs) -> Any:
    """
    Dispatch a method call on a Pine Script variable to the appropriate handler.

    This function serves as the central dispatcher for Pine Script method calls, handling both
    built-in type methods (like array and matrix operations) and user-defined local methods.
    It provides the Pine Script-like method calling syntax by routing calls to the correct
    implementation based on the variable type and method name.

    Closure-converted methods need no special handling: the closure transform
    prepends the closure parameters to the method's signature and inserts the
    matching arguments BEFORE the receiver at the call site, so the plain
    positional order already lines up.

    :param method: The method to call, either as a string name (for built-in methods) or a callable (for local methods)
    :param var: The object/variable on which the method is being called (e.g., array, matrix, or custom object)
    :param args: Positional arguments to pass to the method
    :param kwargs: Keyword arguments to pass to the method
    :return: The result of the method call, or None if the method cannot be dispatched
    :raises AssertionError: If a string method name is provided but no matching method is found for the variable type
    """
    # If method is a string
    if isinstance(method, str):
        # Support for builtin methods
        _method = _get_builtin_method(method, var)
        if _method is not None:
            return _method(var, *args, **kwargs)

        # Modules
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
        mod = sys.modules.get(type(var).__module__)
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
                return _bound_method(_method)(var, *args,
                                              **_adapt_exported_kwargs(_method, kwargs))

        assert False, f'No such method: {var}->{method}'

    # It is a local method, it should be a local function
    elif callable(method):
        # It may not detected well the type and there may be a user with the same method name.
        # So we 1st trt if it is a built-in object and has that method, because it has priority
        _method = _get_builtin_method(method.__name__, var)
        if _method:
            return _method(var, *args, **kwargs)

        return _bound_method(method)(var, *args, **kwargs)

    return None
