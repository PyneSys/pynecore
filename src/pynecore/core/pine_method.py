import sys
from types import ModuleType
from typing import Any, Callable
from ..lib import array, matrix, box, line, label, table, linefill, polyline
from ..lib import map as map_lib
from ..types import matrix as matrix_types
from ..types import line as line_types
from ..types import box as box_types
from ..types import label as label_types
from ..types import table as table_types
from ..types import linefill as linefill_types
from ..types import polyline as polyline_types

from .function_isolation import isolate_function
from .pine_export import Exported

__scope_id__ = ''


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


# noinspection PyShadowingNames
def method_call(method: str | Callable, var: Any, *args, _closure_vars_count: int = -1, **kwargs) -> Any:
    """
    Dispatch a method call on a Pine Script variable to the appropriate handler.

    This function serves as the central dispatcher for Pine Script method calls, handling both
    built-in type methods (like array and matrix operations) and user-defined local methods.
    It provides the Pine Script-like method calling syntax by routing calls to the correct
    implementation based on the variable type and method name.

    :param method: The method to call, either as a string name (for built-in methods) or a callable (for local methods)
    :param var: The object/variable on which the method is being called (e.g., array, matrix, or custom object)
    :param _closure_vars_count: The number of closure variables to pass to the method
    :param args: Positional arguments to pass to the method
    :param kwargs: Keyword arguments to pass to the method
    :return: The result of the method call, or None if the method cannot be dispatched
    :raises AssertionError: If a string method name is provided but no matching method is found for the variable type
    """
    global __scope_id__

    # If method is a string
    if isinstance(method, str):
        # Support for builtin methods\
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
        # module is not always the UDT's own
        mod = sys.modules.get(type(var).__module__)
        _method = getattr(mod, method, None) if mod is not None else None
        if isinstance(_method, Exported):
            return (isolate_function(_method, _method_call_id(_method), __scope_id__, _closure_vars_count)
                    (var, *args, **kwargs))
        caller_globals = sys._getframe(1).f_globals
        _method = caller_globals.get(method)
        if isinstance(_method, Exported):
            return (isolate_function(_method, _method_call_id(_method), __scope_id__, _closure_vars_count)
                    (var, *args, **kwargs))
        for _gval in caller_globals.values():
            if isinstance(_gval, ModuleType) and _gval.__name__.startswith('lib.'):
                _method = getattr(_gval, method, None)
                if isinstance(_method, Exported):
                    return (isolate_function(_method, _method_call_id(_method), __scope_id__,
                                             _closure_vars_count)(var, *args, **kwargs))

        assert False, f'No such method: {var}->{method}'

    # It is a local method, it should be a local function
    elif callable(method):
        # It may not detected well the type and there may be a user with the same method name.
        # So we 1st trt if it is a built-in object and has that method, because it has priority
        _method = _get_builtin_method(method.__name__, var)
        if _method:
            return _method(var, *args, **kwargs)

        if _closure_vars_count > 0:
            pre_args = args[-_closure_vars_count:]
            args = args[:-_closure_vars_count]
        else:
            pre_args = ()

        # If not in kwargs, check if the method has closure arguments information on the function itself
        return (isolate_function(method, _method_call_id(method), __scope_id__, _closure_vars_count)
                (*pre_args, var, *args, **kwargs))

    return None


def _method_call_id(func: Callable) -> str:
    """
    Build a per-function isolation call id. A constant id would make every method
    share one isolation cache slot, rebuilding one method's code with another
    method's (even another module's) globals.

    :param func: The method callable (possibly an Exported proxy)
    :return: The call id string
    """
    target = func.__fn__ if isinstance(func, Exported) else func
    return f"__method_call__·{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', '?')}"
