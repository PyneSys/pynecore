from typing import Callable, TypeVar, Generic, Optional, Any, Union, overload
import sys

from ..types import na as _na

__all__ = ['Exported', 'export', 'in_module_bool_mode']

#: Name the loader bakes a module's bool na choice under (import_hook)
_NA_BOOL = '__pyne_na_bool__'


def in_module_bool_mode(bound: Callable, na_bool: bool) -> Callable:
    """Wrap a library's bound callable so it runs its OWN bool na semantics.

    The three-state bool of a v4/v5 library is a property of its source, not of
    whoever calls it: a v6 caller must not flatten the na its body builds, and a
    v4/v5 caller must not give one to a v6 library. The mode is process-wide, so
    the crossing swaps it for the duration of the call and puts it back.

    While no module in the process has asked for the three-state bool there is
    nothing to keep apart, and the caller is handed the bare callable.

    :param bound: The callable the binding resolved to.
    :param na_bool: The defining module's bool na choice.
    :return: The callable to invoke.
    """
    if not _na._bool_na_seen:
        return bound

    def call(*args, **kwargs) -> Any:
        if na_bool is _na._bool_na:
            return bound(*args, **kwargs)
        _na.set_bool_na(na_bool)
        try:
            return bound(*args, **kwargs)
        finally:
            _na.set_bool_na(not na_bool)

    return call

F = TypeVar('F', bound=Callable[..., Any])  # Function type


class Exported(Generic[F]):
    """
    Function closure proxy with flexible type annotation support

    Supports:
    - Protocol with named parameters: Exported[MyProtocol]
    - Callable types: Exported[Callable[[int, str], bool]]
    - No annotation: Exported (falls back to Any)
    """
    __fn__: Optional[F] = None
    __name__: str
    #: Bool na choice of the module the function was defined in
    __na_bool__: bool = False

    def set(self, client: F, na_bool: bool = False):
        """Set the client function

        :param client: The function the proxy stands for.
        :param na_bool: The defining module's bool na choice.
        """
        self.__fn__ = client
        self.__na_bool__ = na_bool
        # Expose the client's name so callers that inspect the callable
        # (e.g. method_call's builtin-method name check) see the real one
        name: str | None = getattr(client, '__name__', None)
        if name is not None:
            self.__name__ = name

    def __call__(self, *args, **kwargs) -> Any:
        fn = self.__fn__
        if fn is None:
            raise ValueError("Function has not been set yet")
        # A library carries its own bool semantics across the export boundary:
        # the three-state bool of a v4/v5 library is a property of ITS source,
        # and a v6 caller must not flatten the na its body builds (nor the
        # other way round). The mode is process-wide, so it is swapped for the
        # duration of the call and put back afterwards
        na_bool = self.__na_bool__
        if na_bool is _na._bool_na or not _na._bool_na_seen:
            return fn(*args, **kwargs)
        _na.set_bool_na(na_bool)
        try:
            return fn(*args, **kwargs)
        finally:
            _na.set_bool_na(not na_bool)


@overload
def export(func: Callable) -> Callable:
    ...


@overload
def export(*, func_globals: dict[str, Any]) -> Callable:
    ...


def export(
        func: Optional[Callable] = None,
        *,
        func_globals: Optional[dict[str, Any]] = None
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Export decorator that can work with or without parameters.
    It is exporting the function closure to the global scope of the module.

    Usage:
    @export
    def my_func(): pass

    @export(func_globals=some_globals)
    def my_func(): pass
    """
    # Get caller's globals once at decorator definition time
    if func_globals is None:
        func_globals = sys._getframe(1).f_globals

    def decorator(f: Callable) -> Callable:
        func_name = f.__name__
        assert func_globals is not None

        # Check if there's already something with the same name in globals
        if func_name in func_globals:
            existing = func_globals[func_name]
            if isinstance(existing, Exported):
                # Set the function in the existing proxy
                existing.set(f, bool(func_globals.get(_NA_BOOL, False)))
                return existing
            elif callable(existing):
                # Function already exists in global scope, just return it unchanged (decorator as decoration)
                return f

        # No proxy found, throw error explaining what's needed
        raise ValueError(
            f"No Exported proxy found for function '{func_name}' in global scope. "
            f"You must create an Exported proxy first:\n"
            f"  {func_name} = Exported()\n"
            f"  @export\n"
            f"  def {func_name}(): ..."
        )

    if func is not None:
        # Called without parentheses: @export
        return decorator(func)
    else:
        # Called with parentheses: @export() or @export(func_globals=...)
        return decorator
