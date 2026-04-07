from typing import Callable, TypeVar, Generic, Optional, Any, Union, overload
import sys

__all__ = ['Exported', 'export']

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

    def set(self, client: F):
        """Set the client function"""
        self.__fn__ = client

    def __call__(self, *args, **kwargs) -> Any:
        if self.__fn__ is None:
            raise ValueError("Function has not been set yet")
        return self.__fn__(*args, **kwargs)


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
                existing.set(f)
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
