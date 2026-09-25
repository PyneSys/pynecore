"""
Static view of the Pine hybrid decorators.

A ``module_property`` name reads as the value its function returns. A
``module_function_property`` name is both at once: a Pine value of the function's
return type (``time``, ``ta.tr``) and a callable with the function's own signature
(``time("D")``, ``ta.tr(true)``). Runtime (module_property.py) only marks the function.
"""
from typing import Callable, Generic, ParamSpec, TypeVar, overload

from ..types.pine_types import PyneFloat, PyneInt

T = TypeVar('T')
_FI = TypeVar('_FI', bound=Callable[..., PyneInt])
_FF = TypeVar('_FF', bound=Callable[..., PyneFloat])
_P = ParamSpec('_P')


def module_property(func: Callable[..., T]) -> T:
    """
    Decorator for Pine-style hybrid property/functions.

    Statically typed as the wrapped function's return value: the AST
    transformer routes bare reads through the call, so user code sees a value.
    Use :func:`module_function_property` for the hybrids that user code also
    calls with arguments (``time(...)``, ``year(t)``, ``ta.tr(true)``, ...).
    """


class IntFunctionProperty(PyneInt, Generic[_FI]):
    """
    A Pine int that is also a function: ``time`` reads the value, ``time("D")`` calls it.
    """
    __call__: _FI


class FloatFunctionProperty(PyneFloat, Generic[_FF]):
    """
    A Pine float that is also a function: ``ta.tr`` reads the value, ``ta.tr(true)`` calls it.
    """
    __call__: _FF


class BandsFunctionProperty(PyneFloat):
    """
    A Pine float series that is also a function, whose call with a band multiplier returns
    the series with its upper and lower bands (``ta.vwap``).
    """

    @overload
    def __call__(self, source: PyneFloat | None = None, anchor: bool | None = None,
                 stdev_mult: None = None) -> PyneFloat: ...

    @overload
    def __call__(self, source: PyneFloat | None, anchor: bool | None,
                 stdev_mult: PyneFloat) -> tuple[PyneFloat, PyneFloat, PyneFloat]: ...


@overload
def module_function_property(func: _FI) -> IntFunctionProperty[_FI]:
    """
    Same runtime behavior as :func:`module_property`, for the Pine
    function-and-variable hybrids that are also called with arguments.

    Statically the result is both: a value of the function's return type, and a
    callable with the function's own signature.
    """


@overload
def module_function_property(func: _FF) -> FloatFunctionProperty[_FF]: ...


@overload
def module_function_property(
        func: Callable[_P, PyneFloat | tuple[PyneFloat, PyneFloat, PyneFloat]]
) -> BandsFunctionProperty: ...
