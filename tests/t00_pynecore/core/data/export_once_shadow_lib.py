"""
@pyne

Library for ``test_097``: it exports a METHOD named ``picked`` and keeps a
second, unexported definition of that same name, the shape a Pine library takes
when only one overload of a method is exported (``jason5480/chrono_utils``'
``is_bar_included``).

The unexported definition sits outside the export run, so it is rebuilt on every
bar. It must bind a local: were it to travel through the latch's ``global``
declaration it would write over the module-level ``Exported`` proxy the
decorator registered, and ``pine_method.method_call`` — which looks the name up
on the module at CALL time and demands an ``Exported`` — would fail on the very
first bar.
"""
from typing import Any, Protocol

from pynecore.core.pine_export import Exported, export
from pynecore.lib import script
from pynecore.types import Persistent


class Holder:
    """Receiver of the exported method; a stand-in for a compiled Pine UDT."""

    def __init__(self, value: float = 0.0):
        self.value = value

    @staticmethod
    def new(value: float = 0.0) -> 'Holder':
        return Holder(value)


class _Picked(Protocol):
    def __call__(self, this: Holder) -> Any: ...


picked: _Picked = Exported()

__all__ = ['Holder', 'picked']


@script.library("Export Once Shadow Lib")
def main():
    offset: Persistent[float] = 10.0

    @export
    def picked(this: Holder):
        return this.value + offset

    # Unexported same-named definition, rebuilt on every bar
    def picked(this: Holder, other: float):  # noqa: F811
        return this.value + other + offset

    return {"local": picked(Holder(1.0), 2.0)}
