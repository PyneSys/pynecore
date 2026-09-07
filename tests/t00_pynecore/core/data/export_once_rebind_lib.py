"""
@pyne

Library for ``test_097``: two shapes of an export whose name ``main`` also binds
somewhere else, both with the OTHER binding ahead of the export.

``ahead`` carries an unexported definition of the same name before it, the way
a Pine library reads when the exported overload is written second. ``branched``
is preceded by a branch that may rebind the name and, on this data, never does.

Neither may go under the latch. Latched, the export's own binding runs on bar 0
only: from bar 1 ``ahead`` would resolve to the two-argument definition that is
rebuilt every bar, and ``branched`` to nothing at all.
"""
from typing import Any, Protocol

from pynecore.core.pine_export import Exported, export
from pynecore.lib import script
from pynecore.types import Persistent


class Holder:
    """Receiver of the exported methods; a stand-in for a compiled Pine UDT."""

    def __init__(self, value: float = 0.0):
        self.value = value

    @staticmethod
    def new(value: float = 0.0) -> 'Holder':
        return Holder(value)


class _Picked(Protocol):
    def __call__(self, this: Holder) -> Any: ...


ahead: _Picked = Exported()
branched: _Picked = Exported()

__all__ = ['Holder', 'ahead', 'branched']


@script.library("Export Once Rebind Lib")
def main():
    offset: Persistent[float] = 10.0
    flag: Persistent[bool] = False

    # Unexported same-named definition, written BEFORE the export
    def ahead(this: Holder, other: float):
        return this.value + other + offset

    shadowed = ahead

    @export
    def ahead(this: Holder):  # noqa: F811
        return this.value + offset

    if flag:
        branched = 0.0  # noqa: F841

    @export
    def branched(this: Holder):  # noqa: F811
        return this.value + offset + 1.0

    return {"ahead": ahead(Holder(1.0)), "branched": branched(Holder(1.0)),
            "shadowed": shadowed(Holder(1.0), 2.0)}
