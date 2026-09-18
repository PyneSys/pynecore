"""
@pyne

Library for ``test_091``: its ``main`` runs on every bar as a registered
library entry and records what ``na(bool)`` is at that moment. It has no
``na_bool`` of its own: it runs in the importing script's mode.

The record travels to the importing script through an exported function, which
is defined inside ``main`` and therefore reads ``main``'s own state.
"""
from typing import Protocol

from pynecore.core.pine_export import Exported, export
from pynecore.lib import na, script
from pynecore.types import Persistent
from pynecore.types.na import na_bool


class _ProtocolSeen(Protocol):
    def __call__(self) -> float: ...


seen: _ProtocolSeen = Exported()


@script.library("Bool na lib entry")
def main():
    recorded: Persistent[float] = 0.0
    recorded = 1.0 if na(bool) is na_bool else 0.0

    @export
    def seen() -> float:
        return recorded
