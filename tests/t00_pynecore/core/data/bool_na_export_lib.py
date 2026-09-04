"""
@pyne

Library for ``test_096``: a three-state library whose exported function builds
a bool na of its own. Its caller is a v6 script, so the na must survive the
module boundary -- flattened to False it would take the other branch.
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import na, script
from pynecore.types.na import na_bool
from typing import Any, Protocol


class _Direction(Protocol):
    def __call__(self, up: bool) -> Any: ...


direction: _Direction = Exported()

__all__ = ['direction']


@script.library("Bool na export lib", na_bool=True)
def main():
    @export
    def direction(up: bool):
        # A v4/v5 library starts its state as na and asks for it later
        state: bool = na(bool)
        seen_na = 1.0 if state is na_bool else 0.0
        if na(state):
            state = up
        return seen_na, 1.0 if state else 0.0
