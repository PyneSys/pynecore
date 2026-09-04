"""
@pyne

Library for ``test_092``: a two-state module (no ``na_bool`` keyword) that
defines a UDT with a bool na default and builds it for the caller; its
``__future__`` import must stay first past the generated imports.
"""
from __future__ import annotations

from pynecore.core.pine_udt import udt
from pynecore.lib import na, script


@udt
class Flag:
    f: bool = na(bool)


def make() -> Flag:
    return Flag.new()


@script.library("Bool na UDT lib")
def main():
    pass
