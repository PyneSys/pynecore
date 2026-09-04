"""
@pyne

Library for ``test_092``: a three-state module (``na_bool=True``) that
defines a UDT with an ``NA(bool)`` default reached through a package alias,
and builds it for the caller.
"""
import pynecore as p
from pynecore.core.pine_udt import udt
from pynecore.lib import script


@udt
class Flag:
    f: bool = p.types.na.NA(bool)


def make() -> Flag:
    return Flag.new()


@script.library("Bool na UDT lib3", na_bool=True)
def main():
    pass
