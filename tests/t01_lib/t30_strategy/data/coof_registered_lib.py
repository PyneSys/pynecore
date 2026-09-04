"""
@pyne

Library module for ``test_031``: a state-carrying library entry that the
strategy imports. Its ``main`` is registered as a library entry and runs on
every bar next to the strategy's ``main``.
"""
from pynecore.lib import script
from pynecore.types import Persistent


@script.library("COOF Registered Lib")
def main():
    lib_calls: Persistent[int] = 0
    lib_calls += 1
    return lib_calls
