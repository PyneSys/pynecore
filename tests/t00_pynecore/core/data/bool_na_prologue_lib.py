"""
@pyne

Library for ``test_090``: a two-state module (no ``na_bool`` keyword) whose
own prologue switches the mode off when it is imported.
"""
from pynecore.lib import script


def one() -> float:
    return 1.0


@script.library("Bool na prologue lib")
def main():
    pass
