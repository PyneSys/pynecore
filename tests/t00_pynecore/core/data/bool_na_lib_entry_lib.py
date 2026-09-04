"""
@pyne

Library for ``test_091``: its ``main`` runs on every bar as a registered
library entry and records what ``na(bool)`` is at that moment. It has no
``na_bool`` of its own: it runs in the importing script's mode.
"""
from pynecore.lib import array, na, script
from pynecore.types.na import na_bool

seen = array.new_float(1, 0.0)


@script.library("Bool na lib entry")
def main():
    array.set(seen, 0, 1.0 if na(bool) is na_bool else 0.0)
