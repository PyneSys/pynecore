import sys


def is_stdlib(module_name: str) -> bool:
    """
    Check whether a module belongs to Python's standard library.

    Resolution is name-based via ``sys.stdlib_module_names`` (a frozenset of
    every stdlib top-level module name on Python 3.10+; PyneCore requires
    3.11+). This is authoritative and independent of the install layout: a
    path-based check misclassifies packages under
    ``<prefix>/lib/pythonX.Y/site-packages`` as stdlib.

    :param module_name: Full module path (e.g. ``os.path``).
    :return: True if the module's top-level package is part of the stdlib.
    """
    return module_name.split('.', 1)[0] in sys.stdlib_module_names
