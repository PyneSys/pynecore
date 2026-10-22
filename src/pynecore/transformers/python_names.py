"""
The Python names the call-site isolation decides on, as compiled data.

Builtins, keywords and the standard library's top-level modules differ from one
interpreter to the next, and the transform must not follow the interpreter it
runs under: the generated ``python_names.json`` holds the union over every
supported version (see ``scripts/python_names_collector.py``), which the native
transform compiles in.
"""
import json
from pathlib import Path

__all__ = ['is_builtin', 'is_keyword', 'is_stdlib']

_PATH = Path(__file__).parent / 'python_names.json'
_NAMES: dict[str, frozenset[str]] = {}


def _section(name: str) -> frozenset[str]:
    """
    One section of the registry, parsed once per process.

    :param name: ``builtins``, ``keywords`` or ``stdlib``
    :return: Its names
    """
    if not _NAMES:
        _NAMES.update((section, frozenset(names))
                      for section, names in json.loads(_PATH.read_text()).items())
    return _NAMES[name]


def is_builtin(name: str) -> bool:
    """
    Whether a name is a builtin of a supported interpreter (``name in vars(builtins)``).

    :param name: The name
    :return: True for a builtin
    """
    return name in _section('builtins')


def is_keyword(name: str) -> bool:
    """
    Whether the parser reads a name as a keyword (``keyword.iskeyword``).

    :param name: The name
    :return: True for a keyword
    """
    return name in _section('keywords')


def is_stdlib(module: str) -> bool:
    """
    Whether a module's top-level package is part of the standard library
    (``sys.stdlib_module_names``).

    :param module: Full module path (``os.path``)
    :return: True for a standard library module
    """
    return module.split('.', 1)[0] in _section('stdlib')
