"""
Collect the Python names the call-site isolation decides on into ``python_names.json``.

The isolation step routes a call by what its callee name IS: a builtin is called
plainly, an identifier the parser would read back as a keyword cannot be rebuilt
from its spelling, and a callee imported from the standard library carries no
Pyne state. Those three name sets differ from one interpreter to the next, and
the transform must not depend on the interpreter it happens to run under -- the
native transform has none at all -- so they are data, compiled into both
backends.

The file holds the UNION over every supported interpreter: a name one version
has and another lacks only ever routes a call that no module of the other
version can make (an unbound name, an import that fails), so the union answers
for all of them. Run this script with every supported interpreter (3.11 and up)
from the checkout root; each run adds what its interpreter knows and keeps the
rest. It imports nothing of pynecore, so a bare interpreter runs it::

    python3.11 scripts/python_names_collector.py
    python3.12 scripts/python_names_collector.py
    ...

``tests/t00_pynecore/ast/test_143_python_names_registry.py`` fails until the
running interpreter's names are in the file.
"""
import builtins
import json
import keyword
import sys
from pathlib import Path

_JSON_PATH = Path(__file__).resolve().parent.parent / 'src' / 'pynecore' / 'transformers' \
    / 'python_names.json'


def collect() -> dict[str, list[str]]:
    """
    The names of the running interpreter.

    :return: ``builtins`` (``vars(builtins)``), ``keywords`` (``keyword.kwlist``) and
             ``stdlib`` (``sys.stdlib_module_names``), each sorted
    """
    return {
        'builtins': sorted(vars(builtins)),
        'keywords': sorted(keyword.kwlist),
        'stdlib': sorted(sys.stdlib_module_names),
    }


def merged(current: dict[str, list[str]], committed: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    The union of the running interpreter's names and the names already collected.

    :param current: What :func:`collect` found
    :param committed: The file as it is, empty when there is none
    :return: Every section, sorted
    """
    return {section: sorted(set(names) | set(committed.get(section, ())))
            for section, names in current.items()}


def main() -> None:
    committed = json.loads(_JSON_PATH.read_text()) if _JSON_PATH.exists() else {}
    result = merged(collect(), committed)
    _JSON_PATH.write_text(json.dumps(result, indent=1) + '\n')
    added = {section: len(names) - len(committed.get(section, ()))
             for section, names in result.items()}
    print(f'{_JSON_PATH.name}: Python {sys.version_info[0]}.{sys.version_info[1]}, added {added}')


if __name__ == '__main__':
    main()
