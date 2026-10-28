"""
Collect the Python names the call-site isolation decides on into ``python_names.json``.

The isolation step routes a call by what its callee name IS: a builtin is called
plainly, an identifier the parser would read back as a keyword cannot be rebuilt
from its spelling, and a callee imported from the standard library carries no
Pyne state. Those three name sets differ from one interpreter to the next, and
the transform must not depend on the interpreter it happens to run under -- the
native transform has none at all -- so they are data, compiled into both
backends.

The name sets hold the UNION over every supported interpreter: a name one version
has and another lacks only ever routes a call that no module of the other
version can make (an unbound name, an import that fails), so the union answers
for all of them. The ``unprintable`` section is per version instead: the code
points ``str.isprintable`` rejects, which decide ``repr`` and ``ast.unparse`` of a
string and follow each interpreter's own Unicode database. Run this script with every supported interpreter (3.11 and up)
from the checkout root; each run adds what its interpreter knows and keeps the
rest. It imports nothing of pynecore, so a bare interpreter runs it::

    python3.11 scripts/python_names_collector.py
    python3.12 scripts/python_names_collector.py
    ...

``tests/t00_pynecore/ast/test_145_python_names_registry.py`` fails until the
running interpreter's names are in the file.
"""
import builtins
import json
import keyword
import re
import sys
from pathlib import Path
from typing import Any

_JSON_PATH = Path(__file__).resolve().parent.parent / 'src' / 'pynecore' / 'transformers' \
    / 'python_names.json'


def unprintable_ranges() -> list[list[int]]:
    """
    The code points ``str.isprintable`` rejects in the running interpreter.

    :return: Inclusive ``[first, last]`` ranges, ascending
    """
    ranges: list[list[int]] = []
    start = None
    for code in range(sys.maxunicode + 1):
        printable = chr(code).isprintable()
        if not printable and start is None:
            start = code
        elif printable and start is not None:
            ranges.append([start, code - 1])
            start = None
    if start is not None:
        ranges.append([start, sys.maxunicode])
    return ranges


def collect() -> dict[str, Any]:
    """
    The names and the printability of the running interpreter.

    :return: ``builtins`` (``vars(builtins)``), ``keywords`` (``keyword.kwlist``) and
             ``stdlib`` (``sys.stdlib_module_names``), each sorted, and ``unprintable``
             with this interpreter's ``major.minor`` as its one key
    """
    version = f'{sys.version_info[0]}.{sys.version_info[1]}'
    return {
        'builtins': sorted(vars(builtins)),
        'keywords': sorted(keyword.kwlist),
        'stdlib': sorted(sys.stdlib_module_names),
        'unprintable': {version: unprintable_ranges()},
    }


def merged(current: dict[str, Any], committed: dict[str, Any]) -> dict[str, Any]:
    """
    The running interpreter's data merged into the data already collected: the union of
    the name sets, and its own ``unprintable`` entry beside the other versions'.

    :param current: What :func:`collect` found
    :param committed: The file as it is, empty when there is none
    :return: Every section; the name sets sorted, the versions in order
    """
    result: dict[str, Any] = {}
    for section, value in current.items():
        if section == 'unprintable':
            versions = {**committed.get(section, {}), **value}
            result[section] = {version: versions[version]
                               for version in sorted(versions, key=_version_key)}
        else:
            result[section] = sorted(set(value) | set(committed.get(section, ())))
    return result


def _version_key(version: str) -> tuple[int, ...]:
    """
    :param version: ``major.minor``
    :return: Its numeric order key
    """
    return tuple(int(part) for part in version.split('.'))


def main() -> None:
    committed = json.loads(_JSON_PATH.read_text()) if _JSON_PATH.exists() else {}
    result = merged(collect(), committed)
    # One code point range per line
    text = re.sub(r'\[\n\s+(\d+),\n\s+(\d+)\n\s+\]', r'[\1, \2]', json.dumps(result, indent=1))
    _JSON_PATH.write_text(text + '\n')
    added = {section: len(names) - len(committed.get(section, ()))
             for section, names in result.items() if section != 'unprintable'}
    print(f'{_JSON_PATH.name}: Python {sys.version_info[0]}.{sys.version_info[1]}, added {added}')


if __name__ == '__main__':
    main()
