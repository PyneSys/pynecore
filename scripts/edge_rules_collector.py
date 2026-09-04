#!/usr/bin/env python3
"""
Generate ``pynecore/transformers/edge_rules.json`` from the Pyne Edge profile.

The ``@pyne edge`` gate rejects every construct the Edge profile does not
allow, and the profile is defined ONCE, in the PyneIDE's
``pyneide_edge_rules.py``: the IDE lints against it, this gate compiles
against it, and the AOT validator will consume the same data. A hand-copied
rule set would drift, and the drift would show as a script the IDE accepts
and the compiler rejects -- or the other way round -- which is exactly the
guarantee the gate exists to give. So the rules are EXTRACTED: this script
imports the spec by path and writes the sets it declares as data, versioned
by the spec's own ``EDGE_RULES_VERSION``.

One thing lives here rather than in the spec: ``EXTRAS``, what PyneCore
allows ON TOP of the IDE profile. Today that is the ``@overload`` decorator:
an overload group is a Pine construct (one name, several signatures), the type
pass pins its call sites statically, and the compiler emits it. The IDE spec
predates it; it is a profile revision to make there.

PyneCore's own ``@pyne lib`` modules are NOT gated. They are the machines
behind the lib, written in Python -- measured over the four of them, ``ta.py``
alone uses 156 constructs the profile has no place for (``heapq``,
comprehensions, ``assert``, slices, ``isinstance``) -- so a "lib profile"
would have to allow Python, which is no profile at all. The ``lib`` mode word
selects the series semantics of those modules, nothing about the gate.

Usage:
    python3 scripts/edge_rules_collector.py [path/to/pyneide_edge_rules.py]

The default spec path assumes the sibling checkout layout of the PyneSys
monorepo (``../PyneIDE/python/pyneide_edge_rules.py`` next to this repo);
``PYNE_EDGE_RULES_SPEC`` overrides it.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

#: Format version of the JSON. Bump whenever the shape below changes; the
#: consumer (``transformers/pine_edge_gate.py``) pins it.
SCHEMA_VERSION = 1

#: What PyneCore allows on top of the IDE's profile, in the spec's own shapes.
EXTRAS: dict[str, Any] = {
    'func_decorators': [['pynecore.core.overload', 'overload']],
    # Double-underscore names are the pipeline's: a script may spell only the
    # ones the compiler emits (the ``__main__`` guard, block results, switch
    # subjects, input parameters) and a signature shim's ``__call__``
    'dunder_names': ['__all__', '__name__', '__file__', '__call__'],
    'dunder_patterns': ['^__block_result(_\\d+)?__$', '^__block_keep_\\d+__$',
                        '^__switch(_\\d+)?__$', '^__loop_\\d+__$', '^__input_\\d+__$'],
}


def default_spec_path() -> Path:
    """The spec's path under the monorepo's sibling-checkout layout, or the override."""
    override = os.environ.get('PYNE_EDGE_RULES_SPEC')
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / 'PyneIDE' / 'python' / 'pyneide_edge_rules.py'


def load_spec(path: Path) -> Any:
    """Import the spec module by path, without it being on ``sys.path``."""
    spec = importlib.util.spec_from_file_location('pyneide_edge_rules', path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(spec: Any) -> dict[str, Any]:
    """
    The rules as data, every set sorted so the file is diffable.

    :param spec: The imported ``pyneide_edge_rules`` module
    :return: The JSON document
    """
    return {
        'v': SCHEMA_VERSION,
        'rules_version': spec.EDGE_RULES_VERSION,
        'nodes': sorted(spec.ALLOWED_NODES),
        'bin_ops': sorted(spec.ALLOWED_BIN_OPS),
        'unary_ops': sorted(spec.ALLOWED_UNARY_OPS),
        'bool_ops': sorted(spec.ALLOWED_BOOL_OPS),
        'cmp_ops': sorted(spec.ALLOWED_CMP_OPS),
        'import_prefixes': list(spec.ALLOWED_IMPORT_PREFIXES),
        'from_modules': {module: sorted(names)
                         for module, names in sorted(spec.ALLOWED_FROM_MODULES.items())},
        'func_decorators': sorted(list(pair) for pair in spec.ALLOWED_FUNC_DECORATORS),
        'class_decorators': sorted(list(pair) for pair in spec.ALLOWED_CLASS_DECORATORS),
        'builtin_calls': sorted(spec.ALLOWED_BUILTIN_CALLS),
        'extras': EXTRAS,
    }


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else default_spec_path()
    document = collect(load_spec(path))
    target = Path(__file__).resolve().parent.parent / 'src' / 'pynecore' / 'transformers' \
        / 'edge_rules.json'
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote {target} (rules {document["rules_version"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
