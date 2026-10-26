"""
WebAssembly gate: the transform built for the browser against the Python pipeline.

Every input is parsed and run through the whole pipeline natively inside the
WebAssembly module (Node as the host, no Python involved), and through the Python
steps over ``ast.parse`` -- both halves, as the import hook runs them; the two
``ast.dump`` texts must be equal, a module the Python steps reject must be
rejected natively too, and no input may be one the native port declines: the
browser has no Python to fall back on.

Both sides find the modules an input imports the way the import hook does: the
Python steps through ``compile_interface``, the WebAssembly module by transforming
each Pyne dependency itself. What it needs to know about the files around an input
it asks for (a ``needs`` response), and the gate answers from this interpreter and
this file system and sends the request again, until nothing is left open.

    cargo build -p pyne-transform-wasm --target wasm32-unknown-unknown --release
    python rust/tools/wasm_gate.py [PATH ...]
"""
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pynecore
from pynecore.core.import_hook import (  # noqa
    _get_transform_pipeline_hash, _module_mode, _script_bool_na, compile_interface,
)
from pynecore.transformers.export_capture import module_origin
from pynecore.transformers.function_isolation import classify_imported
from pynecore.transformers.module_paths import module_spec_origin
from pynecore.transformers.pipeline import (  # noqa
    ANALYSIS, LOWERING, PipelineContext, _units, run_steps,
)
from pynecore.transformers.slot_layout import ModuleLayout
from pynecore.transformers.source_read import stable_source

from parse_gate import default_inputs

RUST = Path(__file__).resolve().parent.parent
PACKAGE = Path(pynecore.__file__).resolve().parent
WASM = RUST / 'target' / 'wasm32-unknown-unknown' / 'release' / 'pyne_transform_wasm.wasm'


class Answers:
    """What this interpreter and file system say about the modules around the inputs."""

    def __init__(self) -> None:
        self.modules: dict[str, str | None] = {}
        self.origins: dict[tuple[str, str, int], str | None] = {}
        self.files: dict[str, dict | None] = {}
        self.objects: dict[tuple[str, tuple[str, ...], bool], tuple[str, list[str | None]]] = {}

    def answer(self, need: dict, request: dict) -> None:
        """Put the answer to one open question into a request."""
        if 'module' in need:
            dotted = need['module']
            if dotted not in self.modules:
                self.modules[dotted] = module_spec_origin(dotted)
            request['modules'][dotted] = self.modules[dotted]
        elif 'origin' in need:
            name, importer, level = need['origin']
            key = (name, importer, level)
            if key not in self.origins:
                found = module_origin(name, importer, level)
                self.origins[key] = None if found is None else str(found)
            request['origins'].append([name, importer, level, self.origins[key]])
        elif 'file' in need:
            path = need['file']
            if path not in self.files:
                self.files[path] = self.file(path)
            request['files'][path] = self.files[path]
        else:
            module, names, star = need['object']
            key = (module, tuple(names), star)
            if key not in self.objects:
                route, read = classify_imported(module, names, star)
                self.objects[key] = (route, list(read))
            request['objects'].append([module, names, star, *self.objects[key]])

    @staticmethod
    def file(path: str) -> dict | None:
        """A file as the request carries it, None when there is no file at the path."""
        source = Path(path)
        if not source.is_file():
            return None
        stat = source.stat()
        entry = {'resolved': str(source.resolve()), 'mtime_ns': stat.st_mtime_ns,
                 'size': stat.st_size, 'source': None}
        loaded = stable_source(source)
        if loaded is not None:
            data, (entry['mtime_ns'], entry['size']) = loaded
            try:
                entry['source'] = data.decode('utf-8')
            except UnicodeDecodeError:
                pass
        return entry


def run_wasm(requests: list[dict]) -> list[dict]:
    """The responses to requests, each sent again with its open questions answered."""
    answers = Answers()
    responses: list[dict] = [{} for _ in requests]
    pending = list(range(len(requests)))
    while pending:
        result = subprocess.run(['node', str(RUST / 'tools' / 'wasm_gate.mjs'), str(WASM)],
                                input='\n'.join(json.dumps(requests[index]) for index in pending)
                                + '\n', capture_output=True, text=True, check=True)
        still = []
        for index, line in zip(pending, result.stdout.splitlines()):
            response = json.loads(line)
            if 'needs' in response:
                for need in response['needs']:
                    answers.answer(need, requests[index])
                still.append(index)
            else:
                responses[index] = response
        pending = still
    return responses


def main() -> int:
    sys.setrecursionlimit(200_000)
    # The reference is the Python pipeline, whatever backend the environment picks
    os.environ['PYNE_AST_BACKEND'] = 'python'
    # The dependencies the Python steps transform are left out of every __pycache__
    sys.dont_write_bytecode = True
    paths = [Path(arg) for arg in sys.argv[1:]] or default_inputs()
    pipeline_hash = _get_transform_pipeline_hash()
    #: Per input: the Python dump, or None where the Python steps reject the module
    expected: dict[str, str | None] = {}
    requests = []
    for path in paths:
        source = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        pyne_mode = _module_mode(tree)[1]
        user_code = not path.resolve().is_relative_to(PACKAGE)
        analysis = PipelineContext(path=path, pyne_mode=pyne_mode, source=source,
                                   user_code=user_code, analyse=compile_interface,
                                   pipeline_hash=pipeline_hash)
        names = [name for step in ANALYSIS for name, _ in _units(step, analysis)]
        na_bool = False
        try:
            na_bool = bool(_script_bool_na(tree, path))
            lowering = PipelineContext(path=path, pyne_mode=pyne_mode, user_code=user_code,
                                       analyse=compile_interface, pipeline_hash=pipeline_hash,
                                       na_bool=na_bool,
                                       slot_layout=ModuleLayout(compacted_series=pyne_mode == 'lib'))
            names += [name for step in LOWERING for name, _ in _units(step, lowering)]
            tree = run_steps(ANALYSIS, tree, analysis)
            tree = run_steps(LOWERING, tree, lowering)
            expected[str(path)] = ast.dump(tree, include_attributes=True)
        except SyntaxError:
            expected[str(path)] = None
        requests.append({'source': source, 'path': str(path), 'pyne_mode': pyne_mode,
                         'user_code': user_code, 'na_bool': na_bool, 'steps': names,
                         'include_attributes': True,
                         'python': f'{sys.version_info[0]}.{sys.version_info[1]}',
                         'package': str(PACKAGE), 'modules': {}, 'origins': [], 'files': {},
                         'objects': []})
    counts: Counter[str] = Counter()
    #: Why the native port declined, digits folded, so one reason counts once
    reasons: Counter[str] = Counter()
    for path, response in zip(expected, run_wasm(requests)):
        if 'unsupported' in response:
            reasons[re.sub(r'\d+', 'N', response['unsupported'])[:120]] += 1
        if expected[path] is None or 'rejected' in response:
            if expected[path] is None and 'rejected' in response:
                counts['rejected'] += 1
            elif 'unsupported' in response:
                counts['unsupported'] += 1
            else:
                counts['DIFFERENT'] += 1
                print('verdict differs:', path)
        elif 'dump' not in response:
            counts['unsupported' if 'unsupported' in response else 'ERROR'] += 1
            if 'error' in response:
                print(path, response['error'])
        elif response['dump'] == expected[path]:
            counts['equal'] += 1
        else:
            counts['DIFFERENT'] += 1
            print('different:', path)
    for reason, count in reasons.most_common(30):
        print(f'{count:6}  unsupported: {reason}')
    print(dict(counts))
    return 1 if counts['DIFFERENT'] or counts['ERROR'] or counts['unsupported'] else 0


if __name__ == '__main__':
    sys.exit(main())
