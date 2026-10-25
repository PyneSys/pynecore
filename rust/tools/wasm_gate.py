"""
WebAssembly gate: the transform built for the browser against the Python pipeline.

Every input is parsed and run through the native steps inside the WebAssembly
module (Node as the host, no Python involved), and through the Python steps of
the same names over ``ast.parse``, each step under the rules of the pipeline
(``user_code_only``); the two ``ast.dump`` texts must be equal, and a module the
Python steps reject must be rejected natively too.

    cargo build -p pyne-transform-wasm --target wasm32-unknown-unknown --release
    python rust/tools/wasm_gate.py [PATH ...]
"""
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pynecore
from pynecore.core.import_hook import _module_mode  # noqa
from pynecore.transformers.pipeline import ANALYSIS, PipelineContext, Step

from parse_gate import default_inputs

RUST = Path(__file__).resolve().parent.parent
PACKAGE = Path(pynecore.__file__).resolve().parent
WASM = RUST / 'target' / 'wasm32-unknown-unknown' / 'release' / 'pyne_transform_wasm.wasm'


def python_steps() -> list[Step]:
    """The leading ANALYSIS steps that have a native port, in pipeline order."""
    ported = set(json.loads((RUST / 'crates' / 'pyne-transform' / 'ported.json').read_text()))
    steps = []
    for step in ANALYSIS:
        if not isinstance(step, Step) or step.name not in ported:
            break
        steps.append(step)
    return steps


def main() -> int:
    sys.setrecursionlimit(200_000)
    steps = python_steps()
    names = [step.name for step in steps]
    paths = [Path(arg) for arg in sys.argv[1:]] or default_inputs()
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
        ctx = PipelineContext(path=path, pyne_mode=pyne_mode, source=source,
                              user_code=user_code)
        run = [step for step in steps if user_code or not step.user_code_only]
        try:
            for step in run:
                tree = step.run(tree, ctx)
            expected[str(path)] = ast.dump(tree, include_attributes=True)
        except SyntaxError:
            expected[str(path)] = None
        requests.append(json.dumps({'source': source, 'path': str(path),
                                    'pyne_mode': pyne_mode, 'user_code': user_code,
                                    'steps': [step.name for step in run],
                                    'include_attributes': True,
                                    'python': f'{sys.version_info[0]}.{sys.version_info[1]}'}))
    result = subprocess.run(['node', str(RUST / 'tools' / 'wasm_gate.mjs'), str(WASM)],
                            input='\n'.join(requests) + '\n', capture_output=True, text=True,
                            check=True)
    counts: Counter[str] = Counter()
    for path, line in zip(expected, result.stdout.splitlines()):
        response = json.loads(line)
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
    print('steps:', ', '.join(names), dict(counts))
    return 1 if counts['DIFFERENT'] or counts['ERROR'] else 0


if __name__ == '__main__':
    sys.exit(main())
