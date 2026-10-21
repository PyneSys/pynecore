"""
WebAssembly gate: the transform built for the browser against the Python pipeline.

Every input is parsed and run through the native steps inside the WebAssembly
module (Node as the host, no Python involved), and through the Python steps of
the same names over ``ast.parse``; the two ``ast.dump`` texts must be equal.

    cargo build -p pyne-transform-wasm --target wasm32-unknown-unknown --release
    python rust/tools/wasm_gate.py [PATH ...]
"""
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pynecore.transformers import pipeline
from pynecore.transformers.pipeline import ANALYSIS, PipelineContext, Step

from parse_gate import default_inputs

RUST = Path(__file__).resolve().parent.parent
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
    expected: dict[str, str] = {}
    requests = []
    for path in paths:
        source = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        ctx = PipelineContext(path=path, pyne_mode=None, source=source)
        for step in steps:
            tree = step.run(tree, ctx)
        expected[str(path)] = ast.dump(tree, include_attributes=True)
        requests.append(json.dumps({'source': source, 'path': str(path), 'steps': names,
                                    'include_attributes': True}))
    result = subprocess.run(['node', str(RUST / 'tools' / 'wasm_gate.mjs'), str(WASM)],
                            input='\n'.join(requests) + '\n', capture_output=True, text=True,
                            check=True)
    counts: Counter[str] = Counter()
    for path, line in zip(expected, result.stdout.splitlines()):
        response = json.loads(line)
        if 'dump' not in response:
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
