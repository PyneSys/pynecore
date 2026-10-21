# PyneCore native transform

A Rust port of PyneCore's AST transform pipeline (`src/pynecore/transformers`), so
the pipeline can run where there is no Python interpreter (the browser, the AOT
compiler) and, step by step, faster under CPython.

**The Python pipeline is the reference.** Every native step is a port of the
Python step of the same name and must produce the same tree for the same input.
New behaviour is written in Python first; the native step follows.

## Crates

| Crate                 | What it is                                                                     |
|-----------------------|--------------------------------------------------------------------------------|
| `pyne-ast`            | The CPython `ast` tree as an arena; node classes generated from CPython 3.14   |
| `pyne-parse`          | Python source to `pyne-ast`, with CPython's locations (Ruff's parser inside)   |
| `pyne-transform`      | The ported steps and the step registry                                         |
| `pyne-transform-py`   | The `pynecore.transformers._native` extension module (PyO3, abi3)              |
| `pyne-transform-wasm` | The WebAssembly entry                                                          |

A node is addressed by a `NodeId`, the way a Python pass holds a reference to an
`ast.AST` object, so a port can keep ids in sets and maps, share a node between
two parents and mutate nodes it reached elsewhere, as the Python passes do. What
a pass stamps on a node (`_pine_ty`, `pine_exact`, ...) is kept in the node's
`extra` map under the same attribute name. The traversal helpers in
`pyne_ast::walk` visit in the same order as `transformers/ast_walk.py`.

## Which backend runs a step

`PYNE_AST_BACKEND` selects it (see `src/pynecore/transformers/native.py`):

- `python` (default): every step runs in Python.
- `native`: a step with a current native port runs natively; consecutive native
  steps run in one call.
- `verify`: every step runs in Python, and each step with a current port also runs
  natively on the same input; a difference raises `NativeMismatch`.

A port is **current** while the Python sources it was ported from are unchanged.
`crates/pyne-transform/ported.json` records, per step, the Python module and the
digest of its source closure (the module, the `transformers` modules it imports,
the pynecore modules those import, the data files they name). The manifest is
compiled into the extension; at run time the digest of today's sources is
compared with it, and a step whose Python changed runs in Python until its port
catches up. A native step that meets a construct it does not handle answers
`Unsupported` and the Python step runs instead.

## Workflow

```bash
rust/tools/develop.sh                       # build the extension into src/ (debug)
python rust/tools/parse_gate.py             # native parser vs ast.parse over the corpus
PYNE_AST_BACKEND=verify pytest tests/t00_pynecore/ast
python rust/tools/ported.py                 # which ports are current
python rust/tools/ported.py --update NAME   # after a port matches its Python step again
```

Run the Python tools with `PYTHONPATH=src` from the checkout root.

Porting a step:

1. Write the port in `crates/pyne-transform/src/steps/<module>.rs`, mirroring the
   Python module, and register it in `STEPS`.
2. `python rust/tools/ported.py --add <StepName> <module>`, then `develop.sh`.
3. Gate it: the AST tests under `verify`, and the emission oracle
   (`work/ast-oracle/oracle.py` in the monorepo) captured with
   `PYNE_AST_BACKEND=python`, `verify` and `native`, compared pairwise: zero
   mismatches.

`crates/pyne-ast/src/nodes.rs` is generated: `python3.14 crates/pyne-ast/gen/gen_nodes.py`.
