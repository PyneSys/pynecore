<!--
---
weight: 1001
title: "AST Transformation"
description: "How PyneCore uses AST transformation to implement Pine Script behavior"
icon: "code"
date: "2025-03-31"
lastmod: "2026-10-09"
draft: false
toc: true
categories: ["Advanced", "Technical Implementation"]
tags: ["ast", "python", "transformations", "internals"]
---
-->

# AST Transformation

PyneCore runs Pyne code through a chain of AST passes when the module is imported. The passes
rewrite plain Python so it behaves like Pine Script: bar-by-bar history, per-call-site state,
Pine's division, comparison and `na` rules, and the `request.security()` protocol. This page
describes the whole chain: how the loader drives it, which steps there are and why they stand in
that order, how to add or change one, and how the result is cached and inspected.

## Overview

### Import Hook System

The system's entry point is the import hook, which transforms Python files marked with the `@pyne` magic comment:

```python
# Import hook through importlib meta_path system
sys.meta_path.insert(0, PyneImportHook())
```

`PyneImportHook.find_spec` looks for `<name>.py` or `<name>/__init__.py` along the search path and
returns a spec backed by `PyneLoader`, a `SourceFileLoader` whose `source_to_code` decides what
happens to a module:

| Module                                                                            | What the loader does                         |
|-----------------------------------------------------------------------------------|----------------------------------------------|
| Pyne code (the docstring starts with `@pyne`)                                     | Runs the whole pipeline described below      |
| A plain module of the `pynecore` package that contains `cast(` or `TYPE_CHECKING` | Runs the `TypeErasure` step and nothing else |
| Any other module                                                                  | Compiles it untouched                        |

### Marker Recognition Rules

The `@pyne` marker is recognized strictly to avoid accidentally transforming ordinary library modules that merely mention the token in prose:

1. The module's **first statement** must be a string-literal expression (a module docstring).
2. After `lstrip()`, the docstring's content must **start with `@pyne`**.
3. `@pyne` must be followed by whitespace or the end of the docstring (so `@pynex` does not match, and a docstring like `"""Some description.\n@pyne\n"""` does **not** trigger transformation — the marker must come first, not somewhere inside).

The token may be followed by one mode word on the same line, `@pyne lib` or `@pyne edge`. `lib` marks the builtin machines shipped with PyneCore (their series are na-compacted windows), `edge` marks the compiler's output (it is checked by the edge gate), and no word at all is a hand-written script. The mode travels through the pipeline as `PipelineContext.pyne_mode`.

Cheap prefilters skip full AST parsing for files that obviously cannot match: a regex (`@pyne(\s|["']|$)`) on the source in `source_to_code`, and a regex on the first 4096 bytes in `PyneLoader.get_code` (it skips leading comment lines, so a PEP 723 `# /// script` block in front of the docstring does not hide the marker). Files that pass the prefilter still go through the strict AST check above before any transformer runs.

### Reserved Identifier Namespace

The transform injects names of its own into script scope, so two kinds of identifier are reserved in Pyne code. Using one is a `SyntaxError` that names the identifier and its position, raised before a single transformer runs. Strings, comments and docstrings may contain either kind freely: only identifiers are checked — names, parameters, attributes, keyword arguments, imports, `global` / `nonlocal`, match captures and type parameters.

**1. Any identifier containing a middle dot (`·`).** Nearly every injected name carries one: the scope-qualified state parameters and slot constants (`__state·main__`, `__slot·main·x__`), the generated temporaries (`__st·__`, `__cnt·0__`) and the aliased runtime helper imports (`__resolve_slot·__`). The character is a legal Python identifier character, so the hook reserves it in **any** position. Because Python NFKC-normalizes identifiers while parsing, the check compares the normalized form — equivalent spellings (U+0387 GREEK ANO TELEIA, and U+013F / U+0140 LATIN LETTER L WITH MIDDLE DOT, which decompose into one) are rejected the same way. A pure-ASCII module, or one without any of these characters, cannot spell such a name, so the check returns at once for it. This rule applies to every `@pyne` module.

**2. The plain double-underscore names the transform emits.** A few injected names carry no middle dot, because the runtime, the security child processes and the ahead-of-time compiler address them by these exact spellings. They are listed in one place, `transformers/pipeline_names.py` (`PIPELINE_NAME`):

| Kind                            | Names                                                                                                                                                                            |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Protocol namespace              | every `__pyne_<name>__` (`__pyne_slot_layout__`, `__pyne_layout__`, `__pyne_transformed__`, ...)                                                                                 |
| Generated temporaries           | `__cmp<N>__`, `__bool<N>__`, `__hist_<N>__`, `__sec_main_<N>__` (`<N>` is a number)                                                                                              |
| Slot-state plumbing             | `__state__`, `__attach_layout__`, `__resolve_slot__`, `__bind_any__`, `__bind_loop__`, `__bind_slot__`, `__bind_pinned__`, `__loop_state__`, `__slot_state__`, `__dyn_default__` |
| The `request.security` protocol | `__security_contexts__`, `__active_security__`, `__same_context__`, `__sec_read__`, `__sec_write__`, `__sec_wait__`, `__sec_signal__`, `__ltf_unzip__`                           |

A script that bound one of them would silently collide with the emission — for example, a module variable named `__bool1__` would be shadowed inside a function by the truthiness walrus temporary of that name. This rule applies to user code (scripts and their libraries); pynecore's own `@pyne` lib modules are the runtime side of these names. It is checked after the `__test_*__` functions are removed, so a test may use the runtime names. The names PyneComp itself writes into a compiled script (`__block_result__`, `__switch_<N>__`, `__loop_<N>__`, `__input_<N>__`, `__hist_<name>__`) belong to the script and are not reserved.

A new emitted name must be added to `PIPELINE_NAME` in the same change: `tests/t00_pynecore/ast/test_129_pipeline_names.py` fails for any plain `__x__` name literal in the transformers that the list does not cover.

### Before the First Step

`_analyse_tree` in `core/import_hook.py` prepares the module before the pipeline runs:

1. The reserved-name check above (rule 1).
2. It records the module's resolved file path on the tree (`_module_file_path`), which `Security` hashes into the per-module security ids so contexts stay unique across a script and its imported libraries.
3. It removes every module-level `__test_*__` function, so test harness code never reaches a step.
4. For user code, the pipeline-name check above (rule 2).
5. For an `@pyne edge` module the edge gate reads the tree as written, before any step injects plumbing. Its findings join the type diagnostics after the analysis half (see [Debugging](#debugging-the-transformation)).

### The Two Halves

The pipeline is declared in two halves, `ANALYSIS` and `LOWERING`, and the loader runs them with `_analyse_tree` and `_lower_tree`:

- **Analysis** normalizes the tree into the form the Pine type pass reads and stamps the types onto it. It emits none of the state plumbing (no state parameters, no slots, no call-site anchors). It ends with the type pass (`PineType`) and the security slicing (`SecuritySlice`).
- **Lowering** turns the analysed tree into the state-plumbed form the runtime executes, allocates the module's slot layout and fixes up the synthetic locations.

The split is where a module publishes its interface, and where a dependency transformed for another module's lookup stops when it reaches back into a module still under analysis (see [Caching and Invalidation](#caching-and-invalidation)). A module that calls into another Pyne module is typed against the interface that module publishes.

The loader takes the published interface from the analysed tree, before lowering: the isolation step prepends a state parameter to every state-carrying function and the series step rewrites the annotations, so a signature read afterwards would be the emission's, not the module's.

Both halves run under a raised recursion limit with the cyclic garbage collector paused for the duration of the stage (`_with_nesting_headroom`). A module nested deeper than the default limit admits (a Pine `switch` with hundreds of arms compiles to an `elif` chain of that depth) is re-parsed and run again under a limit sized to its nesting depth.

## The Pipeline as Data

The steps, their order and the reason for each position live in one file, `src/pynecore/transformers/pipeline.py`. It declares:

- `ANALYSIS`, a sequence of `Step` objects, and `LOWERING`, a sequence of `Step` and `Phase` objects;
- `PipelineContext`, what the steps of one module share besides the tree;
- `run_steps(steps, tree, ctx)`, which runs one half.

A `Step` has three fields:

| Field            | Meaning                                                                           |
|------------------|-----------------------------------------------------------------------------------|
| `name`           | The name `PYNE_AST_TIMING` reports                                                |
| `run`            | `run(tree, ctx) -> ast.Module`: the pass; it returns the (possibly replaced) tree |
| `user_code_only` | When true, the step never runs on pynecore's own `@pyne` lib modules              |

A `Phase` groups steps written as rules that run in one traversal (see [Fused Phases and the Sequential Mode](#fused-phases-and-the-sequential-mode)). Its rules are `RuleStep` objects with a `name`, a `make(ctx)` factory and their own `user_code_only` flag.

`PipelineContext` carries:

| Field           | Meaning                                                                              |
|-----------------|--------------------------------------------------------------------------------------|
| `path`          | Source path; the script / lib profile is picked from it                              |
| `pyne_mode`     | The module's mode word, `None` for a hand-written script                             |
| `source`        | Full module source, for error locations                                              |
| `user_code`     | Whether the module is user code rather than one of pynecore's own lib modules        |
| `analyse`       | Derives an imported module's signatures without importing it (used by the type pass) |
| `pipeline_hash` | Digest of the pipeline the module is transformed by (used by the type pass)          |
| `na_bool`       | Whether the module keeps Pine's three-state bool (used by `FloatTolerance`)          |
| `emit_layout`   | Whether `ApplyLayout` materializes the slot layout for CPython                       |
| `slot_layout`   | The `ModuleLayout` slot allocator shared by the lowering half                        |

The loader sets `user_code` to false for any module that lives inside the `pynecore` package directory, which is how the builtin machines under `lib/` (`@pyne lib` modules such as `ta`) are told apart from scripts and their libraries. A step marked `user_code_only` is skipped for the former. The reasons differ per step: some passes would break the natively bit-exact builtins (`FloatTolerance`, `ConstFold`), some enforce language rules that the lib modules themselves are the machinery for (`OuterWrite`, `PlotScope`).

### Adding or changing a step

1. Write the pass as a transformer (or, to share a traversal with its neighbours, as a rule, see below). Use the `ast_walk` primitives (see [Performance Notes for Contributors](#performance-notes-for-contributors)).
2. Import it in `pipeline.py` and add a `Step('Name', ...)` to `ANALYSIS` or `LOWERING` at the position it needs. The helpers `_visit(Class)` (run `Class().visit(tree)`), `_lower_layout(Class)` (the same with the shared `ModuleLayout`) and `_in_place(fn)` (a function that edits the tree in place) cover the common shapes.
3. Put a comment directly above the step that says why it stands there: which step it must follow or precede, and what breaks otherwise. The position is a contract with the neighbours, and the comment is where that contract is written down. A step without a comment has no constraint beyond the order it stands in.
4. Decide `user_code_only`: set it when the pass must not touch pynecore's own `@pyne` lib modules.
5. Add the step to the table in this page and write its section. A guard test (`tests/t00_pynecore/ast/test_128_pipeline_docs.py`) compares the tables with `ANALYSIS` and `LOWERING` and fails when they differ.
6. If the pass emits calls to a new helper that compiled scripts call, `FunctionIsolation` classifies those calls: see `NON_TRANSFORMABLE_FUNCTIONS` in `transformers/function_isolation.py` and the [Function Isolation](./function-isolation.md#where-the-classification-comes-from) page.

Editing any file under `transformers/` changes the pipeline hash, which invalidates every cached script bytecode (see [Caching and Invalidation](#caching-and-invalidation)); nothing has to be registered for that.

## The Steps

The tables list the steps in exactly the order `pipeline.py` runs them. "User code only" says whether the step is skipped for pynecore's own `@pyne` lib modules. The last column gives the ordering constraint the comment in `pipeline.py` (or the pass's own docstring) states; a dash means no constraint is stated beyond the order itself.

### Analysis Half

<!-- pipeline:analysis -->
| Step                    | Purpose                                                               | User code only | Ordering constraint                                                              |
|-------------------------|-----------------------------------------------------------------------|----------------|----------------------------------------------------------------------------------|
| `ImportLifter`          | Lift `from pynecore.lib... import` statements out of function bodies  | no             | -                                                                                |
| `TypeCheckingStripper`  | Remove `if TYPE_CHECKING:` blocks and the flag import                 | no             | -                                                                                |
| `TypeErasure`           | Replace `typing.cast(T, x)` with `x`                                  | no             | Before ImportNormalizer: it trusts `cast` by the `typing` imports as written     |
| `BuiltinShadow`         | Send accesses a shadowing library alias cannot serve to the built-ins | no             | Before ImportNormalizer: it adds the imports of the `lib.<ns>.<name>` chains     |
| `ImportNormalizer`      | Rewrite pynecore imports to `from pynecore import lib` and `lib.*`    | no             | Later steps read the normalized `lib.*` chains                                   |
| `PlotScope`             | Require plot declarations to run directly in the script entry         | yes            | Before lowering creates control flow of its own                                  |
| `OuterWrite`            | Reject writes into objects created outside a function                 | yes            | Well before FunctionIsolation (its per-call-site copies would repeat the report) |
| `ExportCapture`         | Reject library exports that capture non-constant library globals      | yes            | Before closure and state lowering                                                |
| `SecurityDrawings`      | Reject drawing creation inside `request.security` expressions         | yes            | Before the request lowering                                                      |
| `ConstFold`             | Fold constant subtrees the way TradingView does at parse time         | yes            | Needs the normalized `lib.math.*` chains                                         |
| `DynamicDefault`        | Evaluate `lib.*`-referencing parameter defaults per call              | no             | Before the series and isolation passes (the moved code needs slots and anchors)  |
| `InlineSeriesHoist`     | Hoist `inline_series` calls out of lazily evaluated positions         | no             | Before call-site anchoring (the hoisted statements are the anchorable sites)     |
| `PineTruthiness`        | Apply the tolerant float-to-bool conversion in the bool contexts      | yes            | Before passes that emit control flow of their own                                |
| `SecurityInstantiation` | Clone security-bearing functions per call site                        | no             | After ImportNormalizer, before Security                                          |
| `Security`              | Rewrite `request.security` into the signal/write/read/wait protocol   | no             | After SecurityInstantiation; assigns the positional security ids                 |
| `PersistentSeries`      | Split `PersistentSeries` into a Persistent and a Series declaration   | no             | Before the Persistent and Series lowering                                        |
| `LibrarySeries`         | Anchor a local Series for every indexed library value                 | no             | Before the Series lowering                                                       |
| `ModuleProperty`        | Turn property reads into calls, route function-and-namespace modules  | no             | Before TaVariableHoist (bare reads are calls by then)                            |
| `TaVariableHoist`       | Evaluate stateful `ta` builtin variables once per bar at top of main  | no             | After ModuleProperty; before Series, Persistent and FunctionIsolation            |
| `ClosureArguments`      | Turn closure variables of nested functions into parameters            | no             | Before PineType and FunctionIsolation                                            |
| `PineType`              | Infer the Pine types and stamp them on the nodes                      | no             | Last point where the tree still looks like Pine (see its section)                |
| `SecuritySlice`         | Emit a backward slice of `main()` per security context                | no             | Last of this half: the clones must see every earlier step                        |
<!-- /pipeline:analysis -->

### Lowering Half

A row named `Expression/<Rule>` is a rule of the fused phase `Expression`; the `Expression` row itself is the phase. A phase has no `user_code_only` flag of its own, its rules do.

<!-- pipeline:lowering -->
| Step                        | Purpose                                                         | User code only | Ordering constraint                                                      |
|-----------------------------|-----------------------------------------------------------------|----------------|--------------------------------------------------------------------------|
| `SecurityDefault`           | Resolve bool defaults of missing security results               | no             | After PineType (needs the inferred types), before the state plumbing     |
| `ExportOnce`                | Define a library's exports once per run instead of once per bar | no             | Before Series and Persistent: its latch is a Persistent slot             |
| `UnusedSeriesDetector`      | Drop the Series annotation of never-indexed variables           | no             | Before Series                                                            |
| `Series`                    | Turn Series variables into slots of the state vector            | no             | -                                                                        |
| `VerifyClosedShift`         | Clear `closed_shift` where the lowered write is no history read | no             | Directly after Series, whose emission it recognizes                      |
| `Persistent`                | Turn Persistent variables into slots of the state vector        | no             | -                                                                        |
| `CallInline`                | Copy the body of trivial stateless builtins into the call site  | no             | Before FunctionIsolation: an inlined site is no call and gets no anchor  |
| `FunctionIsolation`         | Give every call site its own callee state (child slots)         | no             | After Persistent and Series (classification needs their slots)           |
| `ScriptRequirements`        | Detect the broker capabilities a strategy needs                 | no             | -                                                                        |
| `Input`                     | Add `_id` to input calls, rebind source inputs                  | no             | Before the Expression phase (SafeConvert must see the rebinding)         |
| `Expression`                | Expression-level rewrites in one post-order traversal (phase)   | -              | See "Fused Phases and the Sequential Mode"                               |
| `Expression/SafeConvert`    | Lower `float()` / `int()` and truncate Python-native indexes    | no             | After Input (its binding scan must see the rebinding)                    |
| `Expression/SafeDivision`   | Give `/` Pine's division-by-zero and na semantics               | no             | Before FloatTolerance (wrapped operands are bound once)                  |
| `Expression/FloatTolerance` | Give comparisons TradingView's tolerant float semantics         | yes            | After SafeDivision                                                       |
| `FinalizeCloneDefaults`     | Make the defaults of the security clones inert                  | no             | After the input lowering                                                 |
| `ApplyLayout`               | Emit the slot layout, state parameters and attach statements    | no             | After every step that allocates slots; skipped when `emit_layout` is off |
| `FixLocations`              | Give synthetic nodes point locations                            | no             | Last: it locates everything the steps emitted                            |
<!-- /pipeline:lowering -->

## Fused Phases and the Sequential Mode

Most of a pass's time is the walk itself, not the rewriting. A **rule** (`transformers/phase.py`) is a pass written as per-node hooks instead of `visit_<Class>` methods, so several rules can share one traversal of the tree. A **phase** is the group of rules that `run_phase` runs in a single post-order walk. Each rule stays its own class in its own module, and a rule is still a complete transformer: `rule.visit(tree)` runs it alone, which is exactly a phase of one rule.

The hooks a rule may define:

| Hook                                                      | Runs                                          |
|-----------------------------------------------------------|-----------------------------------------------|
| `enter_<Class>(node)`                                     | Before the node's children are visited        |
| `leave_<Class>(node)`                                     | After them; returns the node that replaces it |
| `enter_function_body(node)` / `leave_function_body(node)` | Around the `body` of a `def` or `async def`   |

`leave_<Class>` returns the node itself when nothing changes. The decorators, defaults and annotations of a function are visited outside the `function_body` pair, in the enclosing scope.

A rule class must not define `visit_*` methods (`Rule.__init_subclass__` raises `TypeError`): they are generated from the hooks. At every node the `enter` hooks of the rules run in the phase's order, then every rule visits the children, then the `leave` hooks run in the same order, each one receiving the node the previous rule's hook returned.

**Why a phase must equal the separate passes.** A phase is only an optimization of running its rules one after the other, so its output has to be the same tree. In a phase, a rule's `leave` hook at a node sees children that every rule of the phase has already processed, while in separate passes it would see children processed by the rules before it only. Two properties make the two agree, and the rules of a phase are chosen for them:

- what a rule emits contains nothing that a later rule of the phase rewrites, because in the phase the later rule never visits it;
- a rule's decision at a node does not depend on whether a *later* rule has already rewritten the node's children, because in the phase it has.

The comment above the `Expression` phase in `pipeline.py` records why these hold for `SafeConvert`, `SafeDivision` and `FloatTolerance`: none of them emits what a later one rewrites, and the one decision that reads a child a later rule may already have rewritten (the operand types `SafeDivision` reads) reads the same type either way. When you change one of these rules, or want to add a rule to a phase, check both properties against the other rules.

**The sequential mode.** `PYNE_AST_SEQUENTIAL=1` makes `run_steps` run the rules of every phase as separate passes (`rule.visit(tree)` one after the other) instead of one `run_phase` call. That is the reference a phase must reproduce. To check a change, dump the emission with and without the variable (see [Debugging](#debugging-the-transformation)) and compare; the two outputs must be identical. With `PYNE_AST_TIMING=1` the sequential mode reports one time per rule, the fused mode one time per phase. The variable is read once, when `pipeline.py` is imported.

Rules whose `user_code_only` flag is set are left out of the phase for pynecore's own lib modules in both modes.

## The Slot Layout

The state-related transformers (Series, Persistent, Function Isolation) share one **module layout** (`ModuleLayout` in `transformers/slot_layout.py`): a table that assigns a **slot index** to every piece of per-instance state — persistent variables, series buffers, and the state of isolated call sites. The lowering half creates it once per module and passes it through `PipelineContext.slot_layout`. At the end of the chain `ApplyLayout` emits this table into the module as a plain dict constant, and every state-carrying function gets:

- a hidden first parameter (`__state__`) that receives its **state vector** — a plain Python list whose slots are addressed with literal int indexes, and
- a `__pyne_layout__` attribute describing how to build such a vector (initial values, series slots, `varip` slots, child call sites).

```python
__pyne_slot_layout__ = {'main': {'init': (0.0,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}}

def main(__state__):
    __state__[0] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

The runtime side of this scheme (who creates the state vectors and when, and the full list of layout keys) is described on the [Function Isolation](./function-isolation.md) page.

`_lower_tree` returns the allocated layout next to the tree. The `__pyne_slot_layout__` literal is one consumer's materialization of it; a consumer that builds the state vector itself takes the object and runs the pipeline with `emit_layout=False`, which turns `ApplyLayout` into a no-op.

## Analysis Steps

### ImportLifter

The Import Lifter moves function-level `pynecore.lib` imports to module level.

**Original code:**
```python
def main():
    from pynecore.lib.ta import sma
    result = sma(close, 14)
```

**Transformed code:**
```python
from pynecore.lib.ta import sma

def main():
    result = sma(close, 14)
```

Key aspects:
- Lifts the `from pynecore.lib[.x] import ...` statements that stand directly in a function body; other imports stay where they are
- The lifted statements go after the module docstring
- Prevents duplicate imports

### TypeCheckingStripper

Removes `if TYPE_CHECKING:` blocks and the `TYPE_CHECKING` import itself; a statement's `else` body is what runs, so it is kept in the statement's place. These blocks carry IDE-only type hints (casts, re-annotations) that have no runtime role, so stripping them keeps the transformed module free of dead code. It honors an alias (`from typing import TYPE_CHECKING as TC`) and the `typing.TYPE_CHECKING` spelling. Any other read of the removed flag (`flag = TYPE_CHECKING`) becomes the constant `False`, its runtime value, unless the module binds that name itself.

### TypeErasure

`typing.cast(T, x)` exists for the static type checker only: at runtime it returns `x` unchanged, but it is still a Python-level function call. Inside a builtin that a script calls millions of times, that call costs more than the arithmetic around it. The pass rewrites the call to its value operand:

```python
b = cast(float, base)      # before
b = base                   # after
```

An `if TYPE_CHECKING:` statement is resolved the same way. The name is `False` at runtime, so the statement is replaced by its `else` body, or dropped when it has none:

```python
if TYPE_CHECKING:          # before
    import orjson as json
else:
    import json

import json                # after
```

In the pipeline `TypeCheckingStripper` stands in front and has already resolved such statements; the `TYPE_CHECKING` form matters for the plain modules described below.

A rewrite happens only when it provably changes nothing:

- the name (`cast`, `TYPE_CHECKING`) is bound by a module-level `from typing import ... [as name]` or `import typing [as name]`, and that name is bound nowhere else in the module (no assignment, parameter, definition, other import, `except ... as`, pattern capture, `global` or `del`). A module that rebinds the name keeps every call through it, so `ctypes.cast` or a local `cast` helper is never touched;
- the call has exactly two positional operands, no keyword and no star operand;
- the type operand is a plain type expression (names, attributes, subscripts, constants, tuples, lists and `|` unions of those), so dropping its evaluation loses no effect;
- an `if` / `elif` is resolved only when its whole test is the trusted name; a compound test (`TYPE_CHECKING or x`) is left as written.

The import statement stays in place.

This is the one pass that also runs over PyneCore's **own plain modules** (`pynecore.lib.math`, `pynecore.lib.array`, `pynecore.core.series`, ...), which is where the casts on the hot path live. Such a module gets this pass and nothing else. Its bytecode carries a `__pyne_type_erased__` certificate paired with a digest of the pass, so a cached `.pyc` compiled without the import hook (`pip`'s post-install `compileall`, an IDE) is detected and recompiled, the same way a transformed `@pyne` module is. A package module that contains neither `cast(` nor `TYPE_CHECKING` is never parsed at all. Modules outside the `pynecore` package are left untouched unless they are `@pyne` code.

### BuiltinShadow

The Builtin Shadow transformer resolves workdir-library imports whose alias shadows a built-in namespace.

On TradingView, an `import TradingView/ta/7 as ta` does **not** hide the built-in `ta.*` namespace: Pine resolves `ta.x` against the library's exports first and falls back to the built-in namespace for everything the library does not export. So `ta.valuewhen(...)` keeps working even though the `TradingView/ta` library has no `valuewhen` export (TV's own library deliberately names its functions `ema2`/`atr2`/`rma2` to avoid colliding with the built-ins). A workdir library is imported as `import lib.tv.ta.v7 as ta`, which would route *every* access to the library module — and break the shadowed built-in accesses.

This transformer rewrites only the accesses the library cannot serve to the canonical built-in form, so the downstream transformers (import normalizer, module properties, isolation, series) handle them like any other built-in reference.

**Original code (workdir-library import):**
```python
import lib.tv.ta.v7 as ta

a = ta.valuewhen(cond, src, 0)   # not exported by the library
b = ta.t3(src, length)           # exported by the library
```

**Transformed code:**
```python
import lib.tv.ta.v7 as ta

a = lib.ta.valuewhen(cond, src, 0)   # falls back to the built-in ta namespace
b = ta.t3(src, length)               # served by the library, left untouched
```

Key aspects:
- Only aliases that shadow a known built-in namespace are considered (the namespace must exist in `module_properties.json`, with `str` mapping to the `string` module); a non-shadowing alias like `as myta` is left untouched
- Library membership is **runtime** knowledge: the library module is imported at transform time — the same pattern function-isolation callee resolution already uses. Its `__all__` is the membership test (matching TV, where non-exported names are unreachable through the alias), with a `hasattr` fallback for hand-written libraries that have no `__all__`
- Members the library serves stay on the library; members it cannot serve but the built-in namespace can are rewritten to `lib.<namespace>.<member>` (the nested-key check also covers sub-namespaces like `strategy.commission`)
- Runs **before** the Import Normalizer so the rewritten `lib.<namespace>.<member>` chains get their `from pynecore import lib` and `import pynecore.lib.<namespace>` statements added there
- A function parameter named like the alias masks the fallback inside that scope only
- If the library cannot be imported, the alias is left untouched and the script fails at its own import statement, exactly as before; a member in neither the library nor the built-in namespace is also left on the library to fail at runtime, as before

This keeps a script independent of the libraries it imports: whoever writes or generates it needs no knowledge of a library's contents, and all library-dependent resolution happens here at transform time, where the library is importable.

### ImportNormalizer

The Import Normalizer transforms all PyneCore imports to use a consistent format.

**Original code:**
```python
from pynecore.lib.ta import sma, ema
from pynecore.lib import plot, close

def main():
    plot(close)
    plot(sma(close, 14))
    plot(ema(close, 14))
```

**Transformed code:**
```python
from pynecore import lib
import pynecore.lib.ta

def main():
    lib.plot(lib.close)
    lib.plot(lib.ta.sma(lib.close, 14))
    lib.plot(lib.ta.ema(lib.close, 14))
```

Key aspects:
- Converts all lib-related imports to 'from pynecore import lib'
- Transforms variable references to use fully qualified names (lib.ta.sma)
- Maintains compatibility with wildcard imports: `from pynecore.lib... import *` expands to the module's `__all__` as recorded in the `star_exports` section of the generated `lib_types.json`, not to the live module's (see [Caching and Invalidation](#caching-and-invalidation))
- Ensures consistent import style across the codebase

This is very important to make lib level properties work like `close`, `open`, `high`, `low`, `volume`, etc.
If you would use this kind of import:
```python
a = close
```
That would not work, because the value would never be updated in the next bar.
However, after using the import normalizer, it will work:
```python
a = lib.close
```
Because the module level variable changed, and we access through the lib module object.

Most later steps match the normalized `lib.*` chains this step produces, which is why it stands this early.

### PlotScope

Requires plot declarations to execute directly in the script entry. The check covers `plot`, `plotshape`, `plotchar`, `plotarrow`, `plotcandle`, `plotbar`, `hline`, `fill`, `bgcolor`, `barcolor` and `alertcondition`. Each call must stand unconditionally in the direct body of the module-level `main`, before any statement that can return early; a call under an `if`, loop, `try`, `with`, `match`, boolean operator, conditional expression or comprehension is rejected with a `SyntaxError`. Plot values and color arguments may still be conditional: the declaration itself must execute on every completed bar. A plot function may be bound to one static alias, but it cannot escape as a callback or container value, because its eventual call scope would no longer be verifiable. Functions named `__test_*` are skipped. The step takes the module source (`ctx.source`) for the error location.

### OuterWrite

An object a script creates at module level lives outside everything the runtime rolls back. A
`request.security` child re-runs `main()` on a developing higher-timeframe bar and discards the
result, `calc_on_order_fills` re-executes a bar once per fill, and a live intrabar tick re-executes
the bar per tick. All three restore the script's own slots (`Persistent`, `Series`) and none of them
can restore a plain Python object the script keeps for itself.

This pass rejects the direct forms with a `SyntaxError` when the script is loaded:

```python
STORE = array.new_float(0)

def main():
    array.push(STORE, close)  # SyntaxError: 'STORE' is modified inside a function
```

Rejected inside any function: any `global` statement, an assignment, augmented assignment or `del`
whose target chain is rooted at a module-level binding, a mutating method call
on one (`append`, `extend`, `pop`, `clear`, `sort`, `update`, ...), and a mutating `array.*` /
`matrix.*` / `map.*` builtin whose first argument is one. A local variable or a parameter of the same
spelling shadows the module-level name, so it is not the outer object.

Defining is free: only writing is rejected. A module-level binding whose value is a call
(`color.new(...)`), an enum member (`strategy.fixed`) or an exported-function proxy is unaffected,
and reading any of them from a function is unaffected as well. A write through an alias or a
parameter is not detected; the correct construct for state that must survive is `Persistent` /
`IBPersistent`.

Functions whose name starts with `__test_` are exempt — they are test-harness code rather than script
code.

The step reads the normalized `lib.*` chains, and it runs well before function isolation, whose per-call-site copies would otherwise report the same write once per copy.

### ExportCapture

Validates a library's exports before closure and state lowering. It applies to a module whose top-level `main` carries `script.library`. An exported function may use constants from the library's module / `main` scope, but not objects or per-bar values created there, even through a helper; state created inside the exported call remains valid and belongs to its caller. A violation raises a `SyntaxError`.

To prove an imported enum constant, the step reads the enum's declaration from the imported module's *source* (it never executes the module). The fingerprints of the files it consulted are baked into the module as `__pyne_capture_deps__`, so editing one of them invalidates the cached bytecode (see [Caching and Invalidation](#caching-and-invalidation)).

### SecurityDrawings

Rejects drawing creation inside the expression of `request.security` / `request.security_lower_tf` before the request is lowered. `label.new`, `label.copy`, `line.new`, `line.copy`, `box.new`, `box.copy`, `table.new`, `polyline.new` and `linefill.new` are refused with a `SyntaxError`, also when the call is reached through a variable or a function the expression uses. Empty drawing fields are not forbidden. The step returns at once for a module with no `security` / `security_lower_tf` attribute.

### ConstFold

TradingView folds constant subtrees at parse time with fdlibm (`StrictMath`) transcendentals and embeds the result with a 16-decimal-place half-even cap, while runtime series-fed calls of the same functions go through the Intel-LIBM intrinsics (`lib.math` / `core.pine_math`). This step replays the parse-time side.

```python
a = lib.math.sqrt(2.0) * 3.0          # before
c = lib.close * 2.0 + lib.math.pi

a = 4.242640687119286                 # after
c = lib.close * 2.0 + 3.141592653589793
```

Key aspects:
- Constants propagate through single names in straight-line order; control flow conservatively kills the names it assigns. A `Persistent` that is stored more than once never counts as constant
- The folded surface is `+`, `-`, `*`, `/`, the `lib.math` constants (`pi`, `e`, `phi`, `rphi`), fdlibm `sin`/`cos`/`exp`/`asin`/`acos`/`atan` and the exact `sqrt`, `abs`, `floor`, `ceil`, `min`, `max`, `avg`, `sign`, `round`, `todegrees`, `toradians`. Anything else (`pow`, `log`, `tan`, `random`, ...) stays in the code and evaluates at runtime
- Each maximal constant subtree is replaced by its quantized literal (`quantize_embed`); the cap applies once, where the value is embedded
- A numeric module-level binding that is never rebound or written through `global` / `nonlocal` also folds inside functions; parameters and local bindings shadow it
- The pass needs the `lib.math.*` chains from `ImportNormalizer`, and it skips pynecore's own lib modules, which must keep their raw expressions
- The fdlibm and `pine_math` sources take part in the pipeline hash, because the folded values are baked into the emission

### DynamicDefault

Rewrites function-parameter defaults that reference per-bar runtime state (any `lib.*` expression, `lib.hl2`, `lib.close`, ...) so they are evaluated per call instead of at `def` time. A Python default freezes one value, and an anchored call site keeps reusing the first bar's closure, so the default has to be resolved inside the body.

**Original code:**
```python
def ao(source: float = lib.hl2, shortLength: int = 5):
    ...
```

**Transformed code:**
```python
def ao(source: float = __dyn_default__, shortLength: int = 5):
    if source is __dyn_default__:
        source = lib.hl2
    ...
```

Key aspects:
- Only defaults containing a `lib` reference are rewritten; plain constants keep the def-time path
- Script entry points (`@lib.script.indicator/strategy/library`) are skipped: their defaults are `input.*()` calls consumed by the input machinery at definition time
- A UDT field defaulting to a bool na is lowered to a `dataclasses.field(default_factory=...)` so each construction builds the na under the mode of the running script; only module-level classes are lowered
- The moved expressions must still get series slots and call-site anchors, so the step stands before the series and isolation passes

### InlineSeriesHoist

Pine evaluates a history-referenced expression (`expr[n]`) on every bar its statement executes, even when the `[n]` sits in a ternary branch or in a short-circuited `and` / `or` operand. PyneComp compiles such references to `inline_series(expr, n)`, whose per-anchor buffer advances only when the call site is reached, so inside a lazy position it would return stale history after skipped bars. This pass hoists every `inline_series(...)` call found in a lazy expression position into a temporary assigned immediately before the enclosing statement:

```python
y = a if cond else inline_series(expr, 1)      # before

__hist_0__ = inline_series(expr, 1)            # after
y = a if cond else __hist_0__
```

The assignment stays in the same statement list, so block-level conditional execution (`if` bodies) keeps its documented gap semantics and loop bodies keep their per-iteration frequency. `while` tests are left untouched (a hoist above the loop would freeze them), and lambdas and comprehensions are not descended into. The hoisted statements are the anchorable call sites for the isolation step, so the step stands in front of it.

### PineTruthiness

Gives the bool contexts of the script's own code TradingView's tolerant float-to-bool conversion. Pine treats a float as true only when it is farther than `EPSILON` (1e-10) from zero; Python treats every non-zero float as true, which turns arithmetic residue into a signal TradingView never draws. Every bool context (`if`, `while`, `?:`, `and`, `or`, `not`) is rewritten to the same inline form:

```python
if x:      # before

if (-1e-10 > x or x > 1e-10) if x.__class__ is float else x:   # after
```

Key aspects:
- The type guard (`x.__class__ is float`) keeps the rewrite honest: only a real `float` takes the tolerant branch, so ints stay exact, the `NA` object keeps its own false `__bool__`, and strings, colors and object references are handed back untouched
- An operand that is not a plain name or constant is bound once with a walrus inside the guard, so its side effects run once and in source order
- Expressions that are already bools (comparisons, `and` / `or` / `not`, bool constants) are left alone
- `and` / `or` convert their operands rather than their result, since Python's operators yield an operand and Pine's yield a bool
- The bounds it emits are marked `pine_exact`, so the comparison rewrite at the end of the lowering half leaves them at exactly one `EPSILON`
- It stands early so the `if` statements later passes emit for their own bookkeeping (a persistent's lazy-init flag) keep their plain test

### SecurityInstantiation

In Pine every call of a user function creates a separate instance, so a `request.security()` inside a function called from N sites is N distinct data requests. `SecurityTransformer` allocates one security id per *syntactic* call, so without this pass N call sites would silently share the first call's binding. The step restores Pine's instantiation semantics statically: a function whose subtree contains a `request.security[_lower_tf]` call (or a direct call to another such function) is cloned per direct-name call site, each clone is a full deep copy inserted after the original, and exactly one call site is rewritten to each clone.

Functions keep the shared-context behavior when they are recursive, decorated, referenced outside a direct-call position (aliases, callbacks, stores), defined more than once at the same level, or called only through attribute-style call sites (methods, cross-module library calls). See [request.security() Internals](./request-security-internals.md).

### Security

Rewrites each `lib.request.security()` / `lib.request.security_lower_tf()` call into the signal / write / read / wait protocol the multiprocessing runtime executes, and builds the module-level `__security_contexts__` registry (one entry per context: symbol, timeframe, lookahead, group, `closed_shift` candidate, ...). The sections [AST Transformation](./request-security-internals.md#ast-transformation) and [Skipping Developing Rounds](./request-security-internals.md#skipping-developing-rounds) of the internals page describe the emitted code and the flags. The step returns at once for a module with no security call.

### PersistentSeries

The PersistentSeries transformer converts the combined PersistentSeries type into separate Persistent and Series declarations.

**Original code:**
```python
ps: PersistentSeries[float] = 1
ps += 1
```

**Transformed code:**
```python
p: Persistent[float] = 1
s: Series[float] = p
s += 1
```

Key aspects:
- Splits PersistentSeries declarations into two separate declarations
- Must be applied before both Persistent and Series transformers

This makes easier to declare variables are both persistent and series.

### LibrarySeries

The Library Series transformer prepares library Series variables (like close, open, high, etc.) for proper handling by the Series transformer: every scope that indexes a library value gets a local Series anchor for it.

**Original code:**
```python
def main():
    a = lib.close[1]

    def nested():
        return lib.high[1]
    result = nested()
    print(a, result)
```

**Transformed code (after the Series transformer has assigned the slots):**
```python
__pyne_slot_layout__ = {'main': {'init': (None, None), 'series': ((0, None, 'float'), (1, None, 'float')), 'varip': (), 'children': (), 'names': ('__lib·close', '__lib·high')}}

def main(__state·main__):
    __lib·close = __state·main__[0].add(lib.close)
    __lib·high = __state·main__[1].add(lib.high)
    a = __state·main__[0][1]

    def nested():
        return __state·main__[1][1]
    result = nested()
    print(a, result)
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Creates local Series variables for library Series in each scope that needs them
- Uses Unicode middle dot (·) as separator to prevent name collisions
- The buffers anchor in the outermost function that uses them; nested functions reach them through the parent's state vector
- Prepares variables for Series transformer processing

**Collision Prevention**: The transformer uses `__lib·` prefix with Unicode middle dot separators to prevent naming conflicts. For example:
- `mylib.bar.foo` becomes `__lib·mylib·bar·foo`
- `mylib.bar_foo` becomes `__lib·mylib·bar_foo`

This ensures that hierarchical module names cannot collide with underscore-separated names.

If you import a variable from a library, it does not know if it is a series or not. But if you use indexing (subscription) on it, it should initialize it as a series. This is needed, because the AST transformer does not know anything about the other files just the one it is currently transforming.

### ModuleProperty

The Module Property transformer handles attributes that should be called as functions based on
the generated `module_properties.json` registry.

**Original code:**
```python
t = lib.time
bar_index = lib.bar_index
plot(close, "Close")
d = dayofweek
```

**Transformed code:**
```python
t = lib.time()
bar_index = lib.bar_index
lib.plot.plot(lib.close, "Close")
d = lib.dayofweek.dayofweek()
```

Key aspects:
- The registry (`module_properties.json`, generated from the lib source by
  `scripts/module_property_collector.py`) determines which attributes are properties
- Automatically adds parentheses for property calls; explicit calls are left untouched
- Normal attributes (variables, constants, function references) stay plain attribute reads
- Calls and promoted bare reads of function-and-namespace modules (`plot`, `hline`, `alert`,
  `dayofweek`, `strategy.opentrades`, `strategy.closedtrades`) are routed to the module's
  self-named function
- Unknown names on known `pynecore.lib` modules raise at transform time — this catches typos
  and a stale registry early (the test suite keeps the committed registry current)
- Unknown module paths (user `lib.*` workdir libraries) and `_`-prefixed names are plain reads

### TaVariableHoist

TradingView keeps one engine-level series per stateful `ta` builtin variable, so a read inside an `if` returns the same value as an unconditional read. PyneCore's per-call-site machines would advance only when the gated call runs. This step evaluates the referenced variables once per bar at the top of `main`.

The variables are `nvi`, `obv`, `pvi`, `pvt`, `wad` (engine-global: replaced in every function of the module) and `vwap`, `accdist` (replaced only directly in `main`'s body; inside any other function they keep their own per-call-site machine). `iii`, `wvad` and `tr` are stateless per bar and stay as they are.

```python
global __ta·nvi                 # prologue added to main
__ta·nvi = lib.ta.nvi()
...
x = __ta·nvi                    # every zero-argument read of ta.nvi
```

Key aspects:
- Rewrites zero-argument calls only; `ta.vwap(src)` and the other function forms keep their own Pine semantics
- The prologue call is an ordinary call site for the series, persistent and isolation passes, so the variable's state advances exactly once per bar with a stable identity
- A module without a top-level `main` is left alone
- In library modules (`main` decorated with `script.library`), reads in `main` and in functions nested in it are rewired like in a script; `@export` functions and module-level functions keep their gated per-call-site machines
- The step follows `ModuleProperty`, because a bare `ta.nvi` read is a call only after it

### ClosureArguments

The Closure Arguments transformer converts closure variables in inner functions to explicit function arguments, enabling proper function isolation.

**Original code:**
```python
@lib.script.indicator("Test")
def main():
    length = 14
    multiplier = 2.0

    def calculate(offset=0):
        return lib.ta.sma(lib.close, length) * multiplier + offset

    return calculate() + calculate(10)
```

**Transformed code:**
```python
@lib.script.indicator("Test")
def main():
    length = 14
    multiplier = 2.0

    def calculate(length: int, multiplier: float, offset=0):
        return lib.ta.sma(lib.close, length) * multiplier + offset

    return calculate(length, multiplier) + calculate(length, multiplier, 10)
```

Key aspects:
- Adds closure variables as function parameters at the beginning of parameter list
- Preserves type annotations from original variable declarations
- Updates all function calls to pass closure variables as arguments
- Only processes functions inside a `main` decorated with `lib.script.indicator` or `lib.script.strategy` (or the `script.indicator` / `script.strategy` spellings); a library `main` is not processed
- Maintains proper scope isolation for nested functions
- Prepares functions for the Function Isolation transformer

### PineType

The type pass infers Pine's static types for the module and stamps them on the nodes. It changes nothing about the tree: it clones no function and builds no specialization. A generic helper is analysed once per call-site context and the answers are kept apart in the type table, while the tree keeps one body carrying the join of what the contexts found. The table is attached to the module node (`_pine_types`, read with `module_table`).

It stands at the last point where the tree still looks like Pine:

- the annotations are intact (the series pass rewrites the parameter ones and consumes the declaration ones into the slot layout),
- the `/` is still a `BinOp` (safe division wraps it into a call),
- the security-bearing functions are already instantiated per call site, so each of them is reached by exactly one caller.

Key aspects:
- A call into an **imported** module is typed from the interface that module publishes, never from the call site. Every interface consulted is recorded in the table's `deps`, which the loader bakes into the bytecode so a moved signature invalidates its dependents (see [Caching and Invalidation](#caching-and-invalidation)). `PipelineContext.analyse` (`compile_interface`) is how the interface is found when no transform of the process published it: off the dependency's own `.pyc`, or by transforming the dependency into one
- It decides the machine of every window call (`ta.highest` and its kin) from the Pine qualifier of the length. That happens in user code only, not in pynecore's own lib modules, whose machines *are* those calls (`PineTypeTransformer(qualify_windows=ctx.user_code)`)
- Later steps read the stamped types: `SecurityDefault`, `FunctionIsolation` (overload pins and per-instance vectors), `SafeConvert` and `SafeDivision`
- For `@pyne edge` modules the diagnostics of the table are merged with the edge gate's findings after the analysis half; they are errors under `PYNE_EDGE_STRICT=1`

### SecuritySlice

A security child process re-runs the script's whole `main()` on every bar of its own feed, although only one `__sec_write__` block produces the value the chart waits for. This step computes the backward slice of `main()` for each static security context and emits it as an ordinary module-level function (`__sec_main_<n>__`). Contexts in the same `group` (those resolving to one feed) share one clone. The clone is recorded in `__security_contexts__[sid]['slice_main']`, and the child runs it instead of `main()`. Chart and child load the same bytecode, so the clones live in the same module as `main`.

Key aspects:
- The slice is a conservative over-approximation: every uncertainty keeps a statement, and a construct the pass cannot classify drops the optimization for the module (the child then runs `main()`)
- A clone is also where a guarded write is made unconditional, so a context the slice does not serve still gets a clone whenever its write is guarded
- `PYNE_NO_SECURITY_SLICE=1` switches slicing off; the flag is mixed into the pipeline hash
- The step is last of the analysis half: the clones must see the tree every earlier step produced (the hoisted `ta` assignments among them), and the lowering half then lays them out like any other function with their own slot layout

## Lowering Steps

### SecurityDefault

Resolves the default of a missing `__sec_read__` result after type inference. A bool result (or a bool element of a tuple result) gets a call to the bool-na factory, which reads the running script's mode, so a library's defaults follow its caller. Other result types keep the default the security step supplied. The step returns at once for a module without a `__security_contexts__` registry.

### ExportOnce

A library's `main` is a per-bar entry point, and Pine's export surface is emitted as `@export`-decorated definitions inside it. Executed as written, every bar would allocate a fresh function object per export and run the decorator over it, although the call sites keep using the bar-0 object. This step defines the exports once per run:

- the definitions run under a latch that is a `Persistent` slot of `main`, never a module global: a second run in the same process hands `main` a new state vector, so the exports are defined again
- the definitions bind through `global`, so a read on a later bar resolves to the module-level `Exported` proxy
- an export whose name `main` also binds somewhere else is left out of the latch and keeps being rebuilt per bar

The latch is a Persistent slot, so the step must run before the slot transformers, and the guarded definitions must reach them in their final position.

### UnusedSeriesDetector

The Unused Series Detector optimizes performance by removing Series annotations from variables that are never indexed with the subscript operator.

**Original code:**
```python
def main():
    # This variable is never indexed - can be optimized
    s: Series[float] = close
    t: Series[float] = close * 2

    def f(source: Series[float], m = 1.0):
        # This parameter IS indexed - must keep Series annotation
        return source[1] * m

    r = f(t, 2.0)
    plot(s)
    plot(r)
```

**Transformed code:**
```python
def main():
    # Series annotation removed since s is never indexed in main scope
    s: float = close
    t: float = close * 2

    def f(source: Series[float], m = 1.0):
        # The Series annotation stays: source is indexed in this scope
        return source[1] * m

    r = f(t, 2.0)
    plot(s)
    plot(r)
```

Key aspects:
- Uses scope-aware analysis to track variable usage independently in each function scope
- Distinguishes between variables with the same name in different scopes (e.g., closure vs parameter); a subscript on a name that belongs to an enclosing scope counts as an index of that variable
- Only removes Series annotations from variables that are never used with subscript syntax `[index]`
- Runs before SeriesTransformer to prevent unnecessary SeriesImpl creation
- The detection walk only collects; `optimize` edits the tree
- Avoids creating a `SeriesImpl` buffer for variables that are only used for simple arithmetic

### Series

The Series transformer converts Series annotated variables into operations on a `SeriesImpl` instance (a circular buffer) living in a slot of the function's state vector.

**Original code:**
```python
from pynecore import Series
from pynecore.lib import close

def main():
    s: Series[float] = close
    s += 1
    previous = s[1]
    print(previous)
```

**Transformed code:**
```python
from pynecore import lib
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': ((0, None, 'float'),), 'varip': (), 'children': (), 'names': ('s',)}}

def main(__state__):
    s = __state__[0].add(lib.close)
    s = __state__[0].set(s + 1)
    previous = __state__[0][1]
    print(previous)
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Allocates a series slot in the function's state vector for each Series variable; the runtime puts a fresh `SeriesImpl` into these slots when an instance is created
- Converts the declaration to an `add()` (push the bar's value) and assignments to `set()` operations
- Redirects indexing operations to the slot (`s[1]` becomes `__state__[0][1]`)
- Statement-position `lib.max_bars_back(s, n)` calls become assignments to the slot's `max_bars_back` attribute
- A `Series`-annotated parameter loses the wrapper in its annotation and gets an `add()` prepended to the body
- Each function instance gets its own buffers, because each instance has its own state vector

### VerifyClosedShift

`SecurityTransformer` decides the `closed_shift` flag on the *source* shape of the expression (the value of a `lookahead_on` context cannot change inside a higher-timeframe period, so the chart may skip the developing rounds). That is a re-derivation of what the lowering establishes exactly. After the series step a runtime history reference takes only two forms: `<state param>[slot][k]` with a constant `k >= 1`, and `inline_series(expr, k)`. This step checks the lowered `__sec_write__` expression against those forms and clears the flag, for the context and for every member of its merge group, when the expression is not built from them. It never sets the flag to true. It resolves a name only when the binding is beyond doubt (a single plain assignment in the same statement list, in front of the write); everything less definite stays rejected.

### Persistent

The Persistent transformer converts variables with Persistent type annotation to slots of the function's state vector, so they maintain their values across function calls.

**Original code:**
```python
p: Persistent[float] = 0
p += 1
```

**Transformed code:**
```python
__pyne_slot_layout__ = {'main': {'init': (0.0,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}}

def main(__state__):
    __state__[0] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Allocates a slot with the initial value in the layout's `init` tuple; literal initializers are baked in (a float-typed `0` becomes `0.0`), non-literal initializers get a lazy init-flag companion slot that triggers the assignment on the instance's first call
- Rewrites every read and write of the variable to the slot (`__state__[0]`)
- `IBPersistent` (varip) variables get their slot listed in the layout's `varip` tuple, which excludes them from the `var` rollback on intra-bar re-execution
- Slot reads/writes are plain list indexing with literal indexes — the fastest state access Python offers

**Accumulation**: The `+=` operator stays a plain augmented assignment on the slot, so a running sum accumulates naively. That is deliberate: TradingView accumulates the same way (measured on `ta.cum` and every volume accumulator), and error compensation — a Kahan sum, for instance — would produce a mathematically better sum that no longer matches the reference.

**Important Note**: The state-related transformers use the Unicode character `·` (middle dot, U+00B7) as the internal scope separator in slot names and call-site identifiers (e.g. `main·t·0`). This prevents conflicts when function names contain underscores. Avoid using the `·` character in function or variable names to prevent conflicts with the internal scoping system.

### CallInline

A wrapper like `math.abs` is a few nanoseconds of work behind a Python call that costs tens of them, and a rolling-window script reaches such wrappers tens of millions of times over a run. This pass replaces an allow-listed call with the wrapper's own body, written out as a single expression.

**Original code:**
```python
from pynecore.lib import math, close

def main():
    return math.abs(close - 1.0)
```

**Transformed code:**
```python
from pynecore.core.inline_support import na_float as __inl·na_float__, py_builtins as __inl·py_builtins__

def main():
    return __inl·na_float__ if not (__inl1·__ := lib.close - 1.0) == __inl1·__ else __inl·py_builtins__.abs(__inl1·__)
```

Key aspects:
- The expression is **derived from the wrapper's own source AST**, never hand-written, and only from a restricted body shape: a docstring, `if <test>: return <expr>` guards, name aliases, and one final `return <expr>`. The same operations run in the same order on the same operands, so the result is the same double, the same `na` object and the same exception. A body that grows past that shape simply stops being inlinable.
- `math.max` / `math.min` are varargs and get one written expansion for a fixed positional arity, guarded by a structural check of their real bodies.
- Arguments are classified by what re-evaluating them costs. A constant, a plain name and a slot read emitted by the Series/Persistent lowering (`__state__[7]`) run no user code; anything else must run exactly once and is bound with an assignment expression, or goes in directly when the body reads it once on a path that always runs.
- The call being replaced evaluated every argument before entering the body. When an impure argument is present and the body would read the raising arguments out of source order, all of them are forced in front as `x is x` probes, in source order. With no impure argument nothing is forced — the only observable difference would be which `NameError` an undefined name reports.
- A guard the call decides with a literal (`math.pow(x, 2)`) is folded away at transform time, and so are the operations that guard reads (`2 == 2`, `int(0)`) — by running the same operation on the same literal, never by reasoning about it. A `bool` does not count as a numeric literal.
- The counter of a `for` loop over `pine_range` is never na, so inside that loop the na guard a body runs on it (`i == i`, the index check of `array.get(a, i)`) folds away. The fold is proven per loop: nothing in the body may rebind the counter, and the enclosing function may not declare it `global` or `nonlocal`; it does not reach the loop's `else` branch or nested functions.
- Free names of a copied body resolve through `pynecore.core.inline_support`, not through the call site: a script's `math` is `pynecore.lib.math`, and a script may rebind `abs` or `float`.
- The callee must be **provably** the library function: its dotted path is resolved through the module's import map and compared by object identity. A name the module binds anywhere takes the whole base name out of the pass.
- Module level, class bodies, decorators, defaults, lambdas and comprehensions are skipped (the temporaries would bind in the wrong scope), as are keyword arguments, starred arguments and an unsupported arity. Leaving the call in place is always correct.
- Every comparison the pass emits **for itself** is marked exact, so the Float Tolerance rewrite leaves the copied raw `x == x` na tests alone — but never a comparison inside an argument, which is the user's own Pine code and still gets the tolerant rewrite.
- The inlined wrapper sources and `core/inline_support.py` take part in the pipeline digest, so editing a wrapper body or an anchor invalidates cached script bytecode.

### FunctionIsolation

The Function Isolation transformer ensures each function call site gets its own isolated state. The state of a callee instance lives in a dedicated **child slot of the caller's state vector**, assigned at transform time.

**Original code:**
```python
from pynecore.lib import ta, close

def main():
    print(ta.sma(close, 12))
```

**Transformed code:**
```python
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.instance_state import __resolve_slot__ as __resolve_slot·__
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': (), 'varip': (), 'children': ((0, 'main·lib.ta.sma·0', False),), 'names': ('main·lib.ta.sma·0',)}}

def main(__state__):
    print(lib.ta.sma(__st·__ if (__st·__ := __state__[0]) is not None else __resolve_slot·__(__state__, 0, lib.ta.sma), lib.close, 12))
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- The callee receives its own state vector as hidden first argument; after the first call it is a single list-index read
- Callees the transformer can prove stateful get this fast path; callees it cannot resolve at transform time go through a uniform binding path; stateless callees are called directly; builtins, types and module properties are left untouched
- The route of a `pynecore.lib` callee is read from the `routes` section of the generated `lib_types.json`; a callee of any other module is imported and inspected at transform time (see [Caching and Invalidation](#caching-and-invalidation))
- A call site inside a loop shares one callee instance across the iterations (TradingView keeps one state per call site, however often a loop executes it)
- Classification needs the var and series slots, so the step runs after Persistent and Series

The full routing logic, the loop emission and the runtime side are described on the [Function Isolation](./function-isolation.md) page.

### ScriptRequirements

Detects the broker capabilities a strategy script needs at compile time. It scans the module for calls to `strategy.entry`, `strategy.exit`, `strategy.order`, `strategy.close` and `strategy.close_all` and deduces from the keyword arguments present at each call site which `ScriptRequirements` flags the script needs. The detection is conservative: a keyword that is syntactically present counts as needed, even with an `na` value. The result is injected as the `_broker_requirements` keyword of the `@script.strategy(...)` decorator on `main`, so the `Script` object carries the requirements at runtime, and no second pass or side channel is needed to refuse an under-capable exchange before trading starts. A script without a `@script.strategy` decorator is left alone.

### Input

The Input transformer processes input parameters and adds necessary ID information.

**Original code:**
```python
@script.indicator
def main(source=lib.input.source(lib.close, "Source")):
    result = source * 2
```

**Transformed code:**
```python
from builtins import getattr as __pyne_getattr__

@script.indicator
def main(source=lib.input.source(lib.close, "Source", _id="source")):
    source = __pyne_getattr__(lib, source, lib.na)
    result = source * 2
```

Key aspects:
- Adds _id parameter to input calls
- Adds a `getattr` (imported under a reserved alias) for source inputs at the start of functions
- Enables proper input parameter resolution
- Handles source inputs specially
- Stands in front of the `Expression` phase: the source-input parameter it rebinds can be named `range`, which `SafeConvert`'s binding scan must see

### Expression phase

The three rules below run in one post-order traversal and equal running the three passes in this order (see [Fused Phases and the Sequential Mode](#fused-phases-and-the-sequential-mode)).

#### SafeConvert

A Pine `int` is a double at run time (see [Types — int](../reference/types.md#int)), so the
transformer has two jobs: lower the `float()`/`int()` casts to their Pine meaning, and truncate a
Pine int to a real Python `int` at the places where Python itself insists on one.

**Original code:**
```python
value = float(some_value)
number = int(another_value)
for i in range(lib.bar_index):
    total += weights[number + i]
```

**Transformed code:**
```python
from pynecore.core import safe_convert

value = safe_convert.safe_float(some_value)
number = safe_convert.safe_int(another_value)
for i in range(safe_convert.native_int(lib.bar_index)):
    total += weights[safe_convert.native_int(number + i)]
```

Key aspects:
- `float()` becomes `safe_float()`, `int()` becomes `safe_int()`: both keep `na` as `na` (a `nan`),
  and `safe_int()` returns the truncated value as a Pine int, i.e. a `float`
- Inside a `@pyne lib` module `int()` becomes `native_int()` instead: a lib computes its lengths,
  counts and ring indexes in native `int` and converts back only at its boundary
- A `range()` argument and the index (or slice bound) of a subscript are Python-native consumers:
  every one typed as a Pine int is wrapped in `native_int()`, a direct `int(x)` index becomes
  `native_int(x)` outright, and a folded literal such as `2.0` becomes `2`
- A `range()` loop counter, an `int` literal and a series buffer read (`Series.__getitem__`
  truncates on its own, and it is the hot loop) are left alone
- A module that binds its own `range` (`ta.range`, `array.range`) is not treated as using the builtin
- Only adds the import when a lowering was actually emitted

#### SafeDivision

The Safe Division transformer converts division operations to safe alternatives that handle division by zero like Pine Script.

**Original code:**
```python
def main():
    value = lib.close
    divisor = lib.open
    ratio = value / divisor
    half = divisor / 2
```

**Transformed code:**
```python
from pynecore.core import safe_convert

def main():
    value = lib.close
    divisor = lib.open
    ratio = __div1·__ if (__div1·__ := (value / (divisor or safe_convert.zero_divisor))) == __div1·__ else safe_convert.safe_div(value, divisor)
    half = __div2·__ if (__div2·__ := (divisor / 2)) == __div2·__ else safe_convert.safe_div(divisor, 2)
```

Key aspects:
- Every division whose operands are not both literals gets Pine's semantics from `safe_convert.safe_div`: division by zero answers `inf` / `-inf` / `nan` instead of raising, and a na operand answers na
- Inside a function body, a division of numeric-typed operands (per the type pass) is written out as an expression that runs the plain division first and calls `safe_div` only when that cannot be its result: a quotient that equals itself is exactly what `safe_div` returns, and a zero divisor is swapped for `zero_divisor`, whose reflected division answers `nan` and so lands in the fallback
- Operands are evaluated once and in source order; module level, class bodies, lambdas, comprehensions, decorators and default arguments keep the plain `safe_div` call
- A literal divided by a nonzero literal stays a plain division (in user code `ConstFold` has folded it already); a literal zero divisor keeps the call
- Only adds the import when a division was actually transformed

#### FloatTolerance

Gives the comparison operators TradingView's tolerant float semantics. Pine treats operands closer than `EPSILON` (1e-10) as equal. Every operator is rewritten into an arithmetic form over the difference, with the bound kept on the left:

```python
a <  b   ->  -1e-10 >  a - b
a >  b   ->   1e-10 <  a - b
a <= b   ->  a <= b or  1e-10 >= a - b
a >= b   ->  a >= b or -1e-10 <= a - b
a == b   ->  a == b or -1e-10 <= a - b <= 1e-10
a != b   ->  1e-10 < (d := a - b) or -1e-10 > d
```

Key aspects:
- The forms are written over the difference rather than `abs()`, so a na operand can never satisfy them: a native `nan` difference makes every form false, and an `NA` object propagates through the subtraction
- A script that keeps Pine's three-state bool (`na_bool`, i.e. v4/v5) answers `na` for a comparison with a na operand, so each rewritten comparison gets a tail that checks the operands for na; a two-state (v6) script pays nothing
- Integers keep exact semantics automatically (an int difference is 0 or at least 1). `==` and `!=` are guarded by a runtime type test because Pine also allows them on strings, colors and object references
- Operands that are not simple names or constants are bound once with a walrus, at the operand's first evaluated position
- Comparisons marked `pine_exact` (the bounds of `PineTruthiness`, the na tests of `CallInline`) are left alone
- Runs on user code only: pynecore's own lib modules implement the natively bit-exact builtins and use the raw `x != x` nan idiom, both of which the rewrite would break

### FinalizeCloneDefaults

A security clone (`__sec_main_<n>__`) keeps `main()`'s parameter list, and its `__defaults__` / `__kwdefaults__` are rebound from `main` at run time, so it receives the configured input values without evaluating their expressions again. The copied default expressions stay in the clone until `Input` has processed the signature; this step then replaces them with `None` placeholders. That is why it follows `Input`.

### ApplyLayout

Materializes the collected `ModuleLayout` into the module: it inserts the `__pyne_slot_layout__` dict after the import block, injects the hidden state parameter into every state-carrying function and appends the `__pyne_layout__` attach statement after each of them (a decorated definition gets an innermost `@__attach_layout__(...)` decorator instead, so the layout lands on the raw function). A function that contains nested definitions gets the scope-qualified name `__state·<scope>__` so nested functions can reach its state vector through a closure. With `emit_layout=False` the step does nothing and the caller keeps the `ModuleLayout` object.

### FixLocations

Fills the missing source locations of the synthetic nodes the steps emitted. `ast.fix_missing_locations` copies the parent's full span onto every location-less node, so a statement inserted into a function body would inherit the function's whole range, and CPython would map parts of the prologue onto the function's last source line (a breakpoint there fires on every function entry, mid-prologue). `fix_locations` (`transformers/locations.py`) gives single point anchors instead: a located node is never touched, a location-less statement anchors to the earliest surviving source location inside it (or the enclosing node's start), and every other synthetic node anchors to its statement's point. The loader locates its own statements (the sentinel, the dependency records, the bool-na prologue) afterwards.

## Example of Complete Transformation

Let's see a full example of how a simple Pyne code is transformed:

**Original Pyne Code:**
```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, ta, close, open, high, low, plot, color


@script.indicator("Example")
def main():
    # Persistent counter
    count: Persistent[int] = 0
    count += 1

    # Moving average calculation
    ma: Series[float] = ta.sma(close, 14)

    # Safe division that could cause division by zero
    range_ratio = (close - open) / (high - low)

    # Plot results
    plot(ma, "MA", color=color.blue)
    plot(count, "Count", color=color.red)
    plot(range_ratio, "Range Ratio", color=color.green)
```

**Transformed Code:**
```python
"""
@pyne
"""
from pynecore import lib
import pynecore.lib.color
import pynecore.lib.ta
from pynecore.core.instance_state import __resolve_slot__ as __resolve_slot·__
from pynecore.core import safe_convert
from pynecore.core.instance_state import __attach_layout__
__pyne_slot_layout__ = {'main': {'init': (0.0, None), 'series': (), 'varip': (), 'children': ((1, 'main·lib.ta.sma·0', False),), 'names': ('count', 'main·lib.ta.sma·0')}}

@lib.script.indicator('Example')
@__attach_layout__(__pyne_slot_layout__['main'])
def main(__state__):
    __state__[0] += 1
    ma: float = lib.ta.sma(__st·__ if (__st·__ := __state__[1]) is not None else __resolve_slot·__(__state__, 1, lib.ta.sma), lib.close, 14)
    range_ratio = __div3·__ if (__div3·__ := ((__div1·__ := (lib.close - lib.open)) / ((__div2·__ := (lib.high - lib.low)) or safe_convert.zero_divisor))) == __div3·__ else safe_convert.safe_div(__div1·__, __div2·__)
    lib.plot.plot(ma, 'MA', color=lib.color.blue)
    lib.plot.plot(__state__[0], 'Count', color=lib.color.red)
    lib.plot.plot(range_ratio, 'Range Ratio', color=lib.color.green)
```

Worth noting in the output:

- `count` became slot 0 of main's state vector (`init` starts with its initial value; the `int` is a Pine int, so the literal is the double `0.0`).
- `ma` lost its Series annotation (never indexed — Unused Series Detector), so no series slot was allocated for it.
- The `ta.sma` call got child slot 1: the first call creates the callee's state vector there, subsequent calls reuse it.
- The division is written out by Safe Division as the plain division with a `safe_div` fallback; each operand is evaluated once.
- Since `main` is decorated, the layout attach uses the `@__attach_layout__` decorator form (innermost position, so it tags the raw function before other decorators wrap it).
- The `plot(...)` calls were routed to the module's self-named function (`lib.plot.plot`) by the Module Property transformer and stay direct calls — `plot` is a function-and-namespace module.

This example demonstrates how the different transformers work together to convert a simple Pyne code into equivalent Python code that provides Pine Script-like behavior through PyneCore's runtime system.

## Performance Notes for Contributors

A module is transformed again whenever its cached bytecode is missing or stale (after every edit of the script or a library it imports, and after any change to the pipeline itself). The transform therefore sits in the loop of anyone editing a script, and the pipeline's passes each walk the whole module. Most of a pass's time is the walk, not the rewrite, so the rules below are about the walk.

- **Walk with `ast_walk`.** `transformers/ast_walk.py` holds the traversal and copy primitives every pass uses. They produce the results of their stdlib namesakes, faster:
  - `NodeVisitor` / `NodeTransformer` visit the same nodes in the same order and write back what the stdlib classes write back. The visitor method is resolved once per (visitor class, node class), and a node that has no child and no visitor method of its own is not entered at all. A subclass that overrides `visit` itself gets every node;
  - `walk` and `iter_child_nodes` replace `ast.walk` and `ast.iter_child_nodes`; `iter_descendants(node, stop)` walks the descendants in pre-order without entering the `stop` classes;
  - `walk_statements` and `iter_child_statements` yield only the statements, except handlers and match cases, without entering an expression. Use them when you look for a `def`, `class`, `global` or `import`, which can only stand in a statement list;
  - `fix_missing_locations` and `clone` replace `ast.fix_missing_locations` and `copy.deepcopy` for trees (`clone` keeps the memo protocol and copies the attributes passes stamp on nodes).
- **Decide with a cheap exact early exit.** Check one thing that is true of almost no module before doing a walk, and return the tree untouched. The existing passes show the shape: the reserved-name check returns for a pure-ASCII module; `ImportLifter` returns when no function body holds a `pynecore.lib` import; `SecurityDrawings`, `Security` (via `has_security_call`) and `SecurityDefault` return when the module has no security request or `__security_contexts__`; `TypeCheckingStripper` returns for a module with neither of the two forms it rewrites; the loader parses a plain package module for `TypeErasure` only if its source contains `cast(` or `TYPE_CHECKING`. The test must be exact: a false negative silently skips a rewrite.
- **Do not re-walk subtrees.** A pass that walks the subtree of every statement, or of every function, again is quadratic in nesting. Collect what you need in one walk of the module (`ConstFold` gathers its per-scope facts in a single pass in `_ModuleFacts`; `PlotScope`, `OuterWrite`, `SecurityDrawings` and `ExportCapture` build one scope table in a single `collect`) and answer the questions from that. Where a per-node question repeats, memoize it for the module (`ConstFold._bindings_memo`).
- **Cache per module, and key a process-wide cache so it cannot go stale.** State derived from one module belongs on the transformer instance (the pipeline builds one per module). A process-level cache needs a key that covers everything the value depends on: the dispatch tables of `ast_walk` are keyed by (visitor class, node class), the phase walkers by the tuple of rule classes, and the interface registry by resolved path plus the file fingerprint and the dependency closure.
- **Prefer static information to live objects.** The transform has the module's source and the sources of the modules it imports. The interface a dependency's `.pyc` carries, or the transform that writes that `.pyc`, gives a dependency's signatures without executing it, and `ExportCapture` reads an imported enum's declaration from source instead of executing the module. Two passes do import at transform time because the answer is runtime knowledge: `BuiltinShadow` (the library's `__all__`) and `FunctionIsolation` (the state layout of a cross-module callee outside `pynecore.lib`; the lib's own facts come from the generated registry, see [Caching and Invalidation](#caching-and-invalidation)). Do not add a third where the source says the same thing, since an import costs time and needs the module to be importable while it is being transformed.
- **Fuse passes that are only a walk.** A pass whose work is a handful of node-local rewrites can be written as a rule and join a phase, which removes a whole traversal. It has to satisfy the two conditions in [Fused Phases and the Sequential Mode](#fused-phases-and-the-sequential-mode).
- **Mind import order.** The transformers package is itself loaded through the import hook, so a transformer module that imports `pynecore.lib` at module level would re-enter a half-initialized package. `ConstFold` fills its `lib.math` table on first use for that reason, and the loader imports the pipeline lazily.
- **Measure per step.** `PYNE_AST_TIMING=1` prints the wall time of every step, so a new pass shows what it costs (see [Debugging](#debugging-the-transformation)).

## Caching and Invalidation

CPython validates a cached `.pyc` only against its source file's mtime and size. That cannot tell a transformed module from one compiled without the import hook (`pip`'s post-install `compileall`, an IDE) or one left over by an older pipeline, and it cannot see that a module's types were derived from other modules. `PyneLoader.get_code` therefore adds its own checks.

**The sentinel.** Every transformed module carries the assignment `__pyne_transformed__ = '<pipeline hash>'`, placed after the docstring and any `from __future__` imports. It is a plain constant, so it is marshalled into the `.pyc`. `get_code` accepts cached bytecode only if the sentinel name is in the code object's `co_names` and the current hash is in its `co_consts`. Otherwise it drops the `.pyc` (the path comes from `importlib.util.cache_from_source`, so `sys.pycache_prefix` and `-O` levels are honored) and retransforms. If the stale file cannot be removed (read-only cache), the module is compiled straight from source for that load and the cache is skipped.

**The pipeline hash** (`_get_transform_pipeline_hash`) is a SHA-256 over the *contents* of the files the emission depends on, so it does not depend on mtimes, cache markers or install locations. For each file, sorted by file name, the name and the bytes are hashed:

- `core/import_hook.py`, which drives the halves;
- every regular file directly in `transformers/`, `pipeline.py` among them (which pins the step order) and the data files the passes read (`module_properties.json`, `lib_types.json`, `edge_rules.json`);
- `core/pine_compare.py`, whose comparison tolerance `FloatTolerance` bakes into the emission as a literal;
- `lib/math.py`, `lib/array.py` and `core/inline_support.py`, whose function bodies `CallInline` copies into call sites;
- `core/fdlibm.py` and `core/pine_math.py`, whose results `ConstFold` bakes in, and `types/pine_types.py`, `types/na.py` and `core/overload.py`, through which those folded values are computed;
- the calling interface of the runtime the emission calls into (`_RUNTIME_ABI`): the slot-state helpers of `core/instance_state.py`, the conversions of `core/safe_convert.py`, the security protocol functions of `core/security.py` and the fields of `ScriptRequirements`. Only their parameter lists (a class: its fields) are hashed, read by parsing just those definitions; a cached script calls whatever the current body does, so a body edit does not invalidate it, while a changed parameter list does. `tests/t00_pynecore/core/test_157_runtime_abi_hash.py` checks both, and fails when a listed helper no longer exists.

The environment switch `PYNE_NO_SECURITY_SLICE`, which changes the emitted tree, is mixed into the digest, and the result is memoized together with the switch's value, so flipping it mid-process yields a different hash at once. The first 16 hex characters are used. The switches that change only runtime behavior (`PYNE_NO_SECURITY_MERGE`, `PYNE_NO_SECURITY_DEV_SKIP`) are deliberately not part of it. `PIPELINE_DIGEST` exposes the value taken at import time, for tools that need to tell bundles built by different pipelines apart.

A rule of thumb for contributors: every file a pass bakes a value from must be in the hash, or the constant could change while the digest stays put.

**Lib facts.** Two passes depend on facts of the lib that only exist on the imported, transformed modules: `FunctionIsolation` routes a call by what the callee is (state-carrying, stateless, an overload dispatcher, a module property), and `ImportNormalizer` expands an `import *` by the module's `__all__`. The lib sources are not in the hash, so neither pass reads the live lib. `scripts/lib_type_collector.py` imports the lib, classifies every callee path under `lib` with `FunctionIsolation`'s own `classify_callee`, and records the result in the `routes` section of `lib_types.json` (only the paths that do not route uniform; an absent path routes uniform) and every lib module's `__all__` in `star_exports`. The passes read those sections, and `lib_types.json` is in the hash. A route-relevant lib edit therefore reaches cached script bytecode through the regenerated file:

```bash
python scripts/lib_type_collector.py
```

`tests/t00_pynecore/ast/test_104_lib_types_registry.py` regenerates the registry in memory and fails, naming every changed route and export list, until the committed file matches the lib. Run the collector and commit the regenerated file together with the lib change. The callees of other modules are still inspected at transform time: user Pyne libraries, and `pynecore` modules outside `lib`, which contain no `@pyne` code (`test_130_lib_route_registry.py` checks this), so their callees can only skip or route uniform, which are both correct for whatever a cached script was compiled against.

**Plain package modules.** A module of the `pynecore` package that contains `cast(` or `TYPE_CHECKING` goes through `TypeErasure` and carries `__pyne_type_erased__` instead, paired with a digest of `transformers/type_erasure.py` alone (`_get_type_erasure_hash`). Editing another transformer does not invalidate these modules.

**Dependency records.** The types of a module are derived from the *interfaces* of the Pyne modules it imports, and its emission routes each call into another module by whether the callee keeps state; CPython's check sees neither. `PineType` records every interface it consulted, and every Pyne module the module imports at module level, called or not: the lowering routes a call on whatever object an import binds, which can live one module further out (`from b import g` in library `a` makes a script's `a.g(x)` a call into `b`). The loader bakes the records into the module as `__pyne_type_deps__`, a tuple constant of `(path, mtime_ns, size, digest, routes)` entries, so `get_code` can read them off a `.pyc` without importing anything. The `routes` digest records how the dependent's calls into the dependency were routed: which of its reachable functions carry state (and so take the hidden state argument), plus the module-level structure that decides what an imported name is (decorators, assignments, imports and class bodies, no function bodies). The dependency's lowering settles it; the loader fills in the records the type pass made off an analysis-only interface (a module whose lowering had not run yet) before baking them. A call the lowering routes on the object an import chain binds also records every module on that chain and the module defining the callee; a plain Python module among them publishes no interface, so its record carries `source` (`module_interface.SOURCE_DIGEST`) as both digests and holds only while its `(mtime_ns, size)` pair does. Because a dependent records its dependencies' dependencies as well, plain-module records included, the closure is transitive.

`get_code` accepts the bytecode when each record still holds (`dep_current`): if the dependency's `(mtime_ns, size)` pair is unchanged it costs one `os.stat`; if the file moved, the dependency's interface is looked up again (its `.pyc` when that is current, otherwise a transform of the dependency into its `.pyc`, see below) and both its digest and its routes digest are compared with the recorded ones. The digest covers the exported signatures, the classes a dependent may annotate with and the module's `__all__`, and nothing about any body. A body edit that changes no signature and no function's statefulness therefore leaves every dependent's cached bytecode valid; one that makes a function start or stop keeping state, or that changes a signature, invalidates exactly the dependents that depend on it. `__pyne_capture_deps__` works the same way for the source files `ExportCapture` consulted, with `(path, mtime_ns, size)` entries.

**The published interface.** Every transformed module also carries its own interface, as `__pyne_interface__ = ('__pyne_interface__', mtime_ns, size, payload)`: the `(mtime_ns, size)` of the source bytes it was derived from and a zlib-compressed canonical JSON of its exports, classes, extensions, `__all__`, suppression flag and routes digest (`module_interface.interface_payload`; its dependency closure is the module's own `__pyne_type_deps__`). Like the records it is a constant, so another module's transform reads it off the `.pyc` without executing anything. Python replaces the `.pyc` when the module is recompiled, so nothing accumulates beside it. A module transformed from bytes that could not be paired with a stat carries none.

**Interface lookup.** `module_interface.lookup` finds a module's interface wherever it is cheapest:

1. a process-wide registry, filled by every transform in the process (the loader calls `build_interface` and `register` on the analysed tree, and registers it again once the lowering has settled its routes);
2. the interface the module's own `.pyc` carries (`import_hook.compile_interface`). It is accepted under exactly the checks an import makes, so the import never transforms a module whose interface was read here: CPython's timestamp header against the source's stat, the pipeline sentinel, the baked fingerprint, `__pyne_capture_deps__` and the dependency records with their routes;
3. otherwise the module is transformed into its `.pyc`, by the same `PyneLoader.source_to_code` its import runs, and the `.pyc` is written where and how `SourceLoader.get_code` writes it (none under `sys.dont_write_bytecode`, silently none into an unwritable cache directory). The import that follows finds it and does not transform the module again; where nothing could be written, the registry still answers for the rest of the process. The module is compiled, not executed: its lowering imports the modules it calls, as every transform does.

The registry answer is checked against the module's own fingerprint and its dependency closure before it is handed out, and a stale entry is evicted. A caller that needs the routes (the dependency check of a dependent's bytecode, the settling of its records) does not accept an entry without them. An import cycle terminates because a module that is still being analysed answers `None`. A dependency transformed for a lookup whose analysis reaches back into a module still under analysis is not what its import will produce: that import runs once the analysis is done and finds the interface. Such a transform stops after its analysis (`module_interface.reached_back`), answers with the interface the analysis published, without routes, and writes no `.pyc`. A failed transform is not remembered, since the file may be fixed a moment later; the module that actually imports it raises the real error.

## Debugging the Transformation

To see the transformed code of a script, use:

```bash
pyne debug ast my_script.py
```

or the `PYNE_AST_DEBUG` family of environment variables — see [Debugging](../debugging.md#inspecting-the-transformed-code) for the user-facing summary. The variables the pipeline reads:

| Variable                    | Effect                                                              |
|-----------------------------|---------------------------------------------------------------------|
| `PYNE_AST_DEBUG`            | Print each transformed module, with named slot constants            |
| `PYNE_AST_DEBUG_RAW`        | Print the exact emission: `1` for every module, a file path for one |
| `PYNE_AST_SAVE`             | Write each transformed module to `/tmp/pyne/<module stem>.py`       |
| `PYNE_AST_TIMING`           | Print the wall time of every step to stderr                         |
| `PYNE_AST_SEQUENTIAL`       | Run the rules of a fused phase as separate passes                   |
| `PYNE_TYPE_DIAG`            | `1`: print the type diagnostics of every module to stderr           |
| `PYNE_EDGE_STRICT`          | `1`: the first diagnostic of an `@pyne edge` module raises          |
| `PYNE_NO_SECURITY_SLICE`    | Truthy: switch `SecuritySlice` off (changes the emission)           |
| `PYNE_NO_SECURITY_MERGE`    | Truthy: one child process per security context (runtime only)       |
| `PYNE_NO_SECURITY_DEV_SKIP` | Truthy: no developing-round skip for `closed_shift` (runtime only)  |

Details, as the source reads them:

- `PYNE_AST_DEBUG`, `PYNE_AST_SAVE`, `PYNE_AST_TIMING` and `PYNE_AST_SEQUENTIAL` are enabled by any non-empty value (`PYNE_AST_DEBUG=0` still enables the dump). The dump is syntax-highlighted when `rich` is installed.
- `PYNE_AST_DEBUG_RAW` prints `ast.unparse` of the emission with literal slot indexes, which is what the AST golden tests compare against. It is ignored when `PYNE_AST_DEBUG` is set.
- `PYNE_AST_TIMING` prints `[pyne-ast] <file>: <step> <ms> ms` for every step of every module: one line per phase, or one per rule in sequential mode.
- `PYNE_EDGE_STRICT=1` raises a `PineTypeError` for the first diagnostic of an `@pyne edge` module; a hand-written script keeps running and keeps the list.
- The truthy values of the `PYNE_NO_SECURITY_*` switches are `1`, `true`, `yes` and `on`. Only `PYNE_NO_SECURITY_SLICE` is part of the pipeline hash: the other two change runtime behavior, not the bytecode.

Notes:

- The dumps are written by the transform itself. A module loaded from a valid cached `.pyc` is not transformed, so nothing is printed for it; edit the source or delete its cache entry to force a transform. `PYNE_AST_DEBUG` and `PYNE_AST_DEBUG_RAW` print every Pyne module transformed in the process, the library modules a script imports included (a path filter in `PYNE_AST_DEBUG_RAW` keeps a capture clean).
- `PYNE_AST_TIMING` and `PYNE_AST_SEQUENTIAL` are read once, when `pipeline.py` is imported, so set them in the environment of the process rather than inside a script.
- To check that a fused phase still equals its separate rules, dump the same module with and without `PYNE_AST_SEQUENTIAL=1` (`PYNE_AST_DEBUG_RAW=<path>`, on a script without a cached `.pyc`) and compare the outputs.
- The type diagnostics are collected for every module; `PYNE_TYPE_DIAG=1` prints them, and for an `@pyne edge` module the edge gate's structural findings are merged in (a typed diagnostic that only repeats a structural one at the same place is dropped).
