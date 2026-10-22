# Porting map of the remaining steps

The order the remaining Python steps are ported in, and why. Line references
are to `src/pynecore/transformers/` unless noted; they drift, so treat them as
pointers.

## Shared prerequisites

| Id | What                                         | Needed by                                              |
|----|----------------------------------------------|--------------------------------------------------------|
| F1 | Core of `pine_type_rules.py`                 | ConstFold, ExportCapture, ImplicitConversion, SecuritySlice, PineType |
| F2 | `pine_qualifier.py` + more `lib_types.json`  | ExportCapture, PineType                                |
| F3 | The `_pine_types` table across the boundary  | Every native step after PineType                       |
| F4 | SHA-1, shortest float digits, shallow copy   | Security, ConstFold, SecuritySlice                     |

- **F1.** Pine types are strings: scalars `i f b s c o v ? *`, shaped
  `a:` / `m:` / `M:` / `T:` / `o:<module>#Name`. Port `join`, `binop_type`,
  `unaryop_type`, `compare_type`, the `annotation_type` family,
  `LIB_TYPE_OVERRIDES`, `BUILTIN_CALL_TYPES` and the stamp accessors
  (`stamp_lowering`). Hermetic. Ported as `crates/pyne-transform/src/pine_type_rules.rs`,
  gated against the Python module by `tests/t00_pynecore/ast/test_136_native_type_rules.py`
  through `_native._rules`.
- **F2.** Ported as `crates/pyne-transform/src/pine_qualifier.rs`; `data.rs` carries
  the `routes`, `classes` and `scalar_classes` sections of `lib_types.json` whole,
  and of `names` the fields a native step reads so far (`data::LibName`: kind,
  value type, callable, return types). Both are gated against Python by
  `tests/t00_pynecore/ast/test_137_native_qualifier.py` through `_native._qualifier`
  and `_native._data`. PineType needs the rest of `names` (parameters, defaults).
- **F3.** The module's `_pine_types` (a `PineTypeTable` dataclass) crosses the
  bridge one of two ways, chosen per run:
  - **Handed through** (`Value::Opaque`), the very Python object handed back, when
    no step of the run reads the table (every port so far). Any other stamp that is
    no plain builtin value goes the same way; the rest (`_pine_ty`, `_pine_pin(s)`,
    `_pine_vector`, `_pine_varying`, `_pine_widen*`, `_pine_series_len(s)`) are
    plain values and node references.
  - **Owned natively** (`Value::Native`): `crates/pyne-transform/src/pine_type_table.rs`
    mirrors the table and every dataclass it holds (`Unknown`, `Binding`, `FuncSig`,
    `ContextKey`, `ContextResult`, `CallSite`, `Diag`, `DepRecord`, `ExportSig`,
    `ClassSig`, and `ImplSig` from `pine_type_rules.rs`) field for field; a `dict`
    is an `IndexMap` (insertion order kept), a `frozenset` a `BTreeSet`. The tree
    keeps the object in a type-erased slot of the arena (`Ast::add_native`,
    `Ast::native::<PineTypeTable>`), so `pyne-ast` stays free of the transform's
    types and the WASM host of Python. The bridge converts a Python table into
    one when a step of the run is listed in `TYPE_TABLE_READERS`
    (`FromPython::native_type_tables`: strict, an unexpected type answers
    `Unsupported`), and turns any native table, converted or built by a native
    step, into a real `PineTypeTable` of the same classes, values, container types
    and order (`crates/pyne-transform-py/src/type_table.rs`), which
    `function_isolation`, `module_interface.build_interface` and the import hook
    read as they read the Python pass's own. `native.describe` writes a dataclass
    field by field, so `verify` compares a native table by value. A native
    PineType stamps the module with `Ast::add_native(table)`; a native reader
    joins `TYPE_TABLE_READERS`.

  `node_ids.rs` ports `assign_node_ids`: the operator and context objects take an
  id from the count though they are no arena nodes. Both are gated against Python
  by `tests/t00_pynecore/ast/test_140_native_type_table.py` (`_native._type_table`,
  `_native._node_ids`) over the corpus and a two-module library that fills every
  field. The interface payload (`module_interface.interface_payload`) is no shape
  for the table: it serializes a `ModuleInterface`, which `build_interface`
  derives from the tree and a projection of the table (exports, classes,
  extensions, the suppression message); bindings, contexts, call sites,
  diagnostics, groups, dependency records and call positions do not travel in it.
  Its `ExportSig`/`ClassSig` JSON fits the native types as they are, for the Host
  method that will hand an imported module's interface to a native PineType.
- **Digest closures.** `native._closure` follows `transformers` imports
  transitively. Before porting, cut couplings that would stale a port on
  unrelated edits:
  - ConstFold's closure covers `lib/math.py`, `core/fdlibm.py`,
    `core/safe_convert.py`, `pine_type_rules.py`: correct, those decide its output.

## Steps, in porting order

1. **ImplicitConversion** (S). Hermetic; reads `_pine_widen`, `_pine_widen_site`,
   `_pine_na_class`, `_pine_ty`. Needs F1, F3. Ported as `steps/implicit_conversion.rs`;
   it never reads the `PineTypeTable`.
2. **ExportCapture** (M). Ported. The imported modules are located and read
   through `Host::module_origin` (`export_capture.module_origin`) and
   `Host::read_source` (`source_read.stable_source` plus `path.resolve()`),
   and parsed into the same arena; a source the native parser or the UTF-8 check
   does not take answers `Unsupported`. The WASM host answers both `Unsupported`
   until a request carries the sources of the modules a library imports.
3. **ConstFold** (M-L). Bakes numerics into the tree:
   - `core/fdlibm.py` (all nine public functions and their kernels): port it, do
     not use the `libm` crate (musl's exp/log/pow are not fdlibm 5.3); gate against
     Python over millions of arguments as `test_016_native_math_matches_python`
     does for the Cython twin;
   - the `lib.math` folds (`sqrt`, `abs`, `floor`/`ceil` returning float, `max`/
     `min` with Python's mixed compare, Kahan `avg`, `sign`, `todegrees`/
     `toradians`, `round` — whose precision path uses the platform `pow` and an
     exact tie check that may need a bignum: answer `Unsupported` where exactness
     cannot be guaranteed);
   - Python int semantics for `+ - * / %` (`safe_mod`): checked `i64`, answer
     `Unsupported` on overflow or `Int::Big`;
   - `quantize_embed`: shortest digits, round-down at 1e-16 with an exact binary
     tie check (fits in `u128`), correctly rounded parse.

   Ported as `steps/const_fold.rs` with `fdlibm.rs`. The fdlibm port and
   `quantize_embed` are gated bit for bit by `test_139_native_fdlibm`
   (`PYNE_FDLIBM_SWEEP=2000000` for the full sweep), the evaluation and the step
   by `test_138_native_const_fold`. `Unsupported`: an int literal or result beyond
   `i64`, an int true division with an operand beyond 2**53, and `math.round` with
   a positive precision.
4. **Security** (XL, 4.4k lines). Ported as `steps/security/`: `collection`
   (call arguments, the simple-chain / hoistable-binding rules, the
   history-reference rule behind `closed_shift`), `builders`, `rewrite` (the
   per-function lowering), `lift` (the cross-function signal lift and the
   reachability rules), `groups` (`always_read`, `signal_per_bar`, the feed
   groups), the module visit with the registry in `mod.rs`, and `dependencies`,
   the port of `security_dependencies.py` (`_DependencyAnalyzer`, which the slice
   pass reuses). The sec id prefix is `sha1(module path)[:8]`, the path taken from
   the module's `_module_file_path` stamp as in Python, the digest from
   `crate::sha1` (written out: a few dozen lines, nothing extra for the WebAssembly
   build; it covers F4's SHA-1). Its closure was cut to leaf modules
   (`script_entry`, `factory_fields`, `lib_series_attrs`, `security_dependencies`).
   Gated by `tests/t00_pynecore/ast/test_141_native_security.py`. `Unsupported`:
   more than one write site the analysis's site walk never reaches (Python numbers
   them in string-set order, which follows the hash seed), a `__hist_N__` temp
   spelled with a non-ASCII digit, and a `_module_file_path` that is no str.
5. **SecuritySlice** (L, 2.8k lines). Ported as `steps/security_slice/`: `facts`
   (the reads / writes / mutations of `main()`'s statements and of the functions it
   reaches, the alias classes, the slice), `reach` (whether a write runs on every
   execution of the child's code, the `varip` scan behind `closed_shift`), `forcer`
   (the rewrite making a guarded write unconditional) and the module visit in
   `mod.rs`. It reuses the lexical half of the dependency analyzer
   (`DependencyAnalyzer::scopes_only`, `lookup_func_from`) and reads `_pine_ty`
   stamps only, never the `PineTypeTable`. `PYNE_NO_SECURITY_SLICE` crosses the
   bridge as `Context::security_slice_disabled`, which `native._context` reads from
   `emission_settings.security_slice_disabled()` where the Python step reads it.
   The Python step builds several clones of one module from parts of `main()` copied
   and located once (`_CloneSource`), which yields the clone `_build_clone` builds
   whenever it is used, so the port always builds it the latter way. `VARIP_TYPES`
   moved into the leaf `varip_types.py`, so the closure no longer holds
   `persistent.py` and `slot_layout.py`. Gated by `test_141_native_security.py`
   (typed and untyped inputs, the slicing on and off). Nothing answers
   `Unsupported`.
6. **PineType** (XXL, ~11k lines in its closure). `assign_node_ids` stamps
   `_pine_nid` in preorder on every node; `_Inference` (scopes, statements and
   the loop fixpoint, definitions and per-call-site contexts, expressions,
   qualifiers, lib / user / method calls and overload groups, cross-module calls,
   instance vectors) stamps `_pine_ty`, `_pine_pin(s)`, `_pine_vector`,
   `_pine_varying`, `_pine_widen*`, `_pine_na_class`, `_pine_series_len(s)` and
   builds the `PineTypeTable`. Ported as `steps/pine_type/`, one module per part
   of `_Inference`: `mod` (the engine state, frames, stamping, the entry point),
   `facts` (the module-level helpers of `pine_type_infer.py`), `collect` (imports,
   classes, definitions), `stmts` (statements, the loop fixpoint, stores), `defs`
   (definitions, call-site contexts, signatures), `exprs` (expressions and
   qualifiers), `lib_calls`, `user_calls` (user and method calls, overload
   groups), `imports` (imported callees and dependency records), `finish`
   (instance vectors, window machines, widen sites, pin suppression) and `report`
   (`pine_type_report.py`); `unparse.rs` is CPython 3.13's `ast.unparse` for
   expressions. The lib names come from the compiled `lib_types.json`
   (`data::lib_entries`). Cross-module analysis asks the Host, whose Python side
   calls the functions the Python pass calls: `interface` (`module_interface.lookup`,
   sent as the JSON of `dataclasses.asdict` of the interface and of its
   `dep_record`), `closes_cycle`, `begin_analysis` / `end_analysis`
   (`analysing_scope`), `module_spec_origin` and `package_member`
   (`pine_type_infer`), `is_stdlib`. The WASM host answers them `Unsupported`, so
   there a module that imports anything outside `pynecore` stays in Python.
   `_overload_groups` and the hops of `_declares_method` were sets whose order
   reached the table; they are ordered now. `module_interface` types the layout
   `routes_digest` reads with a protocol, so `slot_layout.py` is out of the
   closure. `Unsupported`: a string annotation the native expression parser
   refuses (`""` among them), `ast.unparse` of a node outside the expression
   subset, the `repr` of text beyond ASCII, Latin-1 and General Punctuation,
   and any failing Host answer.

## Lowering steps

`pipeline.run_steps` dispatches a plain `Step` of `LOWERING` exactly as one of
`ANALYSIS`. A `Phase` is dispatched the same way, as one unit under the phase's
name (`Expression`): `crates/pyne-transform/src/phase.rs` ports `phase.py` (one
post-order traversal, `enter` hooks in rule order, then the children, then the
`leave` hooks, each handed what the previous one answered, the function-body
hooks around a `def`'s `body`), and `PHASES` declares each phase's rules with
their `user_code_only`. Every rule is also a step of `STEPS`, ported from its own
module with its own `ported.json` entry. `native.current_phases` lets a phase run
natively while the native phase declares the same rules (names, order,
`user_code_only`) and every rule is current; otherwise the phase runs in Python.
In sequential mode (`PYNE_AST_SEQUENTIAL`, `PipelineContext.sequential`) each rule
is a unit of its own, so `verify` compares every rule separately; otherwise it
compares the phase's result. The rule constructor arguments come from the context:
`pyne_mode` (`lib`), `user_code` (`pine_mod`) and `na_bool`, which crosses the
bridge with `sequential` (a phase name run with `sequential` runs its rules as
separate passes, for a host without Python). The steps that read
`ctx.slot_layout` get the layout across the boundary as described below.

**The slot layout.** `crates/pyne-transform/src/slot_layout.rs` mirrors
`slot_layout.py` (`ModuleLayout`, `ScopeLayout`, `_Slot` as `Slot` with a
`SlotKind`, the scope scan, `scope_for_function`); `Context::slot_layout` holds
the run's layout (`RefCell`, the steps fill it in place). A slot's `init` and
`max_bars_back` templates are nodes of the arena, and the definition caches
(`_segments`, `_nests`, keyed by `id()` in Python) are keyed by the definition's
`NodeId`. For a run with a step of `SLOT_LAYOUT_STEPS`, the bridge
(`crates/pyne-transform-py/src/slot_layout.rs`) converts `context['slot_layout']`
after the tree with the same `FromPython`, so a template that is a node of the
tree stays one node (strict: exact classes and field types, a `_tree` cached for
another module answers `Unsupported`), and converts the layout the steps left
back with the same `ToPython` into a new `ModuleLayout`
(`context['slot_layout_result']`); a cache entry of a definition the conversion
does not reach is dropped, since no step can ask for it. `native.run(names, tree,
ctx, layout)` replaces `layout`'s state field for field with it: the native
backend passes `ctx.slot_layout`, `verify` a fresh layout per step, which
`native.verify` compares with what the Python step left (`describe(tree,
layout)`: the scopes slot by slot with node references shared with the tree, and
what the layout answers for every definition of the tree). The WASM host starts a
request from `ModuleLayout::new(pyne_mode == "lib")`, as the import hook does.
`apply_layout` moved to `layout_emit.py`, so the layout model no longer imports
`pine_type_rules.py`.

1. **SecurityDefault**, **ExportOnce**, **UnusedSeriesDetector**. Ported as
   `steps/security_default.rs`, `steps/export_once.rs` and
   `steps/unused_series_detector.rs`; none reads the `PineTypeTable`.
   SecurityDefault's closure includes `pine_type_rules.py` (`get_ty`,
   `elements_of`, `stamp_lowering` decide its output). `Unsupported`: a `_pine_ty`
   stamp on a security read that is no string, and a `Series[...]` annotation
   whose slice is neither a name nor a node with a node `value` field (the Python
   pass would put a constant's raw value or `None` into the annotation).
2. **Series**, **VerifyClosedShift**, **Persistent**. Ported as `steps/series.rs`,
   `steps/security_closed_shift_check.rs` and `steps/persistent.rs`; none reads
   the `PineTypeTable`. Their closures were cut to what decides their output: the
   dynamic-default guard predicate lives in `dynamic_default_guard.py` (Series no
   longer covers DynamicDefault), and the `__security_contexts__` readers in
   `security_contexts.py` (VerifyClosedShift no longer covers SecuritySlice,
   Security, LibrarySeries and Persistent; ported as `steps/security/contexts.rs`).
   Series' closure includes `pine_type_rules.py` (`get_ty`, `set_ty`,
   `stamp_lowering` decide its output). `Unsupported`: a `_pine_ty` stamp that is
   neither a string nor `None` where Series types the plumbing it builds (the
   Python pass keeps such a stamp, the native `stamp_lowering` reads strings
   only), a visit inside a dynamic-default guard that answers no single node (the
   Python pass would put the answer into the body as it is), a series declaration
   in a scope the pass never entered (a `KeyError` in Python), and an int
   persistent initializer beyond 64 bits (`float()` of a Python int).
   `Rejected`: a Series or Persistent declaration outside a function.
3. **ScriptRequirements**, **Input**, the `Expression` phase (**SafeConvert**,
   **SafeDivision**, **FloatTolerance**) and **FixLocations**. Ported as
   `steps/script_requirements.rs`, `steps/input.rs`, `steps/safe_convert.rs`,
   `steps/safe_division.rs`, `steps/float_tolerance.rs` and `steps/locations.rs`
   (`pyne_ast::locations`); none reads the `PineTypeTable` or the slot layout.
   `BUILTIN_PRICE_SERIES` moved into the leaf `price_series.py`, so Input's closure
   no longer holds `lib_series.py`; FloatTolerance's holds `core/pine_compare.py`
   (`EPSILON`), as PineTruthiness's does. `Unsupported`: a `_pine_ty` stamp that is
   no string where a rule carries it over to a new node, an int index literal
   beyond `i64` (`lst[1e300]` typed int), an opaque `pine_exact` stamp, a leave hook
   handed a list by an earlier rule, and a node with a line but no column for
   `fix_locations`.
4. **CallInline**. Hermetized first: the inline templates are the generated
   registry `transformers/call_inline_templates.json`, written by
   `scripts/call_inline_collector.py` (the body-shape derivation moved there from
   the step) and guarded by `tests/t00_pynecore/ast/test_142_call_inline_templates.py`,
   which names the function whose template changed. Per function the registry holds
   its parameters and derived expression (a parameter read is `{"param": i}`, every
   other node its class, fields and the wrapper-source location the emission
   carries), or the builtin and anchors of the varargs expansion; plus the callee
   paths (`callees`), the spellings of `pine_range` (`loop_ranges`) and the varargs
   arities. The Python step builds its templates from it and proves a callee by the
   path its import map resolves to, where it compared live objects before;
   `DEFAULT_STATE_PARAM` moved into the leaf `state_param.py`, so the closure is the
   step, its registry, `ast_walk.py`, `pine_type_rules.py` and `factory_fields.py`.
   Ported as `steps/call_inline.rs`, reading the registry through
   `data::inline_templates`; it never reads the `PineTypeTable` or the slot layout.
   `Unsupported`: an `_inl_argument` marker that is no bool, a `_pine_ty` stamp on
   the call that is no string or one in the inlined expression that is neither a
   string nor `None`, and a literal comparison with an int beyond 64 bits.
5. **FinalizeCloneDefaults**. Ported in `steps/security_slice/mod.rs` with the step
   it finishes; it shares SecuritySlice's module and digest.
