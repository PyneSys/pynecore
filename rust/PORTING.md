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
- **F3.** Done for the steps after PineType: the module's `_pine_types` (a
  `PineTypeTable` dataclass) and any other stamp that is no plain builtin value
  cross the bridge as `Value::Opaque`, the very Python object handed back; the
  other stamps (`_pine_ty`, `_pine_pin(s)`, `_pine_vector`, `_pine_varying`,
  `_pine_widen*`, `_pine_series_len(s)`) are plain values and node references.
  `native.describe` compares a dataclass field by field. A native PineType still
  needs the table converted both ways, because `function_isolation`,
  `module_interface.build_interface` and `import_hook` read it as Python
  dataclasses, and the WASM host has no Python object to hand through.
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
   `Host::read_source` (`module_interface.stable_source` plus `path.resolve()`),
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
4. **Security** (XL, 4.4k lines). Split into predicates, collection, builders,
   rewrite, groups, cross-function lift, module emission, and the
   `_DependencyAnalyzer` (~1.5k lines, shared with SecuritySlice) as its own
   module (`steps/security/`, where the call predicates already are).
   Non-hermetic only through `sha1(module path)`.
5. **SecuritySlice** (L, 2.8k lines). Needs Security's dependency analyzer, the
   `_pine_ty` stamps and F3; `PYNE_NO_SECURITY_SLICE` becomes a context flag.
6. **PineType** (XXL, ~11k lines in its closure). `assign_node_ids` stamps
   `_pine_nid` in preorder on every node; `_Inference` (scopes, statements and
   the loop fixpoint, definitions and per-call-site contexts, expressions,
   qualifiers, lib / user / method calls and overload groups, cross-module calls,
   instance vectors) stamps `_pine_ty`, `_pine_pin(s)`, `_pine_vector`,
   `_pine_varying`, `_pine_widen*`, `_pine_na_class`, `_pine_series_len(s)` and
   builds the `PineTypeTable`. Cross-module analysis needs Host methods for the
   module source of a dotted name, package members, path resolution, the stdlib
   test, the analysis-in-progress set, cycle detection and an imported module's
   interface (the `interface_payload` JSON).
