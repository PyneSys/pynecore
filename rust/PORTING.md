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
  (`stamp_lowering`). Hermetic.
- **F2.** `pine_qualifier.py` is hermetic; `data.rs` must also carry the `names`,
  `classes`, `scalar_classes` and `routes` sections of `lib_types.json`.
- **F3.** After the Python PineType step the module carries `_pine_types`, a
  `PineTypeTable` dataclass, which the bridge cannot convert, so every later
  native step answers `Unsupported`. Either pass the object through opaquely
  (a `Value` holding a Python reference in the bridge) or convert the table both
  ways; a native PineType needs the second, because `function_isolation`,
  `module_interface.build_interface` and `import_hook` read the table as Python
  dataclasses.
- **Digest closures.** `native._closure` follows `transformers` imports
  transitively. Before porting, cut couplings that would stale a port on
  unrelated edits:
  - `export_capture.py` imports `lib_types` from `pine_type_infer`;
  - `security.py` and `security_slice.py` import `core.import_hook` for
    `PYNE_RESERVED_NAME_CHAR` and `security_slice_disabled`; move both into a
    small module;
  - ConstFold's closure covers `lib/math.py`, `core/fdlibm.py`,
    `core/safe_convert.py`, `pine_type_rules.py`: correct, those decide its output.

## Steps, in porting order

1. **ImplicitConversion** (S). Hermetic; reads `_pine_widen`, `_pine_widen_site`,
   `_pine_na_class`, `_pine_ty`. Needs F1, F3.
2. **SecurityInstantiation** (S). Hermetic fixpoint over the call graph; needs only
   Security's `has_security_call` / `_is_security_call` predicates.
3. **ExportCapture** (M). Runs only for `@library` modules. Reads other modules'
   sources to prove enum members constant: needs Host methods along the lines of
   `find_spec(name, search) -> Option<Spec>` and
   `read_source(path) -> Option<SourceRead { resolved, bytes, mtime_ns, size }>`.
   Stamps `_export_capture_deps` on the module. Needs F1, F2.
4. **ConstFold** (M-L). Bakes numerics into the tree:
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
5. **Security** (XL, 4.4k lines). Split into predicates, collection, builders,
   rewrite, groups, cross-function lift, module emission, and the
   `_DependencyAnalyzer` (~1.5k lines, shared with SecuritySlice) as its own
   module. Non-hermetic only through `sha1(module path)`.
6. **SecuritySlice** (L, 2.8k lines). Needs Security's dependency analyzer, the
   `_pine_ty` stamps and F3; `PYNE_NO_SECURITY_SLICE` becomes a context flag.
7. **PineType** (XXL, ~11k lines in its closure). `assign_node_ids` stamps
   `_pine_nid` in preorder on every node; `_Inference` (scopes, statements and
   the loop fixpoint, definitions and per-call-site contexts, expressions,
   qualifiers, lib / user / method calls and overload groups, cross-module calls,
   instance vectors) stamps `_pine_ty`, `_pine_pin(s)`, `_pine_vector`,
   `_pine_varying`, `_pine_widen*`, `_pine_na_class`, `_pine_series_len(s)` and
   builds the `PineTypeTable`. Cross-module analysis needs Host methods for the
   module source of a dotted name, package members, path resolution, the stdlib
   test, the analysis-in-progress set, cycle detection and an imported module's
   interface (the `interface_payload` JSON).
