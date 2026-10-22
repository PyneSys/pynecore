//! The type pass's `PineTypeTable` <-> [`PineTypeTable`], both ways.
//!
//! Python to Rust is strict: every dataclass must be of its exact class, every
//! value of the exact builtin type its field holds (a `bool` is no `int`, a
//! `tuple` no `list`), or the conversion answers `Unsupported` and the Python step
//! runs instead. Rust to Python builds the very classes the type pass builds,
//! through their constructors, with the containers the pass uses (`list`,
//! `tuple`, insertion-ordered `dict`, `frozenset`), so the consumers of a table
//! (`function_isolation`, `module_interface.build_interface`, the import hook)
//! cannot tell a converted table from the pass's own.

use std::collections::BTreeSet;
use std::hash::Hash;

use indexmap::IndexMap;
use pyne_ast::{NodeId, Value};
use pyne_transform::node_ids::assign_node_ids;
use pyne_transform::pine_type_rules::ImplSig;
use pyne_transform::pine_type_table::{
    Binding, CallSite, ClassSig, ContextKey, ContextResult, DepRecord, Diag, ExportSig, FuncSig,
    PineTypeTable, Unknown,
};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyInt, PyList, PyString, PyTuple};

use crate::Unsupported;
use crate::convert::{FromPython, ToPython};

/// The Python classes a table is made of.
pub struct Classes<'py> {
    table: Bound<'py, PyAny>,
    unknown: Bound<'py, PyAny>,
    binding: Bound<'py, PyAny>,
    func_sig: Bound<'py, PyAny>,
    context_result: Bound<'py, PyAny>,
    call_site: Bound<'py, PyAny>,
    diag: Bound<'py, PyAny>,
    dep_record: Bound<'py, PyAny>,
    export_sig: Bound<'py, PyAny>,
    class_sig: Bound<'py, PyAny>,
    impl_sig: Bound<'py, PyAny>,
}

impl<'py> Classes<'py> {
    pub fn import(py: Python<'py>) -> PyResult<Classes<'py>> {
        let table = py.import("pynecore.transformers.pine_type_table")?;
        let rules = py.import("pynecore.transformers.pine_type_rules")?;
        Ok(Classes {
            table: table.getattr("PineTypeTable")?,
            unknown: table.getattr("Unknown")?,
            binding: table.getattr("Binding")?,
            func_sig: table.getattr("FuncSig")?,
            context_result: table.getattr("ContextResult")?,
            call_site: table.getattr("CallSite")?,
            diag: table.getattr("Diag")?,
            dep_record: table.getattr("DepRecord")?,
            export_sig: table.getattr("ExportSig")?,
            class_sig: table.getattr("ClassSig")?,
            impl_sig: rules.getattr("ImplSig")?,
        })
    }

    /// Whether an object is a `PineTypeTable` (of that exact class).
    pub fn is_table(&self, object: &Bound<'py, PyAny>) -> bool {
        object.get_type().is(&self.table)
    }

    // --- Python -> Rust -------------------------------------------------------

    pub fn table_from(&self, object: &Bound<'py, PyAny>) -> PyResult<PineTypeTable> {
        let get = |name: &str| attr(object, &self.table, "PineTypeTable", name);
        Ok(PineTypeTable {
            module_path: string(&get("module_path")?)?,
            bindings: dict(&get("bindings")?, string, |scope| {
                dict(scope, string, |item| self.binding_from(item))
            })?,
            funcs: dict(&get("funcs")?, string, |item| self.func_sig_from(item))?,
            contexts: dict(&get("contexts")?, context_key, |item| self.context_from(item))?,
            calls: list(&get("calls")?, |item| self.call_site_from(item))?,
            diags: list(&get("diags")?, |item| self.diag_from(item))?,
            groups: dict(&get("groups")?, string, |impls| {
                tuple(impls, |item| self.impl_sig_from(item))
            })?,
            deps: dict(&get("deps")?, string, |item| self.dep_record_from(item))?,
            classes: dict(&get("classes")?, string, string)?,
            class_sigs: dict(&get("class_sigs")?, string, |item| self.class_sig_from(item))?,
            extensions: dict(&get("extensions")?, string, |methods| {
                dict(methods, string, |item| self.export_sig_from(item))
            })?,
            pins_suppressed: optional(&get("pins_suppressed")?, |item| self.diag_from(item))?,
            exportable: frozenset(&get("exportable")?)?,
            call_pos: dict(&get("call_pos")?, int, |position| {
                let items = tuple(position, int)?;
                match items[..] {
                    [line, col] => Ok((line, col)),
                    _ => Err(mismatch("a call position", position)),
                }
            })?,
        })
    }

    fn unknown_from(&self, object: &Bound<'py, PyAny>) -> PyResult<Unknown> {
        let get = |name: &str| attr(object, &self.unknown, "Unknown", name);
        Ok(Unknown {
            reason: string(&get("reason")?)?,
            line: int(&get("line")?)?,
            col: int(&get("col")?)?,
            detail: string(&get("detail")?)?,
        })
    }

    fn binding_from(&self, object: &Bound<'py, PyAny>) -> PyResult<Binding> {
        let get = |name: &str| attr(object, &self.binding, "Binding", name);
        Ok(Binding {
            name: string(&get("name")?)?,
            ty: string(&get("ty")?)?,
            line: int(&get("line")?)?,
            unknown: optional(&get("unknown")?, |item| self.unknown_from(item))?,
            series: boolean(&get("series")?)?,
            qual: qualifier(&get("qual")?)?,
            depth: int(&get("depth")?)?,
            persistent: boolean(&get("persistent")?)?,
            decl: optional(&get("decl")?, int)?,
            min_read: qualifier(&get("min_read")?)?,
        })
    }

    fn func_sig_from(&self, object: &Bound<'py, PyAny>) -> PyResult<FuncSig> {
        let get = |name: &str| attr(object, &self.func_sig, "FuncSig", name);
        Ok(FuncSig {
            name: string(&get("name")?)?,
            params: list(&get("params")?, string)?,
            ret: string(&get("ret")?)?,
            line: int(&get("line")?)?,
        })
    }

    fn context_from(&self, object: &Bound<'py, PyAny>) -> PyResult<ContextResult> {
        let get = |name: &str| attr(object, &self.context_result, "ContextResult", name);
        Ok(ContextResult {
            cid: int(&get("cid")?)?,
            key: string(&get("key")?)?,
            params: tuple(&get("params")?, string)?,
            ret: string(&get("ret")?)?,
            quals: tuple(&get("quals")?, qualifier)?,
            ret_qual: qualifier(&get("ret_qual")?)?,
            pins: dict(&get("pins")?, int, |pin| optional(pin, string))?,
        })
    }

    fn call_site_from(&self, object: &Bound<'py, PyAny>) -> PyResult<CallSite> {
        let get = |name: &str| attr(object, &self.call_site, "CallSite", name);
        Ok(CallSite {
            callee: string(&get("callee")?)?,
            line: int(&get("line")?)?,
            col: int(&get("col")?)?,
            argc: optional(&get("argc")?, int)?,
            ty: string(&get("ty")?)?,
            pin: optional(&get("pin")?, string)?,
        })
    }

    fn diag_from(&self, object: &Bound<'py, PyAny>) -> PyResult<Diag> {
        let get = |name: &str| attr(object, &self.diag, "Diag", name);
        Ok(Diag {
            message: string(&get("message")?)?,
            line: int(&get("line")?)?,
            col: int(&get("col")?)?,
            origin: optional(&get("origin")?, |item| self.unknown_from(item))?,
            fix: string(&get("fix")?)?,
            end_line: int(&get("end_line")?)?,
            end_col: int(&get("end_col")?)?,
        })
    }

    fn dep_record_from(&self, object: &Bound<'py, PyAny>) -> PyResult<DepRecord> {
        let get = |name: &str| attr(object, &self.dep_record, "DepRecord", name);
        Ok(DepRecord {
            path: string(&get("path")?)?,
            mtime_ns: int(&get("mtime_ns")?)?,
            size: int(&get("size")?)?,
            digest: string(&get("digest")?)?,
            routes: string(&get("routes")?)?,
        })
    }

    fn impl_sig_from(&self, object: &Bound<'py, PyAny>) -> PyResult<ImplSig> {
        let get = |name: &str| attr(object, &self.impl_sig, "ImplSig", name);
        let required = int(&get("required")?)?;
        Ok(ImplSig {
            params: tuple(&get("params")?, string)?,
            required: usize::try_from(required)
                .map_err(|_| unsupported(format!("ImplSig.required {required}")))?,
            open_ended: boolean(&get("open_ended")?)?,
            ret: string(&get("ret")?)?,
            fits: string(&get("fits")?)?,
            names: tuple(&get("names")?, string)?,
        })
    }

    fn export_sig_from(&self, object: &Bound<'py, PyAny>) -> PyResult<ExportSig> {
        let get = |name: &str| attr(object, &self.export_sig, "ExportSig", name);
        Ok(ExportSig {
            name: string(&get("name")?)?,
            kind: string(&get("kind")?)?,
            params: tuple(&get("params")?, string)?,
            required: int(&get("required")?)?,
            open_ended: boolean(&get("open_ended")?)?,
            ret: string(&get("ret")?)?,
            annotated: boolean(&get("annotated")?)?,
            impls: tuple(&get("impls")?, |item| self.impl_sig_from(item))?,
            line: int(&get("line")?)?,
            names: tuple(&get("names")?, string)?,
        })
    }

    fn class_sig_from(&self, object: &Bound<'py, PyAny>) -> PyResult<ClassSig> {
        let get = |name: &str| attr(object, &self.class_sig, "ClassSig", name);
        Ok(ClassSig {
            name: string(&get("name")?)?,
            id: string(&get("id")?)?,
            fields: dict(&get("fields")?, string, string)?,
            required: int(&get("required")?)?,
            methods: dict(&get("methods")?, string, |item| self.export_sig_from(item))?,
        })
    }

    // --- Rust -> Python -------------------------------------------------------

    pub fn table_to(&self, table: &PineTypeTable) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let bindings = PyDict::new(py);
        for (scope, names) in &table.bindings {
            let inner = PyDict::new(py);
            for (name, binding) in names {
                inner.set_item(name, self.binding_to(binding)?)?;
            }
            bindings.set_item(scope, inner)?;
        }
        let funcs = PyDict::new(py);
        for (key, sig) in &table.funcs {
            funcs.set_item(key, self.func_sig_to(sig)?)?;
        }
        let contexts = PyDict::new(py);
        for (key, result) in &table.contexts {
            contexts.set_item(context_key_to(py, key)?, self.context_to(result)?)?;
        }
        let calls = table.calls.iter().map(|call| self.call_site_to(call));
        let diags = table.diags.iter().map(|diag| self.diag_to(diag));
        let groups = PyDict::new(py);
        for (key, impls) in &table.groups {
            groups.set_item(key, self.impl_sigs_to(impls)?)?;
        }
        let deps = PyDict::new(py);
        for (path, record) in &table.deps {
            deps.set_item(path, self.dep_record_to(record)?)?;
        }
        let classes = PyDict::new(py);
        for (name, id) in &table.classes {
            classes.set_item(name, id)?;
        }
        let class_sigs = PyDict::new(py);
        for (id, sig) in &table.class_sigs {
            class_sigs.set_item(id, self.class_sig_to(sig)?)?;
        }
        let extensions = PyDict::new(py);
        for (id, methods) in &table.extensions {
            extensions.set_item(id, self.export_sigs_to(methods)?)?;
        }
        let call_pos = PyDict::new(py);
        for (nid, position) in &table.call_pos {
            call_pos.set_item(nid, PyTuple::new(py, [position.0, position.1])?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("module_path", &table.module_path)?;
        kwargs.set_item("bindings", bindings)?;
        kwargs.set_item("funcs", funcs)?;
        kwargs.set_item("contexts", contexts)?;
        kwargs.set_item("calls", PyList::new(py, calls.collect::<PyResult<Vec<_>>>()?)?)?;
        kwargs.set_item("diags", PyList::new(py, diags.collect::<PyResult<Vec<_>>>()?)?)?;
        kwargs.set_item("groups", groups)?;
        kwargs.set_item("deps", deps)?;
        kwargs.set_item("classes", classes)?;
        kwargs.set_item("class_sigs", class_sigs)?;
        kwargs.set_item("extensions", extensions)?;
        kwargs.set_item("pins_suppressed", self.opt_diag_to(table.pins_suppressed.as_ref())?)?;
        kwargs.set_item("exportable", PyFrozenSet::new(py, &table.exportable)?)?;
        kwargs.set_item("call_pos", call_pos)?;
        self.table.call((), Some(&kwargs))
    }

    fn unknown_to(&self, unknown: Option<&Unknown>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let Some(unknown) = unknown else { return Ok(py.None().into_bound(py)) };
        let kwargs = PyDict::new(py);
        kwargs.set_item("reason", &unknown.reason)?;
        kwargs.set_item("line", unknown.line)?;
        kwargs.set_item("col", unknown.col)?;
        kwargs.set_item("detail", &unknown.detail)?;
        self.unknown.call((), Some(&kwargs))
    }

    fn binding_to(&self, binding: &Binding) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.table.py());
        kwargs.set_item("name", &binding.name)?;
        kwargs.set_item("ty", &binding.ty)?;
        kwargs.set_item("line", binding.line)?;
        kwargs.set_item("unknown", self.unknown_to(binding.unknown.as_ref())?)?;
        kwargs.set_item("series", binding.series)?;
        kwargs.set_item("qual", binding.qual)?;
        kwargs.set_item("depth", binding.depth)?;
        kwargs.set_item("persistent", binding.persistent)?;
        kwargs.set_item("decl", binding.decl)?;
        kwargs.set_item("min_read", binding.min_read)?;
        self.binding.call((), Some(&kwargs))
    }

    fn func_sig_to(&self, sig: &FuncSig) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("name", &sig.name)?;
        kwargs.set_item("params", PyList::new(py, &sig.params)?)?;
        kwargs.set_item("ret", &sig.ret)?;
        kwargs.set_item("line", sig.line)?;
        self.func_sig.call((), Some(&kwargs))
    }

    fn context_to(&self, result: &ContextResult) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let pins = PyDict::new(py);
        for (nid, pin) in &result.pins {
            pins.set_item(nid, pin)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("cid", result.cid)?;
        kwargs.set_item("key", &result.key)?;
        kwargs.set_item("params", PyTuple::new(py, &result.params)?)?;
        kwargs.set_item("ret", &result.ret)?;
        kwargs.set_item("quals", PyTuple::new(py, &result.quals)?)?;
        kwargs.set_item("ret_qual", result.ret_qual)?;
        kwargs.set_item("pins", pins)?;
        self.context_result.call((), Some(&kwargs))
    }

    fn call_site_to(&self, call: &CallSite) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.table.py());
        kwargs.set_item("callee", &call.callee)?;
        kwargs.set_item("line", call.line)?;
        kwargs.set_item("col", call.col)?;
        kwargs.set_item("argc", call.argc)?;
        kwargs.set_item("ty", &call.ty)?;
        kwargs.set_item("pin", &call.pin)?;
        self.call_site.call((), Some(&kwargs))
    }

    fn diag_to(&self, diag: &Diag) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.table.py());
        kwargs.set_item("message", &diag.message)?;
        kwargs.set_item("line", diag.line)?;
        kwargs.set_item("col", diag.col)?;
        kwargs.set_item("origin", self.unknown_to(diag.origin.as_ref())?)?;
        kwargs.set_item("fix", &diag.fix)?;
        kwargs.set_item("end_line", diag.end_line)?;
        kwargs.set_item("end_col", diag.end_col)?;
        self.diag.call((), Some(&kwargs))
    }

    fn opt_diag_to(&self, diag: Option<&Diag>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        diag.map_or_else(|| Ok(py.None().into_bound(py)), |diag| self.diag_to(diag))
    }

    fn dep_record_to(&self, record: &DepRecord) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.table.py());
        kwargs.set_item("path", &record.path)?;
        kwargs.set_item("mtime_ns", record.mtime_ns)?;
        kwargs.set_item("size", record.size)?;
        kwargs.set_item("digest", &record.digest)?;
        kwargs.set_item("routes", &record.routes)?;
        self.dep_record.call((), Some(&kwargs))
    }

    fn impl_sigs_to(&self, impls: &[ImplSig]) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let mut items = Vec::with_capacity(impls.len());
        for imp in impls {
            let kwargs = PyDict::new(py);
            kwargs.set_item("params", PyTuple::new(py, &imp.params)?)?;
            kwargs.set_item("required", imp.required)?;
            kwargs.set_item("open_ended", imp.open_ended)?;
            kwargs.set_item("ret", &imp.ret)?;
            kwargs.set_item("fits", &imp.fits)?;
            kwargs.set_item("names", PyTuple::new(py, &imp.names)?)?;
            items.push(self.impl_sig.call((), Some(&kwargs))?);
        }
        PyTuple::new(py, items)?.into_bound_py_any(py)
    }

    fn export_sig_to(&self, sig: &ExportSig) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("name", &sig.name)?;
        kwargs.set_item("kind", &sig.kind)?;
        kwargs.set_item("params", PyTuple::new(py, &sig.params)?)?;
        kwargs.set_item("required", sig.required)?;
        kwargs.set_item("open_ended", sig.open_ended)?;
        kwargs.set_item("ret", &sig.ret)?;
        kwargs.set_item("annotated", sig.annotated)?;
        kwargs.set_item("impls", self.impl_sigs_to(&sig.impls)?)?;
        kwargs.set_item("line", sig.line)?;
        kwargs.set_item("names", PyTuple::new(py, &sig.names)?)?;
        self.export_sig.call((), Some(&kwargs))
    }

    fn export_sigs_to(&self, sigs: &IndexMap<String, ExportSig>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(self.table.py());
        for (name, sig) in sigs {
            dict.set_item(name, self.export_sig_to(sig)?)?;
        }
        Ok(dict.into_any())
    }

    fn class_sig_to(&self, sig: &ClassSig) -> PyResult<Bound<'py, PyAny>> {
        let py = self.table.py();
        let fields = PyDict::new(py);
        for (name, ty) in &sig.fields {
            fields.set_item(name, ty)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("name", &sig.name)?;
        kwargs.set_item("id", &sig.id)?;
        kwargs.set_item("fields", fields)?;
        kwargs.set_item("required", sig.required)?;
        kwargs.set_item("methods", self.export_sigs_to(&sig.methods)?)?;
        self.class_sig.call((), Some(&kwargs))
    }
}

fn context_key_to<'py>(py: Python<'py>, key: &ContextKey) -> PyResult<Bound<'py, PyTuple>> {
    let env = key
        .env
        .iter()
        .map(|(name, ty, qual)| (name, ty, qual).into_pyobject(py))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(
        py,
        [
            key.key.as_str().into_bound_py_any(py)?,
            PyTuple::new(py, &key.params)?.into_any(),
            key.origin.into_bound_py_any(py)?,
            key.def.into_bound_py_any(py)?,
            PyTuple::new(py, env)?.into_any(),
            PyTuple::new(py, &key.quals)?.into_any(),
        ],
    )
}

fn unsupported(message: String) -> PyErr {
    Unsupported::new_err(message)
}

fn mismatch(what: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let name = value.get_type().name().map(|name| name.to_string()).unwrap_or_default();
    unsupported(format!("type table: {what} is a {name}"))
}

/// A field of a dataclass instance of exactly `class`.
fn attr<'py>(
    object: &Bound<'py, PyAny>,
    class: &Bound<'py, PyAny>,
    what: &str,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if !object.get_type().is(class) {
        return Err(mismatch(what, object));
    }
    object.getattr(name)
}

fn string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(mismatch("a str", value));
    }
    value.extract().map_err(|_| unsupported("type table: a str Rust cannot hold".into()))
}

fn int(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(mismatch("an int", value));
    }
    value.extract().map_err(|_| unsupported(format!("type table: int {value} beyond i64")))
}

fn qualifier(value: &Bound<'_, PyAny>) -> PyResult<u8> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(mismatch("a qualifier", value));
    }
    value.extract().map_err(|_| unsupported(format!("type table: qualifier {value}")))
}

fn boolean(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.cast_exact::<PyBool>().map(|flag| flag.is_true()).map_err(|_| mismatch("a bool", value))
}

fn optional<'py, T>(
    value: &Bound<'py, PyAny>,
    item: impl Fn(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    if value.is_none() { Ok(None) } else { item(value).map(Some) }
}

fn list<'py, T>(
    value: &Bound<'py, PyAny>,
    item: impl Fn(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    let items = value.cast_exact::<PyList>().map_err(|_| mismatch("a list", value))?;
    items.iter().map(|element| item(&element)).collect()
}

fn tuple<'py, T>(
    value: &Bound<'py, PyAny>,
    item: impl Fn(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    let items = value.cast_exact::<PyTuple>().map_err(|_| mismatch("a tuple", value))?;
    items.iter().map(|element| item(&element)).collect()
}

fn dict<'py, K: Hash + Eq, V>(
    value: &Bound<'py, PyAny>,
    key: impl Fn(&Bound<'py, PyAny>) -> PyResult<K>,
    item: impl Fn(&Bound<'py, PyAny>) -> PyResult<V>,
) -> PyResult<IndexMap<K, V>> {
    let items = value.cast_exact::<PyDict>().map_err(|_| mismatch("a dict", value))?;
    let mut out = IndexMap::with_capacity(items.len());
    for (name, element) in items.iter() {
        if out.insert(key(&name)?, item(&element)?).is_some() {
            return Err(unsupported(format!("type table: two keys convert to {name}")));
        }
    }
    Ok(out)
}

fn frozenset(value: &Bound<'_, PyAny>) -> PyResult<BTreeSet<String>> {
    let items = value.cast_exact::<PyFrozenSet>().map_err(|_| mismatch("a frozenset", value))?;
    items.iter().map(|element| string(&element)).collect()
}

fn context_key(value: &Bound<'_, PyAny>) -> PyResult<ContextKey> {
    let items = value.cast_exact::<PyTuple>().map_err(|_| mismatch("a context key", value))?;
    if items.len() != 6 {
        return Err(unsupported(format!("type table: a context key of {} items", items.len())));
    }
    Ok(ContextKey {
        key: string(&items.get_item(0)?)?,
        params: tuple(&items.get_item(1)?, string)?,
        origin: int(&items.get_item(2)?)?,
        def: optional(&items.get_item(3)?, int)?,
        env: tuple(&items.get_item(4)?, |entry| {
            let entry =
                entry.cast_exact::<PyTuple>().map_err(|_| mismatch("an env entry", entry))?;
            if entry.len() != 3 {
                return Err(mismatch("an env entry", entry.as_any()));
            }
            Ok((
                string(&entry.get_item(0)?)?,
                string(&entry.get_item(1)?)?,
                qualifier(&entry.get_item(2)?)?,
            ))
        })?,
        quals: tuple(&items.get_item(5)?, qualifier)?,
    })
}

/// Convert a `PineTypeTable` to its native form and back, the way the bridge
/// converts a tree's table for a run whose steps read it. For the parity tests.
#[pyfunction]
#[pyo3(name = "_type_table")]
pub fn round_trip<'py>(py: Python<'py>, table: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut from = FromPython::new().native_type_tables(py)?;
    let native = from.stamp(table)?;
    if !matches!(native, Value::Native(_)) {
        return Err(unsupported("type table: not a PineTypeTable".into()));
    }
    ToPython::new(py)?.with_objects(from.objects).stamp(&from.ast, &native)
}

/// Number a tree with the native `assign_node_ids`; a new tree and the next unused id.
#[pyfunction]
#[pyo3(name = "_node_ids", signature = (tree, start=0))]
pub fn node_ids<'py>(
    py: Python<'py>,
    tree: &Bound<'py, PyAny>,
    start: i64,
) -> PyResult<(Bound<'py, PyAny>, i64)> {
    let mut from = FromPython::new();
    let root: NodeId = from.node(tree)?;
    let mut ast = from.ast;
    let next = assign_node_ids(&mut ast, root, start);
    Ok((ToPython::new(py)?.with_objects(from.objects).node(&ast, root)?, next))
}
