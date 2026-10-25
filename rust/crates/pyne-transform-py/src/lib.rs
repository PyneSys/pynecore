//! `pynecore.transformers._native`: the native transform, called from the Python pipeline.

mod convert;
mod fold;
mod qualifier;
mod rules;
mod slot_layout;
mod type_table;

use std::cell::RefCell;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyDict;

use convert::{FromPython, ToPython};
use pyne_transform::slot_layout::ModuleLayout;
use slot_layout::LayoutClasses;

create_exception!(
    _native,
    Unsupported,
    PyException,
    "A construct the native transform does not handle; the Python path runs instead."
);
create_exception!(
    _native,
    Rejected,
    PyException,
    "The module is invalid; the Python step raises the error the author sees."
);

fn unsupported(err: pyne_ast::Unsupported) -> PyErr {
    Unsupported::new_err(err.0)
}

/// Parse a module with the native parser into Python ``ast`` objects.
#[pyfunction]
fn parse<'py>(py: Python<'py>, source: &str) -> PyResult<Bound<'py, PyAny>> {
    let (ast, module) = pyne_parse::parse_module(source).map_err(unsupported)?;
    ToPython::new(py)?.node(&ast, module)
}

/// Convert a tree into the arena and back; what crosses the boundary unchanged.
#[pyfunction]
fn roundtrip<'py>(py: Python<'py>, tree: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut from = FromPython::new();
    let root = from.node(tree)?;
    ToPython::new(py)?.with_objects(from.objects).node(&from.ast, root)
}

/// ``ast.dump(tree, include_attributes=...)`` written by the native side.
#[pyfunction]
#[pyo3(signature = (tree, include_attributes=false))]
fn dump(tree: &Bound<'_, PyAny>, include_attributes: bool) -> PyResult<String> {
    let mut from = FromPython::new();
    let root = from.node(tree)?;
    Ok(pyne_ast::dump::dump(&from.ast, root, include_attributes))
}

/// The host the Python pipeline provides: an object with the [`pyne_transform::Host`]
/// methods (``native._Host``).
struct PyHost<'py> {
    object: Option<Bound<'py, PyAny>>,
    /// The running interpreter's `(major, minor)`
    python: (u32, u32),
}

impl PyHost<'_> {
    fn ask<T: for<'a> FromPyObject<'a, 'a>>(
        &self,
        method: &str,
        args: impl for<'a> pyo3::call::PyCallArgs<'a>,
    ) -> Result<T, pyne_ast::Unsupported> {
        let Some(object) = &self.object else {
            return Err(pyne_ast::Unsupported(format!("no host for {method}")));
        };
        object
            .call_method1(method, args)
            .and_then(|answer| answer.extract::<T>().map_err(Into::into))
            .map_err(|err| pyne_ast::Unsupported(format!("host {method}: {err}")))
    }
}

/// The running interpreter's `(major, minor)`.
fn interpreter(py: Python<'_>) -> (u32, u32) {
    let version = py.version_info();
    (u32::from(version.major), u32::from(version.minor))
}

impl pyne_transform::Host for PyHost<'_> {
    fn python(&self) -> (u32, u32) {
        self.python
    }

    fn import_library(&self, module: &str) -> Result<bool, pyne_ast::Unsupported> {
        self.ask("import_library", (module,))
    }

    fn library_serves(&self, module: &str, attr: &str) -> Result<bool, pyne_ast::Unsupported> {
        self.ask("library_serves", (module, attr))
    }

    fn has_module_properties(&self, module: &str) -> Result<bool, pyne_ast::Unsupported> {
        self.ask("has_module_properties", (module,))
    }

    fn module_property(&self, module: &str, name: &str) -> Result<bool, pyne_ast::Unsupported> {
        self.ask("module_property", (module, name))
    }

    fn module_property_type(
        &self,
        module: &str,
        name: &str,
    ) -> Result<Option<String>, pyne_ast::Unsupported> {
        self.ask("module_property_type", (module, name))
    }

    fn star_exports(&self, module: &str) -> Result<Option<Vec<String>>, pyne_ast::Unsupported> {
        self.ask("star_exports", (module,))
    }

    fn lib_name(
        &self,
        name: &str,
    ) -> Result<Option<pyne_transform::data::LibName>, pyne_ast::Unsupported> {
        type Entry = (String, Option<String>, bool, Option<String>, Vec<Option<String>>);
        let entry: Option<Entry> = self.ask("lib_name", (name,))?;
        Ok(entry.map(|(kind, ty, callable, ret, impl_rets)| pyne_transform::data::LibName {
            kind,
            ty,
            callable,
            ret,
            impl_rets,
        }))
    }

    fn module_origin(
        &self,
        name: &str,
        importer: &str,
        level: i64,
    ) -> Result<Option<String>, pyne_ast::Unsupported> {
        self.ask("module_origin", (name, importer, level))
    }

    fn read_source(
        &self,
        path: &str,
    ) -> Result<Option<pyne_transform::SourceRead>, pyne_ast::Unsupported> {
        let read: Option<(String, PyBackedBytes, i64, i64)> = self.ask("read_source", (path,))?;
        Ok(read.map(|(resolved, data, mtime_ns, size)| pyne_transform::SourceRead {
            resolved,
            data: data.to_vec(),
            mtime_ns,
            size,
        }))
    }

    fn interface(
        &self,
        path: &str,
    ) -> Result<Option<pyne_transform::pine_type_table::ModuleInterface>, pyne_ast::Unsupported>
    {
        let text: Option<String> = self.ask("interface", (path,))?;
        text.map(|text| {
            pyne_transform::pine_type_table::ModuleInterface::from_json(&text)
                .map_err(|err| pyne_ast::Unsupported(format!("interface of {path}: {err}")))
        })
        .transpose()
    }

    fn closes_cycle(&self, path: &str) -> Result<bool, pyne_ast::Unsupported> {
        self.ask("closes_cycle", (path,))
    }

    fn begin_analysis(&self, path: &str) -> Result<(), pyne_ast::Unsupported> {
        self.ask::<Option<bool>>("begin_analysis", (path,)).map(|_| ())
    }

    fn end_analysis(&self) -> Result<(), pyne_ast::Unsupported> {
        self.ask::<Option<bool>>("end_analysis", ()).map(|_| ())
    }

    fn module_spec_origin(&self, dotted: &str) -> Result<Option<String>, pyne_ast::Unsupported> {
        self.ask("module_spec_origin", (dotted,))
    }

    fn package_member(
        &self,
        directory: &str,
        dotted: &str,
    ) -> Result<Option<String>, pyne_ast::Unsupported> {
        self.ask("package_member", (directory, dotted))
    }

    fn lib_route(&self, path: &str) -> Result<Option<String>, pyne_ast::Unsupported> {
        self.ask("lib_route", (path,))
    }

    fn call_interface(
        &self,
        path: &str,
        settled: bool,
    ) -> Result<Option<pyne_transform::pine_type_table::CallInterface>, pyne_ast::Unsupported> {
        let text: Option<String> = self.ask("interface", (path, settled))?;
        text.map(|text| {
            pyne_transform::pine_type_table::CallInterface::from_json(&text)
                .map_err(|err| pyne_ast::Unsupported(format!("interface of {path}: {err}")))
        })
        .transpose()
    }

    fn object_route(
        &self,
        module: &str,
        names: &[String],
        star: bool,
    ) -> Result<(String, Vec<Option<String>>), pyne_ast::Unsupported> {
        self.ask("object_route", (module, names.to_vec(), star))
    }

    fn source_key(&self, path: &str) -> Result<(String, bool), pyne_ast::Unsupported> {
        self.ask("source_key", (path,))
    }

    fn source_record(
        &self,
        path: &str,
    ) -> Result<Option<pyne_transform::pine_type_table::DepRecord>, pyne_ast::Unsupported> {
        let record: Option<(String, i64, i64, String, String)> =
            self.ask("source_record", (path,))?;
        Ok(record.map(|(path, mtime_ns, size, digest, routes)| {
            pyne_transform::pine_type_table::DepRecord { path, mtime_ns, size, digest, routes }
        }))
    }
}

/// Run native steps over a module, in order.
///
/// :param tree: The module.
/// :param steps: Step names as ``pipeline.py`` declares them.
/// :param context: ``path``, ``pyne_mode``, ``source``, ``user_code``, ``na_bool``,
///     ``sequential``, ``security_slice_disabled``, ``emit_layout``, ``host``, and
///     ``slot_layout``: the pipeline's ``ModuleLayout``,
///     read when a step of the run uses it (a fresh one when the context has none).
///     The layout such a run leaves is put into the context as
///     ``slot_layout_result``, a new ``ModuleLayout``.
/// :return: A new module object; the input tree and layout are left as they were.
/// :raises Unsupported: When a step meets something its port does not handle.
/// :raises Rejected: When a step finds the module invalid.
#[pyfunction]
fn run<'py>(
    py: Python<'py>,
    tree: &Bound<'py, PyAny>,
    steps: Vec<String>,
    context: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let get = |name: &str| context.get_item(name);
    let host = PyHost { object: get("host")?, python: interpreter(py) };
    let names: Vec<&str> = steps.iter().map(String::as_str).collect();
    let mut from = FromPython::new();
    if pyne_transform::reads_type_table(&names) {
        from = from.native_type_tables(py)?;
    }
    let root = from.node(tree)?;
    // The layout's template expressions may be nodes of the tree: converted after it
    let classes = if pyne_transform::uses_slot_layout(&names) {
        Some(LayoutClasses::import(py)?)
    } else {
        None
    };
    let layout = match (&classes, get("slot_layout")?) {
        (Some(classes), Some(object)) if !object.is_none() => {
            classes.layout_from(&mut from, &object, tree, root)?
        }
        _ => ModuleLayout::default(),
    };
    let ctx = pyne_transform::Context {
        path: match get("path")? {
            Some(value) => value.extract()?,
            None => String::new(),
        },
        pyne_mode: match get("pyne_mode")? {
            Some(value) => value.extract()?,
            None => None,
        },
        source: match get("source")? {
            Some(value) => value.extract()?,
            None => String::new(),
        },
        user_code: match get("user_code")? {
            Some(value) => value.extract()?,
            None => true,
        },
        na_bool: match get("na_bool")? {
            Some(value) => value.extract()?,
            None => false,
        },
        sequential: match get("sequential")? {
            Some(value) => value.extract()?,
            None => false,
        },
        security_slice_disabled: match get("security_slice_disabled")? {
            Some(value) => value.extract()?,
            None => false,
        },
        emit_layout: match get("emit_layout")? {
            Some(value) => value.extract()?,
            None => true,
        },
        host: &host,
        slot_layout: RefCell::new(layout),
    };
    let mut ast = from.ast;
    let root =
        pyne_transform::run_steps(&mut ast, root, &names, &ctx).map_err(|err| match err {
            pyne_transform::Error::Unsupported(message) => Unsupported::new_err(message),
            pyne_transform::Error::Rejected(message) => Rejected::new_err(message),
        })?;
    let mut to = ToPython::new(py)?.with_objects(from.objects);
    let module = to.node(&ast, root)?;
    if let Some(classes) = classes {
        let layout = classes.layout_to(&mut to, &ast, &ctx.slot_layout.borrow())?;
        context.set_item("slot_layout_result", layout)?;
    }
    Ok(module)
}

/// ``ast.unparse(node)`` of an expression written by the native side.
#[pyfunction]
#[pyo3(name = "_unparse")]
fn unparse(node: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut from = FromPython::new();
    let root = from.node(node)?;
    pyne_transform::unparse::unparse(&from.ast, root, interpreter(node.py())).map_err(unsupported)
}

/// The ``ported.json`` manifest the extension was built with.
#[pyfunction]
fn ported_json() -> &'static str {
    pyne_transform::PORTED
}

/// The names of the steps with a native port.
#[pyfunction]
fn steps() -> Vec<&'static str> {
    pyne_transform::STEPS.iter().map(|step| step.name).collect()
}

/// The names of the ported steps that read or fill the pipeline's slot layout.
#[pyfunction]
fn layout_steps() -> Vec<&'static str> {
    pyne_transform::SLOT_LAYOUT_STEPS.to_vec()
}

/// The callees the native ``FunctionIsolation`` calls plainly
/// (``function_isolation.NON_TRANSFORMABLE_FUNCTIONS``).
#[pyfunction]
fn plain_callees() -> Vec<&'static str> {
    pyne_transform::steps::function_isolation::NON_TRANSFORMABLE_FUNCTIONS.to_vec()
}

/// Per native phase, its rules in order, each with its ``user_code_only``.
#[pyfunction]
fn phases() -> std::collections::HashMap<&'static str, Vec<(&'static str, bool)>> {
    pyne_transform::PHASES
        .iter()
        .map(|phase| {
            let rules = phase.rules.iter().map(|rule| (rule.name, rule.user_code_only)).collect();
            (phase.name, rules)
        })
        .collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("Unsupported", m.py().get_type::<Unsupported>())?;
    m.add("Rejected", m.py().get_type::<Rejected>())?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(dump, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(steps, m)?)?;
    m.add_function(wrap_pyfunction!(layout_steps, m)?)?;
    m.add_function(wrap_pyfunction!(phases, m)?)?;
    m.add_function(wrap_pyfunction!(plain_callees, m)?)?;
    m.add_function(wrap_pyfunction!(ported_json, m)?)?;
    m.add_function(wrap_pyfunction!(unparse, m)?)?;
    m.add_function(wrap_pyfunction!(rules::call, m)?)?;
    m.add_function(wrap_pyfunction!(qualifier::call, m)?)?;
    m.add_function(wrap_pyfunction!(qualifier::registry, m)?)?;
    m.add_function(wrap_pyfunction!(fold::fdlibm_call, m)?)?;
    m.add_function(wrap_pyfunction!(fold::quantize_embed, m)?)?;
    m.add_function(wrap_pyfunction!(fold::pow10, m)?)?;
    m.add_function(wrap_pyfunction!(fold::const_eval, m)?)?;
    m.add_function(wrap_pyfunction!(type_table::round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(type_table::node_ids, m)?)?;
    Ok(())
}
