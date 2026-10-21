//! `pynecore.transformers._native`: the native transform, called from the Python pipeline.

mod convert;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use convert::{FromPython, ToPython};

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
    ToPython::new(py)?.node(&from.ast, root)
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

impl pyne_transform::Host for PyHost<'_> {
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
}

/// Run native steps over a module, in order.
///
/// :param tree: The module.
/// :param steps: Step names as ``pipeline.py`` declares them.
/// :param context: ``path``, ``pyne_mode``, ``source``, ``user_code``, ``host``.
/// :return: A new module object; the input tree is left as it was.
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
    let host = PyHost { object: get("host")? };
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
        host: &host,
    };
    let mut from = FromPython::new();
    let root = from.node(tree)?;
    let mut ast = from.ast;
    let names: Vec<&str> = steps.iter().map(String::as_str).collect();
    let root =
        pyne_transform::run_steps(&mut ast, root, &names, &ctx).map_err(|err| match err {
            pyne_transform::Error::Unsupported(message) => Unsupported::new_err(message),
            pyne_transform::Error::Rejected(message) => Rejected::new_err(message),
        })?;
    ToPython::new(py)?.node(&ast, root)
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
    m.add_function(wrap_pyfunction!(ported_json, m)?)?;
    Ok(())
}
