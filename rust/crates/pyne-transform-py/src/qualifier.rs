//! `_native._qualifier` and `_native._data`: the native `pine_qualifier` port and the
//! compiled-in registries, for the parity tests against Python. Not used by the
//! pipeline.

use pyne_transform::{data, pine_qualifier as qualifier};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{FromPython, ToPython};

fn node<'py>(
    from: &mut FromPython<'py>,
    args: &Bound<'py, PyTuple>,
    index: usize,
) -> PyResult<pyne_ast::NodeId> {
    from.node(&args.get_item(index)?)
}

/// Call one function of the native `pine_qualifier` port by its Python name.
#[pyfunction]
#[pyo3(name = "_qualifier", signature = (name, *args))]
pub fn call<'py>(
    py: Python<'py>,
    name: &str,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let str_arg = |index: usize| -> PyResult<String> { args.get_item(index)?.extract() };
    let qualifiers = |index: usize| -> PyResult<Vec<u8>> { args.get_item(index)?.extract() };
    match name {
        "constants" => {
            let dict = PyDict::new(py);
            for (key, value) in [
                ("CONST", qualifier::CONST),
                ("SIMPLE", qualifier::SIMPLE),
                ("SERIES", qualifier::SERIES),
                ("NOT_READ", qualifier::NOT_READ),
            ] {
                dict.set_item(key, value)?;
            }
            for (key, text) in [
                ("SERIES_LENGTH_KEYWORD", qualifier::SERIES_LENGTH_KEYWORD),
                ("SERIES_LEN_ATTR", qualifier::SERIES_LEN_ATTR),
                ("SERIES_LENS_ATTR", qualifier::SERIES_LENS_ATTR),
            ] {
                dict.set_item(key, text)?;
            }
            dict.into_bound_py_any(py)
        }
        "sets" => {
            let dict = PyDict::new(py);
            for (key, items) in [
                ("_SIMPLE_NAMESPACES", qualifier::SIMPLE_NAMESPACES),
                ("_SERIES_NAMESPACES", qualifier::SERIES_NAMESPACES),
                ("_CONST_PRESERVING", qualifier::CONST_PRESERVING),
                ("_SIMPLE_PRESERVING", qualifier::SIMPLE_PRESERVING),
                ("_SIMPLE_CALL_NAMESPACES", qualifier::SIMPLE_CALL_NAMESPACES),
                ("_PLAIN_BUILTINS", qualifier::PLAIN_BUILTINS),
                ("WINDOW_CALLS", qualifier::WINDOW_CALLS),
            ] {
                dict.set_item(key, items.to_vec())?;
            }
            dict.into_bound_py_any(py)
        }
        "lib_value_qualifier" => {
            qualifier::lib_value_qualifier(&str_arg(0)?, &str_arg(1)?).into_bound_py_any(py)
        }
        "lib_call_qualifier" => {
            let argc: Option<usize> = args.get_item(2)?.extract()?;
            qualifier::lib_call_qualifier(&str_arg(0)?, qualifiers(1)?, argc).into_bound_py_any(py)
        }
        "builtin_call_qualifier" => {
            qualifier::builtin_call_qualifier(&str_arg(0)?, qualifiers(1)?).into_bound_py_any(py)
        }
        "window_length" => {
            let mut from = FromPython::new();
            let call = node(&mut from, args, 1)?;
            match qualifier::window_length(&from.ast, &str_arg(0)?, call) {
                Some(length) => from
                    .source(length)
                    .cloned()
                    .ok_or_else(|| PyKeyError::new_err("a node built natively")),
                None => Ok(py.None().into_bound(py)),
            }
        }
        "get_series_len" => {
            let mut from = FromPython::new();
            let call = node(&mut from, args, 0)?;
            qualifier::get_series_len(&from.ast, call).into_bound_py_any(py)
        }
        "get_series_lens" => {
            let mut from = FromPython::new();
            let call = node(&mut from, args, 0)?;
            match qualifier::get_series_lens(&from.ast, call) {
                Some(items) => {
                    let dict = PyDict::new(py);
                    for (context, verdict) in items {
                        dict.set_item(context, verdict)?;
                    }
                    dict.into_bound_py_any(py)
                }
                None => Ok(py.None().into_bound(py)),
            }
        }
        "set_series_len" | "set_series_lens" => {
            let mut from = FromPython::new();
            let call = node(&mut from, args, 0)?;
            let mut ast = from.ast;
            if name == "set_series_len" {
                let value: Option<bool> = args.get_item(1)?.extract()?;
                qualifier::set_series_len(&mut ast, call, value);
            } else {
                let value: Option<Vec<(i64, bool)>> = match args.get_item(1)? {
                    value if value.is_none() => None,
                    value => {
                        let dict = value.cast::<PyDict>().map_err(PyErr::from)?;
                        Some(
                            dict.iter()
                                .map(|(key, item)| Ok((key.extract()?, item.extract()?)))
                                .collect::<PyResult<_>>()?,
                        )
                    }
                };
                qualifier::set_series_lens(&mut ast, call, value.as_deref());
            }
            ToPython::new(py)?.node(&ast, call)
        }
        _ => Err(PyValueError::new_err(format!("no native qualifier rule {name}"))),
    }
}

/// A section of the registries compiled into the extension, in Python's terms.
#[pyfunction]
#[pyo3(name = "_data")]
pub fn registry<'py>(py: Python<'py>, section: &str) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    match section {
        "names" => {
            for (name, entry) in data::lib_names() {
                let rets = entry.impl_rets.clone();
                let item =
                    (entry.kind.clone(), entry.ty.clone(), entry.callable, entry.ret.clone(), rets);
                dict.set_item(name, item)?;
            }
        }
        "classes" => {
            for (name, fields) in data::classes() {
                dict.set_item(name, fields.clone())?;
            }
        }
        "routes" => {
            for (name, route) in data::routes() {
                dict.set_item(name, route)?;
            }
        }
        "scalar_classes" => {
            for (name, scalar) in data::scalar_classes() {
                dict.set_item(name, scalar)?;
            }
        }
        "star_exports" => {
            for (module, names) in data::star_exports() {
                dict.set_item(module, names.clone())?;
            }
        }
        "module_properties" => {
            for (module, members) in data::module_properties() {
                let inner = PyDict::new(py);
                for (member, kind) in members {
                    inner.set_item(member, kind)?;
                }
                dict.set_item(module, inner)?;
            }
        }
        _ => return Err(PyValueError::new_err(format!("no compiled-in section {section}"))),
    }
    dict.into_bound_py_any(py)
}
