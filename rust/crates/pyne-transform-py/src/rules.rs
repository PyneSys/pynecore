//! `_native._rules`: the native `pine_type_rules` port, one function at a time, for
//! the parity tests against the Python module. Not used by the pipeline.

use std::cell::RefCell;

use pyne_ast::walk::walk;
use pyne_ast::{NodeId, Operator, UnaryOperator, Value};
use pyne_transform::pine_type_rules as rules;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::Unsupported;
use crate::convert::{FromPython, ToPython, constant};

/// The positional arguments of one call.
struct Args<'a, 'py> {
    items: &'a Bound<'py, PyTuple>,
}

impl<'py> Args<'_, 'py> {
    fn get(&self, index: usize) -> PyResult<Bound<'py, PyAny>> {
        self.items.get_item(index)
    }

    fn str(&self, index: usize) -> PyResult<String> {
        self.get(index)?.extract()
    }

    fn opt_str(&self, index: usize) -> PyResult<Option<String>> {
        self.get(index)?.extract()
    }

    fn strs(&self, index: usize) -> PyResult<Vec<String>> {
        self.get(index)?.extract()
    }

    fn flag(&self, index: usize) -> PyResult<bool> {
        self.get(index)?.extract()
    }

    /// An `ast` node converted into `from`, `None` for a Python `None`.
    fn node(&self, from: &mut FromPython<'py>, index: usize) -> PyResult<Option<NodeId>> {
        let value = self.get(index)?;
        if value.is_none() { Ok(None) } else { from.node(&value).map(Some) }
    }

    fn classes(&self, index: usize) -> PyResult<rules::Classes> {
        self.get(index)?.extract()
    }

    /// A `dict[str, str]` in its iteration order.
    fn pairs(&self, index: usize) -> PyResult<Vec<(String, String)>> {
        let value = self.get(index)?;
        let dict = value.cast::<PyDict>().map_err(PyErr::from)?;
        dict.iter().map(|(key, item)| Ok((key.extract()?, item.extract()?))).collect()
    }

    /// `ImplSig` tuples: `(params, required, open_ended, ret, fits, names)`.
    fn impls(&self, index: usize) -> PyResult<Vec<rules::ImplSig>> {
        let mut out = Vec::new();
        for item in self.get(index)?.try_iter()? {
            let item = item?;
            let (params, required, open_ended, ret, fits, names): (
                Vec<String>,
                usize,
                bool,
                String,
                String,
                Vec<String>,
            ) = item.extract()?;
            out.push(rules::ImplSig { params, required, open_ended, ret, fits, names });
        }
        Ok(out)
    }
}

fn unsupported(err: pyne_ast::Unsupported) -> PyErr {
    Unsupported::new_err(err.0)
}

fn rule<'py>(py: Python<'py>, rule: rules::CallRule) -> PyResult<Bound<'py, PyAny>> {
    match rule {
        rules::CallRule::Is(text) => text.into_bound_py_any(py),
        rules::CallRule::ByArity(split) => {
            let dict = PyDict::new(py);
            for &(count, inner) in split {
                dict.set_item(count, self::rule(py, inner)?)?;
            }
            dict.into_bound_py_any(py)
        }
        rules::CallRule::FirstOf(options) => {
            let items =
                options.iter().map(|&inner| self::rule(py, inner)).collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_bound_py_any(py)
        }
    }
}

fn rule_table<'py>(
    py: Python<'py>,
    table: &[(&str, rules::CallRule)],
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for &(name, entry) in table {
        dict.set_item(name, rule(py, entry)?)?;
    }
    dict.into_bound_py_any(py)
}

fn str_table<'py>(py: Python<'py>, table: &[(&str, &str)]) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for &(name, value) in table {
        dict.set_item(name, value)?;
    }
    dict.into_bound_py_any(py)
}

fn list_table<'py>(py: Python<'py>, table: &[(&str, &[&str])]) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for &(name, values) in table {
        dict.set_item(name, values.to_vec())?;
    }
    dict.into_bound_py_any(py)
}

/// A stamped attribute value as Python reads it, node references as their originals.
fn value<'py>(
    py: Python<'py>,
    from: &FromPython<'py>,
    value: &Value,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::None => py.None().into_bound(py),
        Value::Bool(flag) => flag.into_bound_py_any(py)?,
        Value::Int(pyne_ast::Int::Small(number)) => number.into_bound_py_any(py)?,
        Value::Int(pyne_ast::Int::Big(digits)) => {
            py.import("builtins")?.getattr("int")?.call1((digits,))?
        }
        Value::Float(number) => number.into_bound_py_any(py)?,
        Value::Str(text) => text.into_bound_py_any(py)?,
        Value::Tuple(items) => {
            let items = items
                .iter()
                .map(|item| self::value(py, from, item))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, items)?.into_any()
        }
        Value::List(items) => {
            let items = items
                .iter()
                .map(|item| self::value(py, from, item))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Dict(items) => {
            let dict = PyDict::new(py);
            for (key, item) in items {
                dict.set_item(self::value(py, from, key)?, self::value(py, from, item)?)?;
            }
            dict.into_any()
        }
        Value::Node(id) => original(from, *id)?,
        Value::Opaque(index) => from.objects[*index as usize].clone(),
        Value::Native(index) => {
            return Err(Unsupported::new_err(format!("native object {index}")));
        }
    })
}

fn original<'py>(from: &FromPython<'py>, id: NodeId) -> PyResult<Bound<'py, PyAny>> {
    from.source(id).cloned().ok_or_else(|| PyKeyError::new_err("a node built natively"))
}

fn context_map<'py>(
    py: Python<'py>,
    map: Option<Vec<(i64, Option<String>)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(items) = map else {
        return Ok(py.None().into_bound(py));
    };
    let dict = PyDict::new(py);
    for (context, text) in items {
        dict.set_item(context, text)?;
    }
    dict.into_bound_py_any(py)
}

/// `ty_of`: a Python callable over the original nodes, or `get_ty` when `None`.
struct TyOf<'a, 'py> {
    callable: Option<Bound<'py, PyAny>>,
    from: &'a FromPython<'py>,
    failure: RefCell<Option<PyErr>>,
}

impl TyOf<'_, '_> {
    fn ty(&self, id: NodeId) -> String {
        let Some(callable) = &self.callable else {
            return rules::get_ty(&self.from.ast, id).to_string();
        };
        let answer = original(self.from, id)
            .and_then(|node| callable.call1((node,)))
            .and_then(|ty| ty.extract::<String>());
        answer.unwrap_or_else(|err| {
            self.failure.borrow_mut().get_or_insert(err);
            rules::UNKNOWN.to_string()
        })
    }

    fn finish<T>(self, result: Result<T, pyne_ast::Unsupported>) -> PyResult<T> {
        if let Some(err) = self.failure.into_inner() {
            return Err(err);
        }
        result.map_err(unsupported)
    }
}

fn binary(name: &str) -> PyResult<Operator> {
    Operator::from_name(name).ok_or_else(|| PyValueError::new_err(format!("operator {name}")))
}

fn unary(name: &str) -> PyResult<UnaryOperator> {
    UnaryOperator::from_name(name).ok_or_else(|| PyValueError::new_err(format!("operator {name}")))
}

/// Call one function of the native `pine_type_rules` port by its Python name.
#[pyfunction]
#[pyo3(name = "_rules", signature = (name, *args))]
pub fn call<'py>(
    py: Python<'py>,
    name: &str,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let a = Args { items: args };
    match name {
        "constants" => {
            let dict = PyDict::new(py);
            for (key, text) in [
                ("TY_ATTR", rules::TY_ATTR),
                ("PIN_ATTR", rules::PIN_ATTR),
                ("PINS_ATTR", rules::PINS_ATTR),
                ("VECTOR_ATTR", rules::VECTOR_ATTR),
                ("VARYING_ATTR", rules::VARYING_ATTR),
                ("WIDEN_ATTR", rules::WIDEN_ATTR),
                ("NA_CLASS_ATTR", rules::NA_CLASS_ATTR),
                ("WIDEN_SITE_ATTR", rules::WIDEN_SITE_ATTR),
                ("WIDENS_ATTR", rules::WIDENS_ATTR),
                ("WIDEN_FLOAT", rules::WIDEN_FLOAT),
                ("WIDEN_ITEMS", rules::WIDEN_ITEMS),
                ("WIDEN_MATRIX", rules::WIDEN_MATRIX),
                ("WIDEN_STRING_NA", rules::WIDEN_STRING_NA),
                ("WIDEN_NUMBER_NA", rules::WIDEN_NUMBER_NA),
                ("WIDEN_NA_TEXT", rules::WIDEN_NA_TEXT),
                ("WIDEN_ARRAY_NA", rules::WIDEN_ARRAY_NA),
                ("WIDEN_MAP_NA", rules::WIDEN_MAP_NA),
                ("WIDEN_MATRIX_NA", rules::WIDEN_MATRIX_NA),
                ("INT", rules::INT),
                ("FLOAT", rules::FLOAT),
                ("BOOL", rules::BOOL),
                ("STR", rules::STR),
                ("COLOR", rules::COLOR),
                ("OBJECT", rules::OBJECT),
                ("VOID", rules::VOID),
                ("UNKNOWN", rules::UNKNOWN),
                ("TYPELESS", rules::TYPELESS),
                ("LIB_MODULE", rules::LIB_MODULE),
                ("CLASS_SEP", rules::CLASS_SEP),
                ("PINE_LOOP", rules::PINE_LOOP),
                ("NONE_DEFAULT", rules::NONE_DEFAULT),
                ("_DYN_DEFAULT", rules::DYN_DEFAULT),
            ] {
                dict.set_item(key, text)?;
            }
            for (key, char) in [
                ("PIN_ANY", rules::PIN_ANY),
                ("FIT_OMISSIBLE", rules::FIT_OMISSIBLE),
                ("FIT_REQUIRED", rules::FIT_REQUIRED),
                ("FIT_UNSURE", rules::FIT_UNSURE),
            ] {
                dict.set_item(key, char.to_string())?;
            }
            dict.into_bound_py_any(py)
        }
        "sets" => {
            let dict = PyDict::new(py);
            for (key, items) in [
                ("NUMERIC", rules::NUMERIC),
                ("KNOWN", rules::KNOWN),
                ("SCALARS", rules::SCALARS),
                ("PINNABLE", rules::PINNABLE),
                ("CONTAINER_NA_WIDENS", rules::CONTAINER_NA_WIDENS),
                ("NA_LITERALS", rules::NA_LITERALS),
            ] {
                dict.set_item(key, items.to_vec())?;
            }
            dict.into_bound_py_any(py)
        }
        "tables" => {
            let dict = PyDict::new(py);
            dict.set_item("LIB_TYPE_OVERRIDES", rule_table(py, rules::LIB_TYPE_OVERRIDES)?)?;
            dict.set_item("BUILTIN_CALL_TYPES", rule_table(py, rules::BUILTIN_CALL_TYPES)?)?;
            dict.set_item("BUILTIN_NAME_TYPES", str_table(py, rules::BUILTIN_NAME_TYPES)?)?;
            dict.set_item("ANNOTATION_TYPES", str_table(py, rules::ANNOTATION_TYPES)?)?;
            dict.set_item("BUILTIN_NAMESPACES", str_table(py, rules::BUILTIN_NAMESPACES)?)?;
            dict.set_item("NA_TEXT_PARAMS", list_table(py, rules::NA_TEXT_PARAMS)?)?;
            dict.set_item("OVERRIDE_PARAM_NAMES", list_table(py, rules::OVERRIDE_PARAM_NAMES)?)?;
            dict.set_item("NOT_PINE_STRINGS", rules::NOT_PINE_STRINGS.to_vec())?;
            let classes = PyDict::new(py);
            for &class in rules::BUILTIN_CLASS_NAMES {
                classes.set_item(class, rules::builtin_class(class))?;
            }
            dict.set_item("BUILTIN_CLASSES", classes)?;
            let pins = PyDict::new(py);
            for &(shaped, char) in rules::PIN_CONTAINERS {
                pins.set_item(shaped, char.to_string())?;
            }
            dict.set_item("PIN_CONTAINERS", pins)?;
            dict.into_bound_py_any(py)
        }
        "lib_type_override" => match rules::lib_type_override(&a.str(0)?) {
            Some(found) => rule(py, found),
            None => Ok(py.None().into_bound(py)),
        },
        "builtin_call_type" => match rules::builtin_call_type(&a.str(0)?) {
            Some(found) => rule(py, found),
            None => Ok(py.None().into_bound(py)),
        },
        "builtin_name_type" => rules::builtin_name_type(&a.str(0)?).into_bound_py_any(py),
        "override_param_names" => {
            rules::override_param_names(&a.str(0)?).map(<[&str]>::to_vec).into_bound_py_any(py)
        }
        "builtin_class" => rules::builtin_class(&a.str(0)?).into_bound_py_any(py),
        "is_na_text_param" => rules::is_na_text_param(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "is_not_pine_string" => {
            rules::is_not_pine_string(&a.str(0)?, &a.str(1)?).into_bound_py_any(py)
        }
        "is_numeric" => rules::is_numeric(&a.str(0)?).into_bound_py_any(py),
        "is_known" => rules::is_known(&a.str(0)?).into_bound_py_any(py),
        "is_scalar" => rules::is_scalar(&a.str(0)?).into_bound_py_any(py),
        "class_id" => rules::class_id(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "builtin_class_id" => rules::builtin_class_id(&a.str(0)?).into_bound_py_any(py),
        "object_ty" => rules::object_ty(&a.str(0)?).into_bound_py_any(py),
        "array_of" => rules::array_of(&a.str(0)?).into_bound_py_any(py),
        "matrix_of" => rules::matrix_of(&a.str(0)?).into_bound_py_any(py),
        "map_of" => rules::map_of(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "tuple_of" => rules::tuple_of(&a.strs(0)?).into_bound_py_any(py),
        "head" => rules::head(&a.str(0)?).into_bound_py_any(py),
        "is_shaped" => rules::is_shaped(&a.str(0)?).into_bound_py_any(py),
        "is_array" => rules::is_array(&a.str(0)?).into_bound_py_any(py),
        "is_matrix" => rules::is_matrix(&a.str(0)?).into_bound_py_any(py),
        "is_map" => rules::is_map(&a.str(0)?).into_bound_py_any(py),
        "is_tuple" => rules::is_tuple(&a.str(0)?).into_bound_py_any(py),
        "arity" => rules::arity(&a.str(0)?).into_bound_py_any(py),
        "elements_of" => PyTuple::new(py, rules::elements_of(&a.str(0)?))?.into_bound_py_any(py),
        "class_of" => rules::class_of(&a.str(0)?).into_bound_py_any(py),
        "element_of" => rules::element_of(&a.str(0)?).into_bound_py_any(py),
        "key_of" => rules::key_of(&a.str(0)?).into_bound_py_any(py),
        "value_of" => rules::value_of(&a.str(0)?).into_bound_py_any(py),
        "shape_mismatch" => rules::shape_mismatch(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "shape_conflict" => {
            let (left, right) = (a.str(0)?, a.str(1)?);
            rules::shape_conflict(&left, &right).into_bound_py_any(py)
        }
        "render_ty" => rules::render_ty(&a.str(0)?).into_bound_py_any(py),
        "namespace_of" => rules::namespace_of(&a.str(0)?).into_bound_py_any(py),
        "join" => rules::join(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "binop_type" => {
            rules::binop_type(binary(&a.str(0)?)?, &a.str(1)?, &a.str(2)?).into_bound_py_any(py)
        }
        "unaryop_type" => rules::unaryop_type(unary(&a.str(0)?)?, &a.str(1)?).into_bound_py_any(py),
        "compare_type" => rules::compare_type(&a.str(0)?, &a.str(1)?).into_bound_py_any(py),
        "constant_type" => rules::constant_type(&constant(&a.get(0)?)?).into_bound_py_any(py),
        "is_int_typed" => rules::is_int_typed(&a.str(0)?).into_bound_py_any(py),
        "is_na_widen" => rules::is_na_widen(a.opt_str(0)?.as_deref()).into_bound_py_any(py),
        "pin_for" => rules::pin_for(&a.strs(0)?).into_bound_py_any(py),
        "pin_chars" => rules::pin_chars(&a.strs(0)?).into_bound_py_any(py),
        "overload_result" => rules::overload_result(&a.strs(0)?).into_bound_py_any(py),
        "default_fit" => {
            let fit = rules::default_fit(&a.str(0)?, a.opt_str(1)?.as_deref(), a.flag(2)?);
            fit.to_string().into_bound_py_any(py)
        }
        "overload_choice" => {
            rules::overload_choice(&a.impls(0)?, &a.str(1)?, &a.strs(2)?).into_bound_py_any(py)
        }
        "overload_pick" => {
            let impls = a.impls(0)?;
            rules::overload_pick(&impls, &a.str(1)?, &a.strs(2)?).into_bound_py_any(py)
        }
        "overload_binding" => {
            let impls = a.impls(0)?;
            let chosen = rules::overload_binding(
                &impls,
                &a.str(1)?,
                &a.strs(2)?,
                &a.pairs(3)?,
                &a.pairs(4)?,
            );
            chosen.into_bound_py_any(py)
        }
        "overload_bind" => {
            let impls = a.impls(0)?;
            let ret =
                rules::overload_bind(&impls, &a.str(1)?, &a.strs(2)?, &a.pairs(3)?, &a.pairs(4)?);
            ret.into_bound_py_any(py)
        }
        _ => tree_call(py, name, &a),
    }
}

/// The functions that read or write a tree.
fn tree_call<'py>(py: Python<'py>, name: &str, a: &Args<'_, 'py>) -> PyResult<Bound<'py, PyAny>> {
    let mut from = FromPython::new();
    match name {
        "annotation_type" => {
            let node = a.node(&mut from, 0)?;
            let classes = a.classes(1)?;
            let ty = rules::annotation_type(&from.ast, node, &classes).map_err(unsupported)?;
            ty.into_bound_py_any(py)
        }
        "annotation_takes_none" => {
            let node = a.node(&mut from, 0)?;
            let takes = rules::annotation_takes_none(&from.ast, node).map_err(unsupported)?;
            takes.into_bound_py_any(py)
        }
        "bare_wrapper" | "dotted" | "is_na_literal" => {
            let node = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            match name {
                "bare_wrapper" => rules::bare_wrapper(&from.ast, node).into_bound_py_any(py),
                "dotted" => rules::dotted(&from.ast, node).into_bound_py_any(py),
                _ => rules::is_na_literal(&from.ast, node).into_bound_py_any(py),
            }
        }
        "impl_sig" => {
            let func = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let ret = a.str(1)?;
            let callable = Some(a.get(2)?).filter(|value| !value.is_none());
            let classes = a.classes(3)?;
            let ty_of = TyOf { callable, from: &from, failure: RefCell::new(None) };
            let sig = rules::impl_sig(&from.ast, func, &ret, &|id| ty_of.ty(id), &classes);
            let sig = ty_of.finish(sig)?;
            let params = PyTuple::new(py, sig.params)?;
            let names = PyTuple::new(py, sig.names)?;
            (params, sig.required, sig.open_ended, sig.ret, sig.fits, names).into_bound_py_any(py)
        }
        "param_fit" => {
            let arg = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let default = a.node(&mut from, 1)?;
            let callable = Some(a.get(2)?).filter(|value| !value.is_none());
            let classes = a.classes(3)?;
            let ty_of = TyOf { callable, from: &from, failure: RefCell::new(None) };
            let fit = rules::param_fit(&from.ast, arg, default, &|id| ty_of.ty(id), &classes);
            ty_of.finish(fit)?.to_string().into_bound_py_any(py)
        }
        "default_ty" => {
            let default = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let callable = Some(a.get(1)?).filter(|value| !value.is_none());
            let ty_of = TyOf { callable, from: &from, failure: RefCell::new(None) };
            let ty = rules::default_ty(&from.ast, default, &|id| ty_of.ty(id));
            ty_of.finish(Ok(ty))?.into_bound_py_any(py)
        }
        "param_defaults" => {
            let func = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let defaults = rules::param_defaults(&from.ast, func).map_err(unsupported)?;
            let dict = PyDict::new(py);
            for (param, default) in defaults {
                dict.set_item(param, original(&from, default)?)?;
            }
            dict.into_bound_py_any(py)
        }
        "factory_fields" => {
            let module = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let mut classes = Vec::new();
            for class in a.get(1)?.try_iter()? {
                classes.push(from.node(&class?)?);
            }
            let factories = rules::FactoryFields::new(&from.ast, module);
            let out = PyList::empty(py);
            for class in classes {
                let calls = factories
                    .of(&from.ast, class)
                    .into_iter()
                    .map(|call| original(&from, call))
                    .collect::<PyResult<Vec<_>>>()?;
                out.append(calls)?;
            }
            out.into_bound_py_any(py)
        }
        "stamp_lowering" => {
            let root = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let result = a.str(1)?;
            let mut ast = from.ast;
            rules::stamp_lowering(&mut ast, root, &result);
            ToPython::new(py)?.node(&ast, root)
        }
        "stamps" => {
            let node = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let ast = &from.ast;
            let vector = match rules::get_vector(ast, node) {
                Some(vector) => value(py, &from, vector)?,
                None => py.None().into_bound(py),
            };
            let varying = match rules::get_varying(ast, node) {
                Some(sites) => {
                    let sites = sites
                        .into_iter()
                        .map(|site| original(&from, site))
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, sites)?.into_any()
                }
                None => py.None().into_bound(py),
            };
            let site = match rules::get_widen_site(ast, node) {
                Some(site) => original(&from, site)?,
                None => py.None().into_bound(py),
            };
            let stamps = [
                rules::get_ty(ast, node).into_bound_py_any(py)?,
                rules::get_pin(ast, node).into_bound_py_any(py)?,
                context_map(py, rules::get_pins(ast, node))?,
                vector,
                varying,
                rules::get_widen(ast, node).into_bound_py_any(py)?,
                rules::get_na_class(ast, node).into_bound_py_any(py)?,
                site,
                context_map(py, rules::get_widens(ast, node))?,
            ];
            PyTuple::new(py, stamps)?.into_bound_py_any(py)
        }
        "restamp" => {
            // Every stamp a node of the tree carries, read and written back through the
            // accessors, so what comes back is what the accessors round-trip
            let root = a.node(&mut from, 0)?.ok_or_else(|| PyValueError::new_err("a node"))?;
            let mut ast = from.ast;
            for node in walk(&ast, root) {
                let has = |attr: &str| ast.extra(node, attr).is_some();
                let (ty, pin, pins, vector, varying, widen, na_class, site, widens) = (
                    has(rules::TY_ATTR).then(|| rules::get_ty(&ast, node).to_string()),
                    has(rules::PIN_ATTR).then(|| rules::get_pin(&ast, node).map(str::to_string)),
                    has(rules::PINS_ATTR).then(|| rules::get_pins(&ast, node)),
                    has(rules::VECTOR_ATTR).then(|| rules::get_vector(&ast, node).cloned()),
                    has(rules::VARYING_ATTR).then(|| rules::get_varying(&ast, node)),
                    has(rules::WIDEN_ATTR)
                        .then(|| rules::get_widen(&ast, node).map(str::to_string)),
                    has(rules::NA_CLASS_ATTR)
                        .then(|| rules::get_na_class(&ast, node).map(str::to_string)),
                    has(rules::WIDEN_SITE_ATTR).then(|| rules::get_widen_site(&ast, node)),
                    has(rules::WIDENS_ATTR).then(|| rules::get_widens(&ast, node)),
                );
                let extra = &mut ast[node].extra;
                for attr in [
                    rules::TY_ATTR,
                    rules::PIN_ATTR,
                    rules::PINS_ATTR,
                    rules::VECTOR_ATTR,
                    rules::VARYING_ATTR,
                    rules::WIDEN_ATTR,
                    rules::NA_CLASS_ATTR,
                    rules::WIDEN_SITE_ATTR,
                    rules::WIDENS_ATTR,
                ] {
                    extra.remove(attr);
                }
                if let Some(ty) = ty {
                    rules::set_ty(&mut ast, node, &ty);
                }
                if let Some(pin) = pin {
                    rules::set_pin(&mut ast, node, pin.as_deref());
                }
                if let Some(pins) = pins {
                    rules::set_pins(&mut ast, node, pins.as_deref());
                }
                if let Some(vector) = vector {
                    rules::set_vector(&mut ast, node, vector);
                }
                if let Some(varying) = varying {
                    rules::set_varying(&mut ast, node, varying.as_deref());
                }
                if let Some(widen) = widen {
                    rules::set_widen(&mut ast, node, widen.as_deref());
                }
                if let Some(Some(na_class)) = na_class {
                    rules::set_na_class(&mut ast, node, &na_class);
                }
                if let Some(site) = site {
                    rules::set_widen_site(&mut ast, node, site);
                }
                if let Some(widens) = widens {
                    rules::set_widens(&mut ast, node, widens.as_deref());
                }
            }
            ToPython::new(py)?.node(&ast, root)
        }
        _ => Err(PyValueError::new_err(format!("no native rule {name}"))),
    }
}
