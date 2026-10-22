//! Python `ast` objects <-> the arena.
//!
//! Both directions keep identity: a Python node object reached twice becomes one
//! [`NodeId`], and one id becomes one Python object, so a tree that shares a node
//! between two parents, or an attribute that references a node of the tree
//! (`_pine_widen_site`), crosses the boundary as it is.
//!
//! A stamped attribute crosses as a [`Value`] when it is plain data of the exact
//! builtin types (`None`, `bool`, `int`, `float`, `str`, `tuple`, `list`, `dict`)
//! or a node. Any other object (the type pass's `PineTypeTable`, a subclass of
//! a builtin such as a `NamedTuple`) becomes a [`Value::Opaque`]: the object is
//! kept in a per-conversion list, and [`ToPython`] given that list puts the very
//! same object back, so it survives a native step unchanged and with its
//! identity.
//!
//! The type pass's `PineTypeTable` is the exception when the steps of a run read
//! it natively ([`FromPython::native_type_tables`]): it is converted into a
//! [`PineTypeTable`] the tree owns ([`Value::Native`]). [`ToPython`] turns a
//! native table, the converted one or one a native step built, into a real
//! Python `PineTypeTable` (see [`crate::type_table`]).

use std::collections::HashMap;

use pyne_ast::{
    Ast, BoolOperator, CLASSES, CmpOperator, Constant, ExprContext, FieldKind, FieldMut, FieldRef,
    Int, Kind, Loc, NodeId, Operator, UnaryOperator, Value, class_index,
};
use pyne_transform::pine_type_table::PineTypeTable;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyString, PyTuple,
};

use crate::Unsupported;
use crate::type_table::Classes;

const LOCATIONS: [&str; 4] = ["lineno", "col_offset", "end_lineno", "end_col_offset"];

fn unsupported(message: impl Into<String>) -> PyErr {
    Unsupported::new_err(message.into())
}

/// Python objects into one arena.
pub struct FromPython<'py> {
    pub ast: Ast,
    /// The objects the [`Value::Opaque`] values of `ast` index, for [`ToPython::with_objects`]
    pub objects: Vec<Bound<'py, PyAny>>,
    memo: HashMap<usize, NodeId>,
    /// Object address -> its index in `objects`, so one object is one opaque value
    object_memo: HashMap<usize, u32>,
    /// The converted objects, kept alive so their addresses stay unique
    keep: Vec<Bound<'py, PyAny>>,
    /// Set when a `PineTypeTable` is converted rather than handed through
    type_tables: Option<Classes<'py>>,
    /// Table address -> its native value, so one table is one native object
    native_memo: HashMap<usize, Value>,
}

impl<'py> FromPython<'py> {
    pub fn new() -> FromPython<'py> {
        FromPython {
            ast: Ast::new(),
            objects: Vec::new(),
            memo: HashMap::new(),
            object_memo: HashMap::new(),
            keep: Vec::new(),
            type_tables: None,
            native_memo: HashMap::new(),
        }
    }

    /// Convert every `PineTypeTable` into a native one instead of handing it through.
    pub fn native_type_tables(mut self, py: Python<'py>) -> PyResult<FromPython<'py>> {
        self.type_tables = Some(Classes::import(py)?);
        Ok(self)
    }

    /// A stamped attribute's value, as a node's attribute converts.
    pub fn stamp(&mut self, value: &Bound<'py, PyAny>) -> PyResult<Value> {
        self.value(value)
    }

    /// The node an object at this address (`id()`) was converted into, if one was.
    pub fn converted(&self, address: usize) -> Option<NodeId> {
        self.memo.get(&address).copied()
    }

    /// The Python object a node was converted from.
    pub fn source(&self, id: NodeId) -> Option<&Bound<'py, PyAny>> {
        let (&key, _) = self.memo.iter().find(|&(_, &known)| known == id)?;
        self.keep.iter().find(|object| object.as_ptr() as usize == key)
    }

    pub fn node(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<NodeId> {
        let key = obj.as_ptr() as usize;
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        let type_name = obj.get_type().name()?.to_string();
        let index = class_index(&type_name)
            .ok_or_else(|| unsupported(format!("node class {type_name}")))?;
        let class = &CLASSES[index];
        // Registered before the fields so a cycle through an attribute ends here
        let id = self.ast.add(Kind::empty(index));
        self.memo.insert(key, id);
        self.keep.push(obj.clone());
        let mut kind = Kind::empty(index);
        for (field_index, field) in class.fields.iter().enumerate() {
            let value = match obj.getattr(field.name) {
                Ok(value) => value,
                Err(_) => match field.kind {
                    // Older Pythons lack the newer list fields (``type_params``)
                    FieldKind::Nodes | FieldKind::Strs => continue,
                    _ => {
                        return Err(unsupported(format!(
                            "{type_name} without its {} field",
                            field.name
                        )));
                    }
                },
            };
            self.field(&mut kind, field_index, field.kind, &value, &type_name)?;
        }
        let mut loc = Loc::default();
        if class.located {
            let read = |name: &str| -> PyResult<Option<i64>> {
                match obj.getattr(name) {
                    Ok(value) if value.is_none() => Ok(None),
                    Ok(value) => value.extract::<i64>().map(Some),
                    Err(_) => Ok(None),
                }
            };
            loc.lineno = read("lineno")?;
            loc.col_offset = read("col_offset")?;
            loc.end_lineno = read("end_lineno")?;
            loc.end_col_offset = read("end_col_offset")?;
        }
        let mut extras = Vec::new();
        if let Ok(state) = obj.getattr("__dict__") {
            let state = state.cast_into::<PyDict>().map_err(PyErr::from)?;
            for (name, value) in state.iter() {
                let name: String = name.extract()?;
                if class.fields.iter().any(|field| field.name == name)
                    || LOCATIONS.contains(&name.as_str())
                {
                    continue;
                }
                extras.push((name, value));
            }
        }
        let mut extra = std::collections::BTreeMap::new();
        for (name, value) in extras {
            let converted = self
                .value(&value)
                .map_err(|err| unsupported(format!("attribute {name} of {type_name}: {err}")))?;
            extra.insert(name, converted);
        }
        let node = &mut self.ast[id];
        node.kind = kind;
        node.loc = loc;
        node.extra = extra;
        Ok(id)
    }

    fn opt_node(&mut self, value: &Bound<'py, PyAny>) -> PyResult<Option<NodeId>> {
        if value.is_none() { Ok(None) } else { self.node(value).map(Some) }
    }

    fn field(
        &mut self,
        kind: &mut Kind,
        index: usize,
        field_kind: FieldKind,
        value: &Bound<'py, PyAny>,
        type_name: &str,
    ) -> PyResult<()> {
        match field_kind {
            FieldKind::Node => {
                let id = self.node(value)?;
                let FieldMut::Node(slot) = kind.field_mut(index) else { unreachable!() };
                *slot = id;
                return Ok(());
            }
            FieldKind::OptNode => {
                let id = self.opt_node(value)?;
                let FieldMut::OptNode(slot) = kind.field_mut(index) else { unreachable!() };
                *slot = id;
                return Ok(());
            }
            FieldKind::Nodes => {
                let mut ids = Vec::new();
                for item in value.try_iter()? {
                    ids.push(self.node(&item?)?);
                }
                let FieldMut::Nodes(slot) = kind.field_mut(index) else { unreachable!() };
                *slot = ids;
                return Ok(());
            }
            FieldKind::OptNodes => {
                let mut ids = Vec::new();
                for item in value.try_iter()? {
                    ids.push(self.opt_node(&item?)?);
                }
                let FieldMut::OptNodes(slot) = kind.field_mut(index) else { unreachable!() };
                *slot = ids;
                return Ok(());
            }
            _ => {}
        }
        match (field_kind, kind.field_mut(index)) {
            (_, FieldMut::Str(slot)) => *slot = string(value)?,
            (_, FieldMut::OptStr(slot)) => {
                *slot = if value.is_none() { None } else { Some(string(value)?) }
            }
            (_, FieldMut::Strs(slot)) => {
                let mut items = Vec::new();
                for item in value.try_iter()? {
                    items.push(string(&item?)?);
                }
                *slot = items;
            }
            (_, FieldMut::Int(slot)) => *slot = value.extract()?,
            (_, FieldMut::OptInt(slot)) => *slot = value.extract()?,
            (_, FieldMut::Constant(slot)) => *slot = constant(value)?,
            (_, FieldMut::ExprContext(slot)) => *slot = enum_member(value, ExprContext::from_name)?,
            (_, FieldMut::Operator(slot)) => *slot = enum_member(value, Operator::from_name)?,
            (_, FieldMut::UnaryOperator(slot)) => {
                *slot = enum_member(value, UnaryOperator::from_name)?
            }
            (_, FieldMut::BoolOperator(slot)) => {
                *slot = enum_member(value, BoolOperator::from_name)?
            }
            (_, FieldMut::CmpOperators(slot)) => {
                let mut ops = Vec::new();
                for item in value.try_iter()? {
                    ops.push(enum_member(&item?, CmpOperator::from_name)?);
                }
                *slot = ops;
            }
            (kind, _) => {
                return Err(PyTypeError::new_err(format!("{type_name}: field kind {kind:?}")));
            }
        }
        Ok(())
    }

    fn value(&mut self, value: &Bound<'py, PyAny>) -> PyResult<Value> {
        if value.is_none() {
            return Ok(Value::None);
        }
        if let Ok(flag) = value.cast::<PyBool>() {
            return Ok(Value::Bool(flag.is_true()));
        }
        // Exact types only: a subclass (an `IntEnum`, a `NamedTuple`) would come
        // back as its base class
        if value.is_exact_instance_of::<PyInt>() {
            return Ok(Value::Int(int(value)?));
        }
        if let Ok(number) = value.cast_exact::<PyFloat>() {
            return Ok(Value::Float(number.value()));
        }
        if value.is_exact_instance_of::<PyString>() {
            return Ok(Value::Str(string(value)?));
        }
        if let Ok(items) = value.cast_exact::<PyTuple>() {
            let mut out = Vec::new();
            for item in items.iter() {
                out.push(self.value(&item)?);
            }
            return Ok(Value::Tuple(out));
        }
        if let Ok(items) = value.cast_exact::<PyList>() {
            let mut out = Vec::new();
            for item in items.iter() {
                out.push(self.value(&item)?);
            }
            return Ok(Value::List(out));
        }
        if let Ok(items) = value.cast_exact::<PyDict>() {
            let mut out = Vec::new();
            for (key, item) in items.iter() {
                out.push((self.value(&key)?, self.value(&item)?));
            }
            return Ok(Value::Dict(out));
        }
        let ast = value.py().import("ast")?;
        if value.is_instance(&ast.getattr("AST")?)? {
            return Ok(Value::Node(self.node(value)?));
        }
        if let Some(classes) = &self.type_tables
            && classes.is_table(value)
        {
            let key = value.as_ptr() as usize;
            if let Some(native) = self.native_memo.get(&key) {
                return Ok(native.clone());
            }
            let table = classes.table_from(value)?;
            let native = self.ast.add_native(table);
            self.native_memo.insert(key, native.clone());
            self.keep.push(value.clone());
            return Ok(native);
        }
        Ok(Value::Opaque(self.object(value)?))
    }

    /// The index of an object handed through as a [`Value::Opaque`].
    fn object(&mut self, value: &Bound<'py, PyAny>) -> PyResult<u32> {
        let key = value.as_ptr() as usize;
        if let Some(&index) = self.object_memo.get(&key) {
            return Ok(index);
        }
        let index = u32::try_from(self.objects.len())
            .map_err(|_| unsupported("more opaque objects than a u32 counts"))?;
        self.object_memo.insert(key, index);
        self.objects.push(value.clone());
        Ok(index)
    }
}

fn string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.extract::<String>().map_err(|_| unsupported("a string Rust cannot hold"))
}

fn int(value: &Bound<'_, PyAny>) -> PyResult<Int> {
    match value.extract::<i64>() {
        Ok(small) => Ok(Int::Small(small)),
        Err(_) => Ok(Int::Big(value.str()?.to_string())),
    }
}

fn enum_member<T>(value: &Bound<'_, PyAny>, from_name: fn(&str) -> Option<T>) -> PyResult<T> {
    let name = value.get_type().name()?.to_string();
    from_name(&name).ok_or_else(|| unsupported(format!("operator class {name}")))
}

pub fn constant(value: &Bound<'_, PyAny>) -> PyResult<Constant> {
    if value.is_none() {
        return Ok(Constant::None);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Constant::Bool(flag.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Constant::Int(int(value)?));
    }
    if let Ok(number) = value.cast::<PyFloat>() {
        return Ok(Constant::Float(number.value()));
    }
    if let Ok(number) = value.cast::<PyComplex>() {
        return Ok(Constant::Complex { real: number.real(), imag: number.imag() });
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Constant::Str(string(value)?));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Constant::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(items) = value.cast::<PyTuple>() {
        return Ok(Constant::Tuple(
            items.iter().map(|item| constant(&item)).collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(items) = value.cast::<PyFrozenSet>() {
        return Ok(Constant::FrozenSet(
            items.iter().map(|item| constant(&item)).collect::<PyResult<_>>()?,
        ));
    }
    if value.is(value.py().Ellipsis()) {
        return Ok(Constant::Ellipsis);
    }
    Err(unsupported(format!("constant of type {}", value.get_type().name()?)))
}

/// The arena back into Python objects.
pub struct ToPython<'py> {
    py: Python<'py>,
    ast_module: Bound<'py, PyModule>,
    /// What the [`Value::Opaque`] values index (see [`ToPython::with_objects`])
    objects: Vec<Bound<'py, PyAny>>,
    memo: HashMap<NodeId, Bound<'py, PyAny>>,
    /// Per class index: the class object, and which of its fields this Python has
    classes: HashMap<usize, (Bound<'py, PyAny>, Vec<bool>)>,
    /// One instance per operator class, as CPython's parser shares them
    singletons: HashMap<&'static str, Bound<'py, PyAny>>,
    /// The table classes, imported when the first native table is converted
    type_tables: Option<Classes<'py>>,
    /// One Python object per [`Value::Native`]
    natives: HashMap<u32, Bound<'py, PyAny>>,
}

impl<'py> ToPython<'py> {
    pub fn new(py: Python<'py>) -> PyResult<ToPython<'py>> {
        Ok(ToPython {
            py,
            ast_module: py.import("ast")?,
            objects: Vec::new(),
            memo: HashMap::new(),
            classes: HashMap::new(),
            singletons: HashMap::new(),
            type_tables: None,
            natives: HashMap::new(),
        })
    }

    /// Resolve the opaque values of a tree [`FromPython`] converted: its `objects`.
    pub fn with_objects(mut self, objects: Vec<Bound<'py, PyAny>>) -> ToPython<'py> {
        self.objects = objects;
        self
    }

    fn class(&mut self, index: usize) -> PyResult<(Bound<'py, PyAny>, Vec<bool>)> {
        if let Some(entry) = self.classes.get(&index) {
            return Ok(entry.clone());
        }
        let class = &CLASSES[index];
        let object = self
            .ast_module
            .getattr(class.name)
            .map_err(|_| unsupported(format!("this Python has no ast.{}", class.name)))?;
        let fields: Vec<String> = object.getattr("_fields")?.extract()?;
        let present =
            class.fields.iter().map(|field| fields.iter().any(|f| f == field.name)).collect();
        self.classes.insert(index, (object.clone(), present));
        Ok(self.classes[&index].clone())
    }

    fn singleton(&mut self, name: &'static str) -> PyResult<Bound<'py, PyAny>> {
        if let Some(object) = self.singletons.get(name) {
            return Ok(object.clone());
        }
        let object = self.ast_module.getattr(name)?.call0()?;
        self.singletons.insert(name, object.clone());
        Ok(object)
    }

    /// The Python object a node has been converted into so far, if it has been.
    pub fn converted(&self, id: NodeId) -> Option<&Bound<'py, PyAny>> {
        self.memo.get(&id)
    }

    pub fn node(&mut self, ast: &Ast, id: NodeId) -> PyResult<Bound<'py, PyAny>> {
        if let Some(object) = self.memo.get(&id) {
            return Ok(object.clone());
        }
        let node = &ast[id];
        let index = node.kind.class_index();
        let (class, present) = self.class(index)?;
        let object = class.call_method1("__new__", (&class,))?;
        self.memo.insert(id, object.clone());
        let info = &CLASSES[index];
        for (field_index, field) in info.fields.iter().enumerate() {
            let value = self.field(ast, node.kind.field(field_index))?;
            if !present[field_index] {
                // A field this Python does not have may only be at its default
                let empty = value.is_none()
                    || value.cast::<PyList>().map(|list| list.is_empty()).unwrap_or(false);
                if !empty {
                    return Err(unsupported(format!(
                        "{}.{} on this Python",
                        info.name, field.name
                    )));
                }
                continue;
            }
            object.setattr(field.name, value)?;
        }
        if info.located {
            let loc = node.loc;
            for (name, value) in LOCATIONS.iter().zip([
                loc.lineno,
                loc.col_offset,
                loc.end_lineno,
                loc.end_col_offset,
            ]) {
                if let Some(value) = value {
                    object.setattr(*name, value)?;
                }
            }
        }
        for (name, value) in &node.extra {
            let value = self.value(ast, value)?;
            object.setattr(name.as_str(), value)?;
        }
        Ok(object)
    }

    fn opt_node(&mut self, ast: &Ast, id: Option<NodeId>) -> PyResult<Bound<'py, PyAny>> {
        match id {
            Some(id) => self.node(ast, id),
            None => Ok(self.py.None().into_bound(self.py)),
        }
    }

    fn field(&mut self, ast: &Ast, field: FieldRef<'_>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match field {
            FieldRef::Node(id) => self.node(ast, *id)?,
            FieldRef::OptNode(id) => self.opt_node(ast, *id)?,
            FieldRef::Nodes(ids) => {
                let items =
                    ids.iter().map(|id| self.node(ast, *id)).collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, items)?.into_any()
            }
            FieldRef::OptNodes(ids) => {
                let items =
                    ids.iter().map(|id| self.opt_node(ast, *id)).collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, items)?.into_any()
            }
            FieldRef::Str(text) => PyString::new(py, text).into_any(),
            FieldRef::OptStr(text) => match text {
                Some(text) => PyString::new(py, text).into_any(),
                None => py.None().into_bound(py),
            },
            FieldRef::Strs(items) => PyList::new(py, items)?.into_any(),
            FieldRef::Int(value) => value.into_pyobject(py)?.into_any(),
            FieldRef::OptInt(value) => match value {
                Some(value) => value.into_pyobject(py)?.into_any(),
                None => py.None().into_bound(py),
            },
            FieldRef::Constant(value) => self.constant(value)?,
            FieldRef::ExprContext(value) => self.singleton(value.name())?,
            FieldRef::Operator(value) => self.singleton(value.name())?,
            FieldRef::UnaryOperator(value) => self.singleton(value.name())?,
            FieldRef::BoolOperator(value) => self.singleton(value.name())?,
            FieldRef::CmpOperators(ops) => {
                let items =
                    ops.iter().map(|op| self.singleton(op.name())).collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, items)?.into_any()
            }
        })
    }

    fn int(&self, value: &Int) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match value {
            Int::Small(value) => value.into_pyobject(py)?.into_any(),
            Int::Big(digits) => py.get_type::<PyInt>().call1((digits.as_str(),))?,
        })
    }

    fn constant(&self, value: &Constant) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match value {
            Constant::None => py.None().into_bound(py),
            Constant::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
            Constant::Int(value) => self.int(value)?,
            Constant::Float(value) => PyFloat::new(py, *value).into_any(),
            Constant::Complex { real, imag } => {
                PyComplex::from_doubles(py, *real, *imag).into_any()
            }
            Constant::Str(text) => PyString::new(py, text).into_any(),
            Constant::Bytes(bytes) => PyBytes::new(py, bytes).into_any(),
            Constant::Ellipsis => py.Ellipsis().into_bound(py),
            Constant::Tuple(items) => {
                let items =
                    items.iter().map(|item| self.constant(item)).collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, items)?.into_any()
            }
            Constant::FrozenSet(items) => {
                let items =
                    items.iter().map(|item| self.constant(item)).collect::<PyResult<Vec<_>>>()?;
                PyFrozenSet::new(py, items)?.into_any()
            }
        })
    }

    fn value(&mut self, ast: &Ast, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match value {
            Value::None => py.None().into_bound(py),
            Value::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
            Value::Int(value) => self.int(value)?,
            Value::Float(value) => PyFloat::new(py, *value).into_any(),
            Value::Str(text) => PyString::new(py, text).into_any(),
            Value::Tuple(items) => {
                let items =
                    items.iter().map(|item| self.value(ast, item)).collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, items)?.into_any()
            }
            Value::List(items) => {
                let items =
                    items.iter().map(|item| self.value(ast, item)).collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, items)?.into_any()
            }
            Value::Dict(items) => {
                let dict = PyDict::new(py);
                for (key, item) in items {
                    dict.set_item(self.value(ast, key)?, self.value(ast, item)?)?;
                }
                dict.into_any()
            }
            Value::Node(id) => self.node(ast, *id)?,
            Value::Opaque(index) => self
                .objects
                .get(*index as usize)
                .ok_or_else(|| unsupported(format!("opaque value {index} of another conversion")))?
                .clone(),
            Value::Native(index) => self.native(ast, *index, value)?,
        })
    }

    /// A stamped attribute's value, as a node's attribute converts.
    pub fn stamp(&mut self, ast: &Ast, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        self.value(ast, value)
    }

    /// The Python object of a native one: the same object every time it is reached.
    fn native(&mut self, ast: &Ast, index: u32, value: &Value) -> PyResult<Bound<'py, PyAny>> {
        if let Some(object) = self.natives.get(&index) {
            return Ok(object.clone());
        }
        let table = ast
            .native::<PineTypeTable>(value)
            .ok_or_else(|| unsupported(format!("native object {index} of an unknown type")))?;
        if self.type_tables.is_none() {
            self.type_tables = Some(Classes::import(self.py)?);
        }
        let classes = self.type_tables.as_ref().expect("imported above");
        let object = classes.table_to(table)?;
        self.natives.insert(index, object.clone());
        Ok(object)
    }
}
