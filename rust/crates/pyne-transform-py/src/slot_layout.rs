//! The pipeline's `ModuleLayout` (`transformers/slot_layout.py`) <-> [`ModuleLayout`].
//!
//! The layout is converted together with the tree, by the same [`FromPython`] /
//! [`ToPython`]: a slot's `init` or `max_bars_back` expression that is also a node
//! of the tree is one node of the arena, and comes back as one object. The
//! definition caches (`_segments`, `_nests`), keyed by `id()` of a definition, are
//! keyed by the definition's node on the native side, so they cross for every
//! definition the conversion reaches; `_tree` crosses as the module being converted.
//!
//! Python to Rust is strict: every object must be of its exact class, every field of
//! the exact type it holds, or the conversion answers `Unsupported` and the Python
//! step runs instead. Rust to Python builds a new `ModuleLayout` of the very classes
//! the steps fill, through their constructors; `native.run` copies its state field
//! for field into the pipeline's own layout object.

use pyne_ast::{Ast, NodeId};
use pyne_transform::slot_layout::{ModuleLayout, ScopeLayout, Slot, SlotKind};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString};

use crate::Unsupported;
use crate::convert::{FromPython, ToPython};

fn mismatch(what: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let name = value.get_type().name().map(|name| name.to_string()).unwrap_or_default();
    Unsupported::new_err(format!("slot layout: {what} is a {name}"))
}

fn string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(mismatch("a str", value));
    }
    value.extract().map_err(|_| Unsupported::new_err("slot layout: a str Rust cannot hold"))
}

fn optional_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() { Ok(None) } else { string(value).map(Some) }
}

fn index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(mismatch("an index", value));
    }
    value.extract().map_err(|_| Unsupported::new_err(format!("slot layout: index {value}")))
}

fn boolean(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.cast_exact::<PyBool>().map(|flag| flag.is_true()).map_err(|_| mismatch("a bool", value))
}

/// The classes a layout is made of.
pub struct LayoutClasses<'py> {
    module_layout: Bound<'py, PyAny>,
    scope_layout: Bound<'py, PyAny>,
    slot: Bound<'py, PyAny>,
}

impl<'py> LayoutClasses<'py> {
    pub fn import(py: Python<'py>) -> PyResult<LayoutClasses<'py>> {
        let module = py.import("pynecore.transformers.slot_layout")?;
        Ok(LayoutClasses {
            module_layout: module.getattr("ModuleLayout")?,
            scope_layout: module.getattr("ScopeLayout")?,
            slot: module.getattr("_Slot")?,
        })
    }

    // --- Python -> Rust -------------------------------------------------------

    /// The layout of a run, converted after its tree (`tree`, converted to `root`).
    pub fn layout_from(
        &self,
        from: &mut FromPython<'py>,
        object: &Bound<'py, PyAny>,
        tree: &Bound<'py, PyAny>,
        root: NodeId,
    ) -> PyResult<ModuleLayout> {
        if !object.get_type().is(&self.module_layout) {
            return Err(mismatch("the layout", object));
        }
        let scopes_object = object.getattr("scopes")?;
        let scopes_dict =
            scopes_object.cast_exact::<PyDict>().map_err(|_| mismatch("scopes", &scopes_object))?;
        let mut layout = ModuleLayout::new(boolean(&object.getattr("compacted_series")?)?);
        for (key, scope) in scopes_dict.iter() {
            let scope = self.scope_from(from, &scope)?;
            if layout.scopes.insert(string(&key)?, scope).is_some() {
                return Err(Unsupported::new_err(format!("slot layout: two scopes {key}")));
            }
        }
        let cached = object.getattr("_tree")?;
        layout.tree = if cached.is_none() {
            None
        } else if cached.is(tree) {
            Some(root)
        } else {
            return Err(Unsupported::new_err("slot layout: scopes scanned for another module"));
        };
        // A cache entry of a definition the conversion did not reach can never be
        // asked for: no node of the arena is that definition
        let segments = object.getattr("_segments")?;
        for (key, segment) in
            segments.cast_exact::<PyDict>().map_err(|_| mismatch("_segments", &segments))?.iter()
        {
            if let Some(id) = from.converted(index(&key)?) {
                layout.segments.insert(id, string(&segment)?);
            }
        }
        let nests = object.getattr("_nests")?;
        for (key, nested) in
            nests.cast_exact::<PyDict>().map_err(|_| mismatch("_nests", &nests))?.iter()
        {
            if let Some(id) = from.converted(index(&key)?) {
                layout.nests.insert(id, boolean(&nested)?);
            }
        }
        Ok(layout)
    }

    fn scope_from(
        &self,
        from: &mut FromPython<'py>,
        object: &Bound<'py, PyAny>,
    ) -> PyResult<ScopeLayout> {
        if !object.get_type().is(&self.scope_layout) {
            return Err(mismatch("a scope", object));
        }
        let slots_object = object.getattr("slots")?;
        let slots_list =
            slots_object.cast_exact::<PyList>().map_err(|_| mismatch("slots", &slots_object))?;
        let mut slots = Vec::with_capacity(slots_list.len());
        for slot in slots_list.iter() {
            slots.push(self.slot_from(from, &slot)?);
        }
        let pin_slot = object.getattr("pin_slot")?;
        Ok(ScopeLayout {
            scope: string(&object.getattr("scope")?)?,
            slots,
            state_param: string(&object.getattr("state_param")?)?,
            pin_slot: if pin_slot.is_none() { None } else { Some(index(&pin_slot)?) },
        })
    }

    fn slot_from(&self, from: &mut FromPython<'py>, object: &Bound<'py, PyAny>) -> PyResult<Slot> {
        if !object.get_type().is(&self.slot) {
            return Err(mismatch("a slot", object));
        }
        let get = |name: &str| object.getattr(name);
        let kind = string(&get("kind")?)?;
        let max_bars_back = get("max_bars_back")?;
        Ok(Slot {
            index: index(&get("index")?)?,
            kind: SlotKind::from_name(&kind)
                .ok_or_else(|| Unsupported::new_err(format!("slot layout: slot kind {kind}")))?,
            name: string(&get("name")?)?,
            init: self.node(from, &get("init")?)?,
            max_bars_back: if max_bars_back.is_none() {
                None
            } else {
                Some(self.node(from, &max_bars_back)?)
            },
            series_elem: optional_string(&get("series_elem")?)?,
            var_elem: optional_string(&get("var_elem")?)?,
            call_id: optional_string(&get("call_id")?)?,
            in_loop: boolean(&get("in_loop")?)?,
            varip: boolean(&get("varip")?)?,
        })
    }

    /// A slot's template expression: a node, which the arena may already hold.
    fn node(&self, from: &mut FromPython<'py>, object: &Bound<'py, PyAny>) -> PyResult<NodeId> {
        let ast = object.py().import("ast")?;
        if !object.is_instance(&ast.getattr("AST")?)? {
            return Err(mismatch("a slot template", object));
        }
        from.node(object)
    }

    // --- Rust -> Python -------------------------------------------------------

    /// A new `ModuleLayout` holding a native layout, converted after its tree.
    pub fn layout_to(
        &self,
        to: &mut ToPython<'py>,
        ast: &Ast,
        layout: &ModuleLayout,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.module_layout.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("compacted_series", layout.compacted_series)?;
        let object = self.module_layout.call((), Some(&kwargs))?;
        let scopes = PyDict::new(py);
        for (key, scope) in &layout.scopes {
            scopes.set_item(key, self.scope_to(to, ast, scope)?)?;
        }
        object.setattr("scopes", scopes)?;
        let segments = PyDict::new(py);
        for (&id, segment) in &layout.segments {
            if let Some(node) = to.converted(id) {
                segments.set_item(node.as_ptr() as usize, segment)?;
            }
        }
        object.setattr("_segments", segments)?;
        let nests = PyDict::new(py);
        for (&id, &nested) in &layout.nests {
            if let Some(node) = to.converted(id) {
                nests.set_item(node.as_ptr() as usize, nested)?;
            }
        }
        object.setattr("_nests", nests)?;
        let tree = match layout.tree {
            Some(id) => to.node(ast, id)?,
            None => py.None().into_bound(py),
        };
        object.setattr("_tree", tree)?;
        Ok(object)
    }

    fn scope_to(
        &self,
        to: &mut ToPython<'py>,
        ast: &Ast,
        scope: &ScopeLayout,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.scope_layout.py();
        let slots = scope
            .slots
            .iter()
            .map(|slot| self.slot_to(to, ast, slot))
            .collect::<PyResult<Vec<_>>>()?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("scope", &scope.scope)?;
        kwargs.set_item("slots", PyList::new(py, slots)?)?;
        kwargs.set_item("state_param", &scope.state_param)?;
        kwargs.set_item("pin_slot", scope.pin_slot)?;
        self.scope_layout.call((), Some(&kwargs))
    }

    fn slot_to(
        &self,
        to: &mut ToPython<'py>,
        ast: &Ast,
        slot: &Slot,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.slot.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", slot.index)?;
        kwargs.set_item("kind", slot.kind.name())?;
        kwargs.set_item("name", &slot.name)?;
        kwargs.set_item("init", to.node(ast, slot.init)?)?;
        let max_bars_back = match slot.max_bars_back {
            Some(id) => to.node(ast, id)?,
            None => py.None().into_bound(py),
        };
        kwargs.set_item("max_bars_back", max_bars_back)?;
        kwargs.set_item("series_elem", &slot.series_elem)?;
        kwargs.set_item("var_elem", &slot.var_elem)?;
        kwargs.set_item("call_id", &slot.call_id)?;
        kwargs.set_item("in_loop", slot.in_loop)?;
        kwargs.set_item("varip", slot.varip)?;
        self.slot.call((), Some(&kwargs))
    }
}
