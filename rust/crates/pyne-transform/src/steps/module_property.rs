//! Port of `transformers/module_property.py`: rewrite `lib.xxx` references from the
//! `module_properties.json` registry.
//!
//! A bare read of a `property` entry becomes a call (`lib.ta.tr` -> `lib.ta.tr()`),
//! a function-and-namespace module is routed to its self-named function
//! (`lib.plot(x)` -> `lib.plot.plot(x)`), and an unknown name on a known lib module
//! rejects the module. The registry is the host's knowledge ([`crate::Host`]),
//! asked where the Python step reads it.

use std::collections::HashSet;

use pyne_ast::walk::{Transformer, Visited, generic_visit, walk};
use pyne_ast::{Ast, ExprContext, Kind, NodeId, Value};

use crate::{Context, Error, Host};

struct ModuleProperty<'h> {
    host: &'h dyn Host,
    /// The callee of every call met so far, by identity
    callees: HashSet<NodeId>,
    /// Every node of every type annotation met so far, by identity
    annotation_nodes: HashSet<NodeId>,
    /// The first error: the step stops there
    error: Option<Error>,
}

impl ModuleProperty<'_> {
    fn note_annotation(&mut self, ast: &Ast, annotation: Option<NodeId>) {
        if let Some(annotation) = annotation {
            self.annotation_nodes.extend(walk(ast, annotation));
        }
    }

    fn visit_attribute(&mut self, ast: &mut Ast, id: NodeId) -> Result<NodeId, Error> {
        // Intermediate modules of the chain are not the topmost attribute: only
        // what hangs below the chain is visited, and they stay as they are
        let mut inner = id;
        loop {
            let Kind::Attribute { value, .. } = ast.kind(inner) else { unreachable!() };
            if !matches!(ast.kind(*value), Kind::Attribute { .. }) {
                break;
            }
            inner = *value;
        }
        let Kind::Attribute { value, .. } = ast.kind(inner) else { unreachable!() };
        let new_value = match self.visit(ast, *value) {
            Visited::Node(new) => new,
            answer => panic!("{answer:?} for an attribute value"),
        };
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        let Kind::Attribute { value, .. } = ast.kind_mut(inner) else { unreachable!() };
        *value = new_value;

        if ast.extra(id, "_processed").is_some() {
            return Ok(id);
        }
        let Some((module_path, name)) = lib_path(ast, id) else { return Ok(id) };

        // Skip if inside type annotations
        if self.annotation_nodes.contains(&id) {
            return Ok(id);
        }

        let full_path = format!("{module_path}.{name}");

        // Call site: explicit calls stay as they are, except when the callee is a
        // function-and-namespace module: ``lib.plot(...)`` routes to ``lib.plot.plot(...)``
        if self.callees.contains(&id) {
            if self.has_member(&full_path, &name)? {
                let result =
                    ast.add(Kind::Attribute { value: id, attr: name, ctx: ExprContext::Load });
                ast.set_extra(result, "_processed", Value::Bool(true));
                return Ok(result);
            }
            return Ok(id);
        }

        if !self.host.has_module_properties(&module_path)? {
            // Unknown module path: a user workdir library or a class-attribute chain
            return Ok(id);
        }

        let result = if let Some(kind) = self.host.module_property_type(&module_path, &name)? {
            if kind == "property" {
                if full_path == "lib.na" {
                    // Bare ``na`` loads the interned typeless NA instead of calling ``lib.na()``
                    let lib = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
                    let result = ast.add(Kind::Attribute {
                        value: lib,
                        attr: "_na_none".into(),
                        ctx: ExprContext::Load,
                    });
                    ast.set_extra(result, "_processed", Value::Bool(true));
                    return Ok(result);
                }
                let func = if self.has_member(&full_path, &name)? {
                    // Promoted self-named property of a function-and-namespace module
                    let value = copy_node(ast, id);
                    ast.add(Kind::Attribute { value, attr: name, ctx: ExprContext::Load })
                } else {
                    copy_node(ast, id)
                };
                ast.add(Kind::Call { func, args: Vec::new(), keywords: Vec::new() })
            } else {
                id
            }
        } else if self.host.has_module_properties(&full_path)? || name.starts_with('_') {
            // A submodule reference, or an internal name: plain attribute access
            id
        } else {
            let line = ast[id].loc.lineno.map_or_else(|| "None".to_string(), |l| l.to_string());
            return Err(Error::Rejected(format!(
                "unknown attribute '{name}' on module '{module_path}' (line {line}); if this \
                 is a new pynecore.lib name, regenerate module_properties.json with \
                 scripts/module_property_collector.py"
            )));
        };
        ast.set_extra(result, "_processed", Value::Bool(true));
        Ok(result)
    }

    /// Whether a module has a registry entry naming a member.
    fn has_member(&self, module: &str, name: &str) -> Result<bool, Error> {
        Ok(self.host.has_module_properties(module)? && self.host.module_property(module, name)?)
    }
}

/// `(module path, final attribute)` of an attribute chain starting at the name `lib`
/// (`lib.display.data_window` -> `("lib.display", "data_window")`).
fn lib_path(ast: &Ast, id: NodeId) -> Option<(String, String)> {
    let mut attrs = Vec::new();
    let mut current = id;
    while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
        attrs.push(attr.as_str());
        current = *value;
    }
    if !matches!(ast.kind(current), Kind::Name { id, .. } if id == "lib") {
        return None;
    }
    attrs.push("lib");
    attrs.reverse();
    let name = attrs.pop()?.to_string();
    Some((attrs.join("."), name))
}

/// A shallow copy of an attribute chain, without locations.
fn copy_node(ast: &mut Ast, id: NodeId) -> NodeId {
    match ast.kind(id).clone() {
        Kind::Name { id, ctx } => ast.add(Kind::Name { id, ctx }),
        Kind::Attribute { value, attr, ctx } => {
            let value = copy_node(ast, value);
            ast.add(Kind::Attribute { value, attr, ctx })
        }
        _ => id,
    }
}

impl Transformer for ModuleProperty<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        match ast.kind(id) {
            Kind::Call { func, .. } => {
                self.callees.insert(*func);
            }
            Kind::AnnAssign { annotation, .. } => self.note_annotation(ast, Some(*annotation)),
            Kind::Arg { annotation, .. } => self.note_annotation(ast, *annotation),
            Kind::FunctionDef { returns, .. } | Kind::AsyncFunctionDef { returns, .. } => {
                self.note_annotation(ast, *returns);
            }
            Kind::Attribute { .. } => {
                return match self.visit_attribute(ast, id) {
                    Ok(new) => Visited::Node(new),
                    Err(err) => {
                        self.error = Some(err);
                        Visited::Node(id)
                    }
                };
            }
            _ => {}
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    let mut step = ModuleProperty {
        host: ctx.host,
        callees: HashSet::new(),
        annotation_nodes: HashSet::new(),
        error: None,
    };
    generic_visit(&mut step, ast, module);
    match step.error {
        Some(err) => Err(err),
        None => Ok(module),
    }
}
