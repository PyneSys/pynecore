//! Port of `transformers/persistent_series.py`: split every `PersistentSeries` /
//! `IBPersistentSeries` declaration into its persistent and `Series` halves, and drop
//! the split types from the pynecore imports.

use pyne_ast::walk::{
    Transformer, Visited, copy_location, fix_missing_locations, generic_visit, walk_statements,
};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// A series-carrying persistent annotation mapped to its non-series half.
fn persistent_type(name: &str) -> Option<&'static str> {
    match name {
        "PersistentSeries" => Some("Persistent"),
        "IBPersistentSeries" => Some("IBPersistent"),
        _ => None,
    }
}

/// The persistent half of a series-carrying persistent declaration.
fn persistent_half(ast: &Ast, node: NodeId) -> Option<&'static str> {
    let Kind::AnnAssign { target, annotation, .. } = ast.kind(node) else { return None };
    if !matches!(ast.kind(*target), Kind::Name { .. }) {
        return None;
    }
    let annotation = match ast.kind(*annotation) {
        Kind::Subscript { value, .. } => *value,
        _ => *annotation,
    };
    let Kind::Name { id, .. } = ast.kind(annotation) else { return None };
    persistent_type(id)
}

fn is_pynecore(module: &Option<String>) -> bool {
    module.as_deref().is_some_and(|module| module.starts_with("pynecore"))
}

struct PersistentSeries;

impl PersistentSeries {
    /// Remove the split types from a pynecore import; an import left empty goes.
    fn visit_import_from(ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::ImportFrom { module, names, .. } = ast.kind(id) else { unreachable!() };
        if !is_pynecore(module) {
            return Visited::Node(id);
        }
        let kept: Vec<NodeId> = names
            .iter()
            .copied()
            .filter(|&alias| {
                let Kind::Alias { name, .. } = ast.kind(alias) else { unreachable!() };
                persistent_type(name).is_none()
            })
            .collect();
        if kept.is_empty() {
            return Visited::Remove;
        }
        let Kind::ImportFrom { names, .. } = ast.kind_mut(id) else { unreachable!() };
        *names = kept;
        Visited::Node(id)
    }

    /// Split a series-carrying persistent declaration into its two halves.
    fn visit_ann_assign(ast: &mut Ast, id: NodeId) -> Visited {
        let Some(persistent) = persistent_half(ast, id) else { return Visited::Node(id) };
        let Kind::AnnAssign { target, annotation, value, .. } = ast.kind(id) else {
            unreachable!()
        };
        let series_type = match ast.kind(*annotation) {
            Kind::Subscript { slice, .. } => Some(*slice),
            _ => None,
        };
        let Kind::Name { id: var_name, .. } = ast.kind(*target) else { unreachable!() };
        let var_name = var_name.clone();
        let value = *value;

        let persistent_decl = declaration(ast, &var_name, persistent, series_type, value);
        // The split stands where the declaration stood: a diagnostic on either half
        // has to point at the line the user wrote
        copy_location(ast, persistent_decl, id);
        fix_missing_locations(ast, persistent_decl);

        let series_value = ast.add(Kind::Name { id: var_name.clone(), ctx: ExprContext::Load });
        let series_decl = declaration(ast, &var_name, "Series", series_type, Some(series_value));
        copy_location(ast, series_decl, id);
        fix_missing_locations(ast, series_decl);

        Visited::Many(vec![persistent_decl, series_decl])
    }
}

/// `var_name: <type>[<series_type>] = value`, or `var_name: <type> = value` without
/// an element type.
fn declaration(
    ast: &mut Ast,
    var_name: &str,
    type_name: &str,
    series_type: Option<NodeId>,
    value: Option<NodeId>,
) -> NodeId {
    let target = ast.add(Kind::Name { id: var_name.to_string(), ctx: ExprContext::Store });
    let type_node = ast.add(Kind::Name { id: type_name.to_string(), ctx: ExprContext::Load });
    let annotation = match series_type {
        Some(slice) => ast.add(Kind::Subscript { value: type_node, slice, ctx: ExprContext::Load }),
        None => type_node,
    };
    ast.add(Kind::AnnAssign { target, annotation, value, simple: 1 })
}

impl Transformer for PersistentSeries {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::ImportFrom { .. } => Self::visit_import_from(ast, id),
            Kind::AnnAssign { .. } => Self::visit_ann_assign(ast, id),
            _ => {
                generic_visit(self, ast, id);
                Visited::Node(id)
            }
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("PersistentSeries on a non-module".into()));
    }
    // Both the split declarations and the imports of the split types are
    // statements: a module with neither is left as it is
    let candidate = walk_statements(ast, module).into_iter().any(|stmt| match ast.kind(stmt) {
        Kind::AnnAssign { .. } => persistent_half(ast, stmt).is_some(),
        Kind::ImportFrom { module, names, .. } => {
            is_pynecore(module)
                && names.iter().any(|&alias| {
                    let Kind::Alias { name, .. } = ast.kind(alias) else { unreachable!() };
                    persistent_type(name).is_some()
                })
        }
        _ => false,
    });
    if !candidate {
        return Ok(module);
    }
    generic_visit(&mut PersistentSeries, ast, module);
    Ok(module)
}
