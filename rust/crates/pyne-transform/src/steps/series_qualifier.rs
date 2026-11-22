//! Port of `transformers/series_qualifier.py`: rewrite every parameter annotation
//! `Marker[Series[T]]` (`Simple`, `Const`) into `Series[Marker[T]]`, so the steps that
//! look for a `Series` annotation head see one.

use pyne_ast::walk::walk_statements;
use pyne_ast::{Ast, Kind, NodeId};

use crate::{Context, Error};

/// The qualifier markers of `types/qualifier.py`.
const MARKERS: &[&str] = &["Simple", "Const"];

/// The name a subscript head spells, `""` for anything else (`_head`).
fn head(ast: &Ast, id: NodeId) -> &str {
    match ast.kind(id) {
        Kind::Name { id: name, .. } => name,
        Kind::Attribute { attr, .. } => attr,
        _ => "",
    }
}

/// Swap the heads of `Marker[Series[T]]` in place (`_normalize`): each subscript
/// keeps its identity and its location.
fn normalize(ast: &mut Ast, annotation: Option<NodeId>) {
    let Some(annotation) = annotation else { return };
    let Kind::Subscript { value: marker, slice: inner, .. } = *ast.kind(annotation) else {
        return;
    };
    if !MARKERS.contains(&head(ast, marker)) {
        return;
    }
    let Kind::Subscript { value: series, .. } = *ast.kind(inner) else { return };
    if !matches!(ast.kind(series), Kind::Name { id, .. } if id == "Series") {
        return;
    }
    if let Kind::Subscript { value, .. } = ast.kind_mut(annotation) {
        *value = series;
    }
    if let Kind::Subscript { value, .. } = ast.kind_mut(inner) {
        *value = marker;
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    for stmt in walk_statements(ast, module) {
        let (Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. }) =
            *ast.kind(stmt)
        else {
            continue;
        };
        let Kind::Arguments { posonlyargs, args: positional, vararg, kwonlyargs, kwarg, .. } =
            ast.kind(args)
        else {
            continue;
        };
        let params: Vec<NodeId> = posonlyargs
            .iter()
            .chain(positional)
            .chain(kwonlyargs)
            .copied()
            .chain(*vararg)
            .chain(*kwarg)
            .collect();
        for param in params {
            let Kind::Arg { annotation, .. } = *ast.kind(param) else { continue };
            normalize(ast, annotation);
        }
    }
    Ok(module)
}
