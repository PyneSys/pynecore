//! Port of `transformers/export_once.py`: a library `main`'s `@export` definitions run
//! under a per-root `Persistent` latch, so they are built on the first bar only.
//!
//! An export whose name `main` binds anywhere else stays out of the latch and is
//! rebuilt per bar, as the Python module documents.

use std::collections::HashSet;

use pyne_ast::walk::{children, docstring_end};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, UnaryOperator};

use crate::{Context, Error};

/// Per-root latch the definitions run under, a Persistent slot of `main`
const EXPORT_LATCH: &str = "__exports·__";

/// Every name the statements bind in `main`'s own scope; nested function and class
/// bodies and lambdas are other scopes and are not entered.
fn bound_names(ast: &Ast, nodes: &[NodeId]) -> HashSet<String> {
    let mut bound = HashSet::new();
    let mut stack = nodes.to_vec();
    while let Some(node) = stack.pop() {
        match ast.kind(node) {
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                bound.insert(name.clone());
                continue;
            }
            Kind::Lambda { .. } => continue,
            Kind::Name { id, ctx: ExprContext::Store } => {
                bound.insert(id.clone());
            }
            _ => {}
        }
        stack.extend(children(ast, node));
    }
    bound
}

/// Whether a statement is a definition carrying `@export` or `@<mod>.export`.
fn is_export(ast: &Ast, stmt: NodeId) -> bool {
    let Kind::FunctionDef { decorator_list, .. } = ast.kind(stmt) else { return false };
    decorator_list.iter().any(|&decorator| {
        let target = match ast.kind(decorator) {
            Kind::Call { func, .. } => *func,
            _ => decorator,
        };
        match ast.kind(target) {
            Kind::Name { id, .. } => id == "export",
            Kind::Attribute { attr, .. } => attr == "export",
            _ => false,
        }
    })
}

fn function_name(ast: &Ast, stmt: NodeId) -> &str {
    let Kind::FunctionDef { name, .. } = ast.kind(stmt) else { unreachable!("not a definition") };
    name
}

fn latch(ast: &mut Ast, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: EXPORT_LATCH.into(), ctx })
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("ExportOnce on a non-module".into()));
    };
    let main = body
        .iter()
        .copied()
        .find(|&stmt| matches!(ast.kind(stmt), Kind::FunctionDef { name, .. } if name == "main"));
    let Some(main) = main else { return Ok(module) };
    let Kind::FunctionDef { body: main_body, .. } = ast.kind(main) else { unreachable!() };
    let main_body = main_body.clone();
    if !main_body.iter().any(|&stmt| is_export(ast, stmt)) {
        return Ok(module);
    }

    // A name main binds anywhere else stays out of the latch
    let others: Vec<NodeId> =
        main_body.iter().copied().filter(|&stmt| !is_export(ast, stmt)).collect();
    let outside = bound_names(ast, &others);

    // Contiguous runs of latchable export definitions, each guarded where it stands
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &stmt) in main_body.iter().enumerate() {
        if is_export(ast, stmt) && !outside.contains(function_name(ast, stmt)) {
            start.get_or_insert(i);
            let name = function_name(ast, stmt);
            if !names.iter().any(|known| known == name) {
                names.push(name.to_string());
            }
        } else if let Some(begin) = start.take() {
            runs.push((begin, i));
        }
    }
    if let Some(begin) = start {
        runs.push((begin, main_body.len()));
    }
    if runs.is_empty() {
        return Ok(module);
    }

    let mut body: Vec<NodeId> = Vec::new();
    let mut cursor = 0;
    for (n, &(begin, end)) in runs.iter().enumerate() {
        body.extend_from_slice(&main_body[cursor..begin]);
        let mut guarded = main_body[begin..end].to_vec();
        if n == runs.len() - 1 {
            // The last run closes the latch
            let target = latch(ast, ExprContext::Store);
            let value = ast.add(Kind::Constant { value: Constant::Bool(true), kind: None });
            guarded.push(ast.add(Kind::Assign {
                targets: vec![target],
                value,
                type_comment: None,
            }));
        }
        let operand = latch(ast, ExprContext::Load);
        let test = ast.add(Kind::UnaryOp { op: UnaryOperator::Not, operand });
        body.push(ast.add(Kind::If { test, body: guarded, orelse: Vec::new() }));
        cursor = end;
    }
    body.extend_from_slice(&main_body[cursor..]);

    let global = ast.add(Kind::Global { names });
    let target = latch(ast, ExprContext::Store);
    let persistent = ast.add(Kind::Name { id: "Persistent".into(), ctx: ExprContext::Load });
    let slice = ast.add(Kind::Name { id: "bool".into(), ctx: ExprContext::Load });
    let annotation = ast.add(Kind::Subscript { value: persistent, slice, ctx: ExprContext::Load });
    let value = ast.add(Kind::Constant { value: Constant::Bool(false), kind: None });
    let declaration =
        ast.add(Kind::AnnAssign { target, annotation, value: Some(value), simple: 1 });

    // A `global` must precede every use of the names it declares, so the prologue
    // goes above any statement but the docstring
    let insert_at = docstring_end(ast, &body);
    body.splice(insert_at..insert_at, [global, declaration]);
    let Kind::FunctionDef { body: main_body, .. } = ast.kind_mut(main) else { unreachable!() };
    *main_body = body;
    Ok(module)
}
