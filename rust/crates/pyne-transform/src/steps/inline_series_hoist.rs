//! Port of `transformers/inline_series_hoist.py`: hoist `inline_series` calls out of
//! lazily evaluated expression positions.
//!
//! Every `inline_series(...)` call in a ternary branch or in an `and` / `or`
//! operand after the first is bound to a `__hist_<n>__` temporary assigned just
//! before its statement, in the same statement list. `while` tests, lambdas and
//! comprehensions are left alone.

use pyne_ast::locations::fix_locations;
use pyne_ast::walk::{Transformer, Visited, generic_visit};
use pyne_ast::{Ast, Category, ExprContext, FieldKind, FieldMut, FieldRef, Kind, NodeId};

use crate::{Context, Error};

/// Fields holding nested statement lists (handlers are special-cased).
const STATEMENT_LIST_FIELDS: [&str; 3] = ["body", "orelse", "finalbody"];

fn is_category(ast: &Ast, id: NodeId, category: Category) -> bool {
    ast.kind(id).class().category == category
}

/// Whether a node is a direct `inline_series(...)` call.
fn is_inline_series(ast: &Ast, id: NodeId) -> bool {
    match ast.kind(id) {
        Kind::Call { func, .. } => {
            matches!(ast.kind(*func), Kind::Name { id, .. } if id == "inline_series")
        }
        _ => false,
    }
}

/// Rewrites one statement's expression tree: every `inline_series` call in a lazy
/// position becomes a temporary assignment in `hoisted`, inner calls first.
struct ExprHoister<'a> {
    counter: &'a mut usize,
    hoisted: Vec<NodeId>,
    /// Whether the current subtree only evaluates when short-circuiting or
    /// ternary branch selection reaches it
    lazy: bool,
}

impl ExprHoister<'_> {
    /// Visit a subtree with an explicit laziness flag.
    fn visit_as(&mut self, ast: &mut Ast, id: NodeId, lazy: bool) -> NodeId {
        let prev = std::mem::replace(&mut self.lazy, lazy);
        let Visited::Node(new) = self.visit(ast, id) else { unreachable!() };
        self.lazy = prev;
        new
    }

    fn next_name(&mut self) -> String {
        let name = format!("__hist_{}__", self.counter);
        *self.counter += 1;
        name
    }
}

impl Transformer for ExprHoister<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::BoolOp { values, .. } => {
                let values = values.clone();
                let lazy = self.lazy;
                let new_values: Vec<NodeId> = values
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| self.visit_as(ast, value, if i == 0 { lazy } else { true }))
                    .collect();
                let Kind::BoolOp { values, .. } = ast.kind_mut(id) else { unreachable!() };
                *values = new_values;
            }
            &Kind::IfExp { test, body, orelse } => {
                let lazy = self.lazy;
                let test = self.visit_as(ast, test, lazy);
                let Kind::IfExp { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = test;
                let body = self.visit_as(ast, body, true);
                let Kind::IfExp { body: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = body;
                let orelse = self.visit_as(ast, orelse, true);
                let Kind::IfExp { orelse: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = orelse;
            }
            // Deferred / repeated evaluation contexts: hoisting from inside would
            // change how often the call runs
            Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. } => {}
            Kind::Call { args, keywords, .. } if self.lazy && is_inline_series(ast, id) => {
                let (args, keywords) = (args.clone(), keywords.clone());
                // The hoisted call runs at statement level: its arguments leave the
                // lazy context, so nested positions are judged from an eager root
                let new_args: Vec<NodeId> =
                    args.into_iter().map(|arg| self.visit_as(ast, arg, false)).collect();
                let Kind::Call { args, .. } = ast.kind_mut(id) else { unreachable!() };
                *args = new_args;
                for keyword in keywords {
                    let &Kind::Keyword { value, .. } = ast.kind(keyword) else { unreachable!() };
                    let value = self.visit_as(ast, value, false);
                    let Kind::Keyword { value: slot, .. } = ast.kind_mut(keyword) else {
                        unreachable!()
                    };
                    *slot = value;
                }
                let name = self.next_name();
                let target = ast.add(Kind::Name { id: name.clone(), ctx: ExprContext::Store });
                let assign =
                    ast.add(Kind::Assign { targets: vec![target], value: id, type_comment: None });
                self.hoisted.push(assign);
                return Visited::Node(ast.add(Kind::Name { id: name, ctx: ExprContext::Load }));
            }
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

/// Walks the statement lists, hoisting from every statement's own expressions.
struct InlineSeriesHoist {
    counter: usize,
}

impl InlineSeriesHoist {
    /// Process one statement list; returns the list with the hoists inserted.
    fn process_body(&mut self, ast: &mut Ast, body: Vec<NodeId>) -> Vec<NodeId> {
        let mut new_body = Vec::with_capacity(body.len());
        for stmt in body {
            // Nested statement lists first: each is its own hoist target, which
            // keeps block-level (gap) semantics intact
            let class = ast.kind(stmt).class();
            for name in STATEMENT_LIST_FIELDS {
                let Some(index) = class.fields.iter().position(|field| field.name == name) else {
                    continue;
                };
                let FieldRef::Nodes(items) = ast.kind(stmt).field(index) else {
                    continue;
                };
                if !items.first().is_some_and(|&first| is_category(ast, first, Category::Stmt)) {
                    continue;
                }
                let processed = self.process_body(ast, items.clone());
                let FieldMut::Nodes(slot) = ast.kind_mut(stmt).field_mut(index) else {
                    unreachable!()
                };
                *slot = processed;
            }
            let handlers = match ast.kind(stmt) {
                Kind::Try { handlers, .. } | Kind::TryStar { handlers, .. } => handlers.clone(),
                _ => Vec::new(),
            };
            for handler in handlers {
                let Kind::ExceptHandler { body, .. } = ast.kind(handler) else { unreachable!() };
                let processed = self.process_body(ast, body.clone());
                let Kind::ExceptHandler { body, .. } = ast.kind_mut(handler) else {
                    unreachable!()
                };
                *body = processed;
            }
            new_body.extend(self.hoist_from(ast, stmt));
            new_body.push(stmt);
        }
        new_body
    }

    /// Rewrite the statement's own expression fields; returns the temporary
    /// assignments to place before it.
    fn hoist_from(&mut self, ast: &mut Ast, stmt: NodeId) -> Vec<NodeId> {
        // Statements whose own expressions evaluate at definition time
        if matches!(
            ast.kind(stmt),
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::ClassDef { .. }
        ) {
            return Vec::new();
        }
        let is_while = matches!(ast.kind(stmt), Kind::While { .. });
        let mut hoister =
            ExprHoister { counter: &mut self.counter, hoisted: Vec::new(), lazy: false };
        let class = ast.kind(stmt).class();
        for (index, field) in class.fields.iter().enumerate() {
            // A while test is re-evaluated per iteration: it must stay in place
            if is_while && field.name == "test" {
                continue;
            }
            match field.kind {
                FieldKind::Node | FieldKind::OptNode => {
                    let value = match ast.kind(stmt).field(index) {
                        FieldRef::Node(&value) | FieldRef::OptNode(&Some(value)) => value,
                        _ => continue,
                    };
                    if !is_category(ast, value, Category::Expr) {
                        continue;
                    }
                    let value = hoister.visit_as(ast, value, false);
                    match ast.kind_mut(stmt).field_mut(index) {
                        FieldMut::Node(slot) => *slot = value,
                        FieldMut::OptNode(slot) => *slot = Some(value),
                        _ => unreachable!(),
                    }
                }
                FieldKind::Nodes => {
                    let FieldRef::Nodes(items) = ast.kind(stmt).field(index) else {
                        unreachable!()
                    };
                    if !items.first().is_some_and(|&first| is_category(ast, first, Category::Expr))
                    {
                        continue;
                    }
                    let items = items.clone();
                    let new_items: Vec<NodeId> =
                        items.into_iter().map(|item| hoister.visit_as(ast, item, false)).collect();
                    let FieldMut::Nodes(slot) = ast.kind_mut(stmt).field_mut(index) else {
                        unreachable!()
                    };
                    *slot = new_items;
                }
                _ => {}
            }
        }
        hoister.hoisted
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("InlineSeriesHoist on a non-module".into()));
    };
    let mut pass = InlineSeriesHoist { counter: 0 };
    let body = pass.process_body(ast, body.clone());
    let Kind::Module { body: slot, .. } = ast.kind_mut(module) else { unreachable!() };
    *slot = body;
    if pass.counter > 0 {
        fix_locations(ast, module, 1, 0)?;
    }
    Ok(module)
}
