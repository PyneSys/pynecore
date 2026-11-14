//! Port of `transformers/inline_series_hoist.py`: hoist `inline_series` calls out of
//! lazily evaluated expression positions.
//!
//! Every `inline_series(...)` call in a ternary branch or in an `and` / `or`
//! operand after the first is bound to a `__hist_<n>__` temporary assigned just
//! before its statement, in the same statement list. `while` tests, lambdas and
//! comprehensions are left alone. The test of an `elif` is a lazy position of the
//! whole `if` statement: its calls go before the head of the chain. A call that
//! may read a name a `:=` of the same statement (for an `elif`: of the chain's
//! tests) binds stays in place; with such a `:=` present, a nested call counts as a
//! reader too (a local helper can read the name through its closure).

use pyne_ast::locations::fix_locations;
use std::collections::HashSet;

use pyne_ast::walk::{Transformer, Visited, generic_visit, walk};
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

/// The names the `:=` expressions of a statement bind.
fn walrus_targets(ast: &Ast, exprs: &[NodeId]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &expr in exprs {
        for node in walk(ast, expr) {
            if let &Kind::NamedExpr { target, .. } = ast.kind(node)
                && let Kind::Name { id, .. } = ast.kind(target)
            {
                names.insert(id.clone());
            }
        }
    }
    names
}

/// Whether a call may read any of `names`: a `Name` of the subtree is one of them,
/// or the subtree contains a nested call (a local helper can read them through its
/// closure).
fn may_read(ast: &Ast, id: NodeId, names: &HashSet<String>) -> bool {
    !names.is_empty()
        && walk(ast, id).into_iter().any(|node| match ast.kind(node) {
            Kind::Name { id, .. } => names.contains(id),
            Kind::Call { .. } => node != id,
            _ => false,
        })
}

/// The `elif` that continues an `if` statement: the single `If` of its `orelse`,
/// starting in the column of the `if` on a later line (an `if` nested in an
/// `else` block is indented deeper).
fn elif_of(ast: &Ast, id: NodeId) -> Option<NodeId> {
    let Kind::If { orelse, .. } = ast.kind(id) else { return None };
    let &[child] = orelse.as_slice() else { return None };
    if !matches!(ast.kind(child), Kind::If { .. }) {
        return None;
    }
    let (loc, child_loc) = (ast[id].loc, ast[child].loc);
    let (Some(line), Some(child_line)) = (loc.lineno, child_loc.lineno) else { return None };
    (child_line > line && child_loc.col_offset == loc.col_offset).then_some(child)
}

/// Rewrites one statement's expression tree: every `inline_series` call in a lazy
/// position becomes a temporary assignment in `hoisted`, inner calls first. A call
/// that may read a name of `bound` (assigned by a `:=` of the statement) stays.
struct ExprHoister<'a> {
    counter: &'a mut usize,
    bound: HashSet<String>,
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
            Kind::Call { args, keywords, .. }
                if self.lazy
                    && is_inline_series(ast, id)
                    && !may_read(ast, id, &self.bound) =>
            {
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
            // The elif tests belong to this statement: hoisted before its nested
            // lists are processed, so only the calls left in place remain for them
            let chain = self.hoist_elif_tests(ast, stmt);
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
            new_body.extend(chain);
            new_body.push(stmt);
        }
        new_body
    }

    /// Rewrite the tests of the `elif` chain continuing `stmt`; returns the
    /// temporary assignments to place before `stmt`.
    fn hoist_elif_tests(&mut self, ast: &mut Ast, stmt: NodeId) -> Vec<NodeId> {
        let mut chain = Vec::new();
        let mut branch = elif_of(ast, stmt);
        while let Some(id) = branch {
            chain.push(id);
            branch = elif_of(ast, id);
        }
        if chain.is_empty() {
            return Vec::new();
        }
        let tests: Vec<NodeId> = std::iter::once(stmt)
            .chain(chain.iter().copied())
            .map(|id| {
                let &Kind::If { test, .. } = ast.kind(id) else { unreachable!() };
                test
            })
            .collect();
        let bound = walrus_targets(ast, &tests);
        let mut hoister =
            ExprHoister { counter: &mut self.counter, bound, hoisted: Vec::new(), lazy: false };
        for id in chain {
            let &Kind::If { test, .. } = ast.kind(id) else { unreachable!() };
            let test = hoister.visit_as(ast, test, true);
            let Kind::If { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
            *slot = test;
        }
        hoister.hoisted
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
        let class = ast.kind(stmt).class();
        let mut exprs = Vec::new();
        for (index, field) in class.fields.iter().enumerate() {
            match ast.kind(stmt).field(index) {
                FieldRef::Node(&value) | FieldRef::OptNode(&Some(value))
                    if matches!(field.kind, FieldKind::Node | FieldKind::OptNode)
                        && is_category(ast, value, Category::Expr) =>
                {
                    exprs.push(value);
                }
                FieldRef::Nodes(items)
                    if items.first().is_some_and(|&first| is_category(ast, first, Category::Expr)) =>
                {
                    exprs.extend(items.iter().copied());
                }
                _ => {}
            }
        }
        let bound = walrus_targets(ast, &exprs);
        let mut hoister =
            ExprHoister { counter: &mut self.counter, bound, hoisted: Vec::new(), lazy: false };
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
