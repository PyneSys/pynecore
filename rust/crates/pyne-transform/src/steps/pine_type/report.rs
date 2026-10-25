//! Where a module's types run out, reported once each (`pine_type_report.py`).

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{children, walk};
use pyne_ast::{Constant, ExprContext, Kind, NodeId};

use super::{Engine, R};
use crate::pine_type_rules::{TY_ATTR, UNKNOWN, dotted, get_ty, render_ty};
use crate::pine_type_table::{Binding, Diag, Unknown, qualify};

/// How each provenance reason is remedied (`_FIXES`).
fn fix_for(reason: &str) -> &'static str {
    match reason {
        "unannotated-param" => "annotate the parameter",
        "unannotated-import" => "annotate the export in the module it comes from",
        "unknown-value" => "assign it a value of a known type",
        "joined-branches" => "make every branch assign the same type",
        "rebound-name" => "do not rebind the name",
        "recursion" => "annotate the parameters and the return of the recursive function",
        "import-cycle" => "break the import cycle",
        "context-budget" => "call it with fewer distinct argument types",
        "shape-mismatch" => "make both sides the same type",
        "suppressed-import" => "fix the imported module first",
        "unknown-name" => "define it with a value of a known type",
        "function-value" => "call it where it is used; a function is not a value",
        "unknown-return" => "annotate the return",
        "unknown-lib" => "give its arguments known types",
        "unknown-lib-name" => "use a lib name that exists",
        "unknown-call" => "call defined functions and lib names only",
        "unknown-field" => "declare the field on a @udt class",
        "unknown-index" => "index a series, a tuple, or an array through array.get()",
        "not-pine" => "write it with Pine constructs only",
        "unknown-op" => "give the operands types the operation accepts together",
        _ => "give it a known type",
    }
}

/// Whether a name is one the transforms own (`_SYNTHETIC.match`).
fn synthetic(name: &str) -> bool {
    (name.len() >= 4 && name.starts_with("__") && name.ends_with("__")) || name.starts_with('·')
}

/// Whether an expression is something the transforms emitted (`_is_plumbing`).
fn is_plumbing(ast: &pyne_ast::Ast, node: NodeId) -> bool {
    walk(ast, node).into_iter().any(|sub| match ast.kind(sub) {
        Kind::Name { id, .. } => synthetic(id),
        Kind::Attribute { attr, .. } => attr == "__class__",
        Kind::Call { func, .. } => {
            let callee = dotted(ast, *func).unwrap_or_default();
            callee.starts_with("__sec_") || callee == "run" || callee == "shadowed_namespace"
        }
        _ => false,
    })
}

fn is_expr(ast: &pyne_ast::Ast, node: NodeId) -> bool {
    ast.kind(node).class().category == pyne_ast::Category::Expr
}

/// Whether a node is still untyped (`getattr(node, TY_ATTR, UNKNOWN) == UNKNOWN`).
fn untyped(ast: &pyne_ast::Ast, node: NodeId) -> bool {
    match ast.extra(node, TY_ATTR) {
        None => true,
        Some(pyne_ast::Value::Str(ty)) => ty == UNKNOWN,
        Some(_) => false,
    }
}

struct Report<'e, 'a, 'h> {
    engine: &'e Engine<'a, 'h>,
    parent_of: HashMap<NodeId, Option<NodeId>>,
    scope_of: HashMap<NodeId, String>,
    skip: HashSet<NodeId>,
    attr_bases: HashSet<NodeId>,
    statement_values: HashSet<NodeId>,
    untyped: Vec<(usize, usize, NodeId)>,
    at: Option<HashMap<(i64, i64), Vec<NodeId>>>,
    seen: HashSet<(String, i64, i64, String)>,
    reported_at: HashSet<(i64, i64)>,
    following: HashSet<NodeId>,
    diags: Vec<Diag>,
}

/// One diagnostic per root cause of an UNKNOWN value in the module (`unknown_diags`).
pub(super) fn unknown_diags(engine: &Engine<'_, '_>, tree: NodeId) -> R<Vec<Diag>> {
    let mut report = Report {
        engine,
        parent_of: HashMap::new(),
        scope_of: HashMap::new(),
        skip: HashSet::new(),
        attr_bases: HashSet::new(),
        statement_values: HashSet::new(),
        untyped: Vec::new(),
        at: None,
        seen: engine
            .diags
            .iter()
            .filter_map(|diag| diag.origin.as_ref())
            .map(|origin| (origin.reason.clone(), origin.line, origin.col, origin.detail.clone()))
            .collect(),
        reported_at: engine.diags.iter().map(|diag| (diag.line, diag.col)).collect(),
        following: HashSet::new(),
        diags: Vec::new(),
    };
    report.run(tree)?;
    Ok(report.diags)
}

impl<'e> Report<'e, '_, '_> {
    fn ast(&self) -> &'e pyne_ast::Ast {
        self.tree()
    }

    /// The stamped module, borrowed for as long as the engine is, not the report.
    fn tree(&self) -> &'e pyne_ast::Ast {
        let engine: &'e Engine = self.engine;
        engine.ast
    }

    fn line(&self, node: NodeId) -> i64 {
        self.ast()[node].loc.lineno.unwrap_or(0)
    }

    fn col(&self, node: NodeId) -> i64 {
        self.ast()[node].loc.col_offset.unwrap_or(0)
    }

    fn run(&mut self, tree: NodeId) -> R<()> {
        for node in self.index(tree) {
            let ast = self.ast();
            if self.attr_bases.contains(&node)
                && matches!(ast.kind(node), Kind::Name { .. } | Kind::Attribute { .. })
            {
                continue;
            }
            if !self.maximal(node) || self.statement_values.contains(&node) {
                continue;
            }
            if is_plumbing(ast, node) || self.line(node) == 0 {
                continue;
            }
            for root in self.roots(node) {
                self.report(root)?;
            }
        }
        self.diags.sort_by_key(|diag| (diag.line, diag.col));
        Ok(())
    }

    /// Record the parents and scopes the report reads (`_index`); the untyped value
    /// expressions in breadth-first order.
    fn index(&mut self, tree: NodeId) -> Vec<NodeId> {
        let ast = self.tree();
        let mut position = 0usize;
        let mut stack: Vec<(NodeId, Option<NodeId>, String, usize)> =
            vec![(tree, None, String::new(), 0)];
        while let Some((node, parent, mut scope, mut depth)) = stack.pop() {
            position += 1;
            if is_expr(ast, node) {
                if untyped(ast, node) {
                    self.parent_of.insert(node, parent);
                    self.untyped.push((depth, position, node));
                }
                match ast.kind(node) {
                    Kind::Name { ctx, .. } => {
                        self.scope_of.insert(node, scope.clone());
                        if *ctx == ExprContext::Load {
                            continue;
                        }
                    }
                    Kind::Call { .. } => {
                        self.scope_of.insert(node, scope.clone());
                    }
                    _ => {}
                }
            }
            scope = self.mark(node, scope);
            depth += 1;
            for child in children(ast, node).into_iter().rev() {
                stack.push((child, Some(node), scope.clone(), depth));
            }
        }
        let mut entries = self.untyped.clone();
        entries.sort_by_key(|&(depth, position, _)| (depth, position));
        entries
            .into_iter()
            .map(|(_, _, node)| node)
            .filter(|node| !self.skip.contains(node))
            .collect()
    }

    /// What under one node is not a value position (`_mark`).
    fn mark(&mut self, node: NodeId, scope: String) -> String {
        let ast = self.tree();
        let mut scope = scope;
        match ast.kind(node) {
            Kind::Call { func, args, .. } => {
                self.skip_subtree(*func);
                let callee = dotted(ast, *func);
                if matches!(callee.as_deref(), Some("method_call" | "lib.method_call"))
                    && let Some(&first) = args.first()
                {
                    self.skip_subtree(first);
                }
            }
            Kind::FunctionDef { name, args, decorator_list, returns, .. }
            | Kind::AsyncFunctionDef { name, args, decorator_list, returns, .. } => {
                for &decorator in decorator_list {
                    self.skip_subtree(decorator);
                }
                if let Kind::Arguments { posonlyargs, args, kwonlyargs, .. } = ast.kind(*args) {
                    for &arg in args.iter().chain(posonlyargs).chain(kwonlyargs) {
                        if let Kind::Arg { annotation: Some(annotation), .. } = ast.kind(arg) {
                            self.skip_subtree(*annotation);
                        }
                    }
                }
                if let Some(returns) = returns {
                    self.skip_subtree(*returns);
                }
                scope = qualify(&scope, name);
            }
            Kind::ClassDef { bases, decorator_list, .. } => {
                for &decorator in decorator_list {
                    self.skip_subtree(decorator);
                }
                for &base in bases {
                    self.skip_subtree(base);
                }
                for value in self.engine.factory.of(ast, node) {
                    self.skip_subtree(value);
                }
            }
            Kind::AnnAssign { annotation, .. } => self.skip_subtree(*annotation),
            Kind::MatchCase { pattern, .. } => self.skip_subtree(*pattern),
            Kind::ExceptHandler { r#type: Some(caught), .. } => self.skip_subtree(*caught),
            _ => {}
        }
        match ast.kind(node) {
            Kind::Expr { value } => {
                if matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. }) {
                    self.skip.insert(*value);
                } else {
                    self.statement_values.insert(*value);
                }
            }
            Kind::Attribute { value, .. } => {
                self.attr_bases.insert(*value);
            }
            Kind::Name { ctx, .. }
            | Kind::Subscript { ctx, .. }
            | Kind::Tuple { ctx, .. }
            | Kind::List { ctx, .. }
            | Kind::Starred { ctx, .. }
                if *ctx != ExprContext::Load =>
            {
                self.skip.insert(node);
                if matches!(ast.kind(node), Kind::Tuple { .. } | Kind::List { .. }) {
                    self.skip_subtree(node);
                }
            }
            _ => {}
        }
        scope
    }

    /// Mark every expression of a subtree as no value position (`_skip_subtree`).
    fn skip_subtree(&mut self, node: NodeId) {
        let ast = self.tree();
        let mut node = node;
        while let Kind::Attribute { value, .. } = ast.kind(node) {
            self.skip.insert(node);
            node = *value;
        }
        if matches!(ast.kind(node), Kind::Name { .. }) {
            self.skip.insert(node);
            return;
        }
        self.skip.extend(walk(ast, node));
    }

    /// The untyped expressions starting at each position (`_positions`).
    fn positions(&mut self) -> &HashMap<(i64, i64), Vec<NodeId>> {
        if self.at.is_none() {
            let ast = self.tree();
            let mut at: HashMap<(i64, i64), Vec<NodeId>> = HashMap::new();
            for &(_, _, node) in &self.untyped {
                let loc = ast[node].loc;
                if let Some(line) = loc.lineno {
                    at.entry((line, loc.col_offset.unwrap_or(0))).or_default().push(node);
                }
            }
            self.at = Some(at);
        }
        self.at.as_ref().expect("built")
    }

    /// Whether no enclosing value expression is unknown too (`_maximal`).
    fn maximal(&self, node: NodeId) -> bool {
        let ast = self.ast();
        let parent = self.parent_of.get(&node).copied().flatten();
        let Some(parent) = parent else { return true };
        if let Kind::Call { func, .. } = ast.kind(parent) {
            if node != *func && is_plumbing(ast, parent) {
                return false;
            }
            return node == *func || get_ty(ast, parent) != UNKNOWN;
        }
        !(is_expr(ast, parent) && get_ty(ast, parent) == UNKNOWN)
    }

    /// Descend through unknown children to the expressions that started it (`_roots`).
    fn roots(&self, node: NodeId) -> Vec<NodeId> {
        let ast = self.ast();
        let below: Vec<NodeId> = self
            .value_children(node)
            .into_iter()
            .filter(|&child| {
                get_ty(ast, child) == UNKNOWN
                    && !is_plumbing(ast, child)
                    && self.line(child) != 0
                    && !self.skip.contains(&child)
            })
            .collect();
        if below.is_empty() {
            return vec![node];
        }
        below.into_iter().flat_map(|child| self.roots(child)).collect()
    }

    fn value_children(&self, node: NodeId) -> Vec<NodeId> {
        let ast = self.ast();
        match ast.kind(node) {
            Kind::Call { args, keywords, .. } => {
                let mut out = args.clone();
                out.extend(keywords.iter().filter_map(|&keyword| match ast.kind(keyword) {
                    Kind::Keyword { value, .. } => Some(*value),
                    _ => None,
                }));
                out
            }
            Kind::Attribute { value, .. } => {
                if matches!(ast.kind(*value), Kind::Name { .. }) && self.binding(*value).is_some() {
                    return vec![*value];
                }
                if matches!(ast.kind(*value), Kind::Name { .. } | Kind::Attribute { .. }) {
                    vec![]
                } else {
                    vec![*value]
                }
            }
            Kind::Lambda { .. } => vec![],
            _ => children(ast, node).into_iter().filter(|&child| is_expr(ast, child)).collect(),
        }
    }

    /// The binding a name reads, in its scope or any enclosing one (`_binding`).
    fn binding(&self, name: NodeId) -> Option<&Binding> {
        let Kind::Name { id, .. } = self.ast().kind(name) else { return None };
        let mut scope = self.scope_of.get(&name).cloned().unwrap_or_default();
        loop {
            if let Some(found) = self.engine.bindings.get(&scope).and_then(|names| names.get(id)) {
                return Some(found);
            }
            if scope.is_empty() {
                return None;
            }
            scope = scope.rsplit_once('·').map_or(String::new(), |(head, _)| head.to_string());
        }
    }

    /// Whether a bare callee names a function this module defines (`_is_user_function`).
    fn is_user_function(&self, callee: &str, node: NodeId) -> bool {
        let mut scope = self.scope_of.get(&node).cloned().unwrap_or_default();
        loop {
            if self.engine.funcs.contains_key(&qualify(&scope, callee)) {
                return true;
            }
            if scope.is_empty() {
                return false;
            }
            scope = scope.rsplit_once('·').map_or(String::new(), |(head, _)| head.to_string());
        }
    }

    /// Whether a diagnostic already stands inside an unknown expression (`_covered`).
    fn covered(&self, node: NodeId) -> bool {
        let ast = self.ast();
        walk(ast, node).into_iter().any(|sub| {
            sub != node
                && is_expr(ast, sub)
                && self.reported_at.contains(&(self.line(sub), self.col(sub)))
        })
    }

    /// The unknown value expression a binding's provenance points at (`_assigned_from`).
    fn assigned_from(&mut self, origin: &Unknown) -> Option<NodeId> {
        if origin.reason != "unknown-value" {
            return None;
        }
        let candidates =
            self.positions().get(&(origin.line, origin.col)).cloned().unwrap_or_default();
        let ast = self.ast();
        candidates
            .into_iter()
            .find(|&candidate| get_ty(ast, candidate) == UNKNOWN && !self.skip.contains(&candidate))
    }

    fn report(&mut self, node: NodeId) -> R<()> {
        let (line, col) = (self.line(node), self.col(node));
        let ast = self.tree();
        if self.reported_at.contains(&(line, col)) || is_plumbing(ast, node) || self.covered(node) {
            return Ok(());
        }
        let at = |reason: &str, detail: &str| Unknown {
            reason: reason.to_string(),
            line,
            col,
            detail: detail.to_string(),
        };
        let (origin, message) = match ast.kind(node) {
            Kind::Name { id, .. } => {
                let id = id.clone();
                let origin = self
                    .binding(node)
                    .and_then(|binding| binding.unknown.clone())
                    .unwrap_or_else(|| at("unknown-name", &id));
                if let Some(assigned) = self.assigned_from(&origin) {
                    if self.following.insert(assigned) {
                        for root in self.roots(assigned) {
                            self.report(root)?;
                        }
                    }
                    return Ok(());
                }
                if origin.reason == "unknown-name" && self.is_user_function(&id, node) {
                    (at("function-value", &id), format!("'{id}' is a function, not a value"))
                } else {
                    (origin, format!("'{id}' has no known type"))
                }
            }
            Kind::Call { func, args, .. } => match dotted(ast, *func) {
                None => (at("unknown-call", ""), "this call has no known type".to_string()),
                Some(callee) if callee == "method_call" && args.len() >= 2 => {
                    let selector = args[0];
                    let method = match ast.kind(selector) {
                        Kind::Constant { value: Constant::Str(text), .. } => text.clone(),
                        _ => match dotted(ast, selector) {
                            Some(spelled) if !spelled.is_empty() => spelled,
                            _ => crate::unparse::unparse(ast, selector, self.engine.host.python())?,
                        },
                    };
                    (
                        at("unknown-return", &method),
                        format!("the call to method '{method}' has no known type"),
                    )
                }
                Some(callee) if callee.starts_with("lib.") => (
                    at("unknown-lib", &callee),
                    format!("'{}' has no known type here", &callee[4..]),
                ),
                Some(callee) if self.is_user_function(&callee, node) => (
                    at("unknown-return", &callee),
                    format!("the call to '{callee}' has no known type"),
                ),
                Some(callee) => (
                    at("unknown-call", &callee),
                    format!("the call to '{callee}' has no known type"),
                ),
            },
            Kind::Attribute { value, attr, .. } => match dotted(ast, node) {
                Some(spelled) if spelled.starts_with("lib.") => (
                    at("unknown-lib-name", &spelled),
                    format!("'{}' is not a known lib name", &spelled[4..]),
                ),
                _ => (
                    at("unknown-field", attr),
                    format!("'.{attr}' has no known type on {}", render_ty(get_ty(ast, *value))),
                ),
            },
            Kind::Subscript { value, .. } => (
                at("unknown-index", ""),
                format!("indexing {} has no known type", render_ty(get_ty(ast, *value))),
            ),
            Kind::BinOp { .. }
            | Kind::UnaryOp { .. }
            | Kind::BoolOp { .. }
            | Kind::Compare { .. }
            | Kind::IfExp { .. } => {
                let operands = match ast.kind(node) {
                    Kind::IfExp { body, orelse, .. } => vec![*body, *orelse],
                    _ => self.value_children(node),
                };
                let over: Vec<String> =
                    operands.iter().map(|&operand| render_ty(get_ty(ast, operand))).collect();
                let spelled =
                    crate::unparse::unparse_prefix(ast, node, 40, self.engine.host.python())?;
                (
                    at("unknown-op", ast.name(node)),
                    format!("'{spelled}' has no known type over {}", over.join(", ")),
                )
            }
            _ => {
                let spelled =
                    crate::unparse::unparse_prefix(ast, node, 40, self.engine.host.python())?;
                (at("not-pine", ast.name(node)), format!("'{spelled}' is not a Pine expression"))
            }
        };
        let key = (origin.reason.clone(), origin.line, origin.col, origin.detail.clone());
        if !self.seen.insert(key) {
            return Ok(());
        }
        let fix = if origin.reason == "unannotated-param" && !origin.detail.is_empty() {
            format!("annotate '{}'", origin.detail)
        } else if origin.reason == "unknown-return" && !origin.detail.is_empty() {
            format!("annotate the return of '{}'", origin.detail)
        } else {
            fix_for(&origin.reason).to_string()
        };
        self.diags.push(Diag {
            message,
            line,
            col,
            origin: Some(origin),
            fix,
            end_line: 0,
            end_col: 0,
        });
        Ok(())
    }
}
