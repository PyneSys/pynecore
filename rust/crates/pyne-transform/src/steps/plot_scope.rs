//! Port of `transformers/plot_scope.py`: plot declarations must execute directly in
//! the script entry.
//!
//! The step changes nothing; it rejects a module. A rejected module answers
//! [`Error::Rejected`], and the Python host runs the Python step, which raises the
//! source-located `SyntaxError` the script author sees.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{child_statements, children, walk};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error};

const PLOT_NAMES: [&str; 11] = [
    "plot",
    "plotshape",
    "plotchar",
    "plotarrow",
    "plotcandle",
    "plotbar",
    "hline",
    "fill",
    "bgcolor",
    "barcolor",
    "alertcondition",
];

/// The plot function a dotted path names, if any.
fn plot_name(path: &str) -> Option<&'static str> {
    match path {
        "lib.plot.plot" => Some("plot"),
        "lib.hline.hline" => Some("hline"),
        _ => path
            .strip_prefix("lib.")
            .and_then(|name| PLOT_NAMES.iter().find(|&&n| n == name).copied()),
    }
}

fn is_local_context(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::If { .. }
            | Kind::For { .. }
            | Kind::AsyncFor { .. }
            | Kind::While { .. }
            | Kind::Try { .. }
            | Kind::TryStar { .. }
            | Kind::With { .. }
            | Kind::AsyncWith { .. }
            | Kind::Match { .. }
            | Kind::BoolOp { .. }
            | Kind::IfExp { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
    )
}

struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, Vec<NodeId>>,
}

#[derive(Default)]
struct Collector {
    scopes: Vec<Scope>,
    owners: HashMap<NodeId, usize>,
    parents: HashMap<NodeId, NodeId>,
    allowed: HashSet<NodeId>,
    /// A single static alias of a plot function: the aliased reference -> the name it binds
    aliases: HashMap<NodeId, NodeId>,
    calls: Vec<NodeId>,
    references: Vec<NodeId>,
}

impl Collector {
    fn new_scope(&mut self, parent: Option<usize>) -> usize {
        self.scopes.push(Scope { parent, bindings: HashMap::new() });
        self.scopes.len() - 1
    }

    fn bind(&mut self, scope: usize, name: &str, value: NodeId) {
        self.scopes[scope].bindings.entry(name.to_string()).or_default().push(value);
    }

    /// The lexical scope that binds a name.
    fn resolve(&self, mut scope: usize, name: &str) -> Option<usize> {
        loop {
            if self.scopes[scope].bindings.contains_key(name) {
                return Some(scope);
            }
            scope = self.scopes[scope].parent?;
        }
    }

    fn collect(&mut self, ast: &Ast, current: NodeId, scope: usize, mut direct: bool) {
        self.owners.insert(current, scope);
        let kind = ast.kind(current);
        match kind {
            Kind::FunctionDef { name, args, body, .. }
            | Kind::AsyncFunctionDef { name, args, body, .. } => {
                if name.starts_with("__test_") {
                    return;
                }
                self.bind(scope, name, current);
                let child = self.function_scope(ast, scope, *args);
                let entry = matches!(kind, Kind::FunctionDef { .. })
                    && name == "main"
                    && self
                        .parents
                        .get(&current)
                        .is_some_and(|&parent| matches!(ast.kind(parent), Kind::Module { .. }));
                let body_ids: HashSet<NodeId> = body.iter().copied().collect();
                for part in children(ast, current) {
                    if !body_ids.contains(&part) {
                        self.parents.insert(part, current);
                        self.collect(ast, part, scope, false);
                    }
                }
                // Only the entry's direct body has to know whether an earlier
                // statement can return, and only until one can
                let mut may_return = !entry;
                for &stmt in body {
                    self.parents.insert(stmt, current);
                    self.collect(ast, stmt, child, !may_return);
                    if !may_return {
                        may_return = has_return(ast, stmt);
                    }
                }
                return;
            }
            Kind::Lambda { args, body } => {
                let child = self.function_scope(ast, scope, *args);
                for part in children(ast, current) {
                    self.parents.insert(part, current);
                    if part == *body {
                        self.collect(ast, part, child, false);
                    } else {
                        self.collect(ast, part, scope, false);
                    }
                }
                return;
            }
            Kind::ClassDef { name, .. } => {
                self.bind(scope, name, current);
                let child = self.new_scope(Some(scope));
                for part in children(ast, current) {
                    self.parents.insert(part, current);
                    self.collect(ast, part, child, false);
                }
                return;
            }
            _ => {}
        }
        let targets: Option<(Vec<NodeId>, Option<NodeId>)> = match kind {
            Kind::Assign { targets, value, .. } => Some((targets.clone(), Some(*value))),
            Kind::AnnAssign { target, value, .. } => Some((vec![*target], *value)),
            Kind::NamedExpr { target, value } => Some((vec![*target], Some(*value))),
            Kind::AugAssign { target, value, .. } => Some((vec![*target], Some(*value))),
            _ => None,
        };
        if let Some((targets, value)) = targets {
            let bound = value.unwrap_or(current);
            for &target in &targets {
                for name in walk(ast, target) {
                    if let Kind::Name { id, ctx: ExprContext::Store } = ast.kind(name) {
                        self.bind(scope, id, bound);
                    }
                }
            }
            if matches!(kind, Kind::Assign { .. } | Kind::AnnAssign { .. })
                && targets.len() == 1
                && matches!(ast.kind(targets[0]), Kind::Name { .. })
                && let Some(value) = value
                && matches!(ast.kind(value), Kind::Name { .. } | Kind::Attribute { .. })
            {
                self.aliases.insert(value, targets[0]);
            }
        }
        if matches!(kind, Kind::Call { .. }) {
            self.calls.push(current);
            if direct {
                self.allowed.insert(current);
            }
        }
        if matches!(
            kind,
            Kind::Name { ctx: ExprContext::Load, .. }
                | Kind::Attribute { ctx: ExprContext::Load, .. }
        ) {
            self.references.push(current);
        }
        if is_local_context(kind) {
            direct = false;
        }
        for part in children(ast, current) {
            self.parents.insert(part, current);
            self.collect(ast, part, scope, direct);
        }
    }

    fn function_scope(&mut self, ast: &Ast, scope: usize, args: NodeId) -> usize {
        let child = self.new_scope(Some(scope));
        let Kind::Arguments { posonlyargs, args: positional, vararg, kwonlyargs, kwarg, .. } =
            ast.kind(args)
        else {
            unreachable!()
        };
        for &param in
            posonlyargs.iter().chain(positional).chain(kwonlyargs).chain(vararg).chain(kwarg)
        {
            let Kind::Arg { arg, .. } = ast.kind(param) else { unreachable!() };
            self.scopes[child].bindings.insert(arg.clone(), Vec::new());
        }
        child
    }

    /// The dotted path a reference resolves to through single static bindings ('' when none).
    fn path(&self, ast: &Ast, value: NodeId, seen: &mut Vec<(usize, String)>) -> String {
        match ast.kind(value) {
            Kind::Attribute { value: base, attr, .. } => {
                let base = self.path(ast, *base, seen);
                if base.is_empty() { base } else { format!("{base}.{attr}") }
            }
            Kind::Name { id, .. } => {
                let Some(scope) = self.resolve(self.owners[&value], id) else {
                    return id.clone();
                };
                let key = (scope, id.clone());
                if seen.contains(&key) {
                    return String::new();
                }
                let bindings = &self.scopes[scope].bindings[id];
                if bindings.len() == 1
                    && matches!(ast.kind(bindings[0]), Kind::Name { .. } | Kind::Attribute { .. })
                {
                    let depth = seen.len();
                    seen.push(key);
                    let result = self.path(ast, bindings[0], seen);
                    seen.truncate(depth);
                    return result;
                }
                String::new()
            }
            _ => String::new(),
        }
    }
}

/// Whether a statement can return from its own enclosing function.
fn has_return(ast: &Ast, node: NodeId) -> bool {
    let mut pending = vec![node];
    while let Some(current) = pending.pop() {
        match ast.kind(current) {
            Kind::Return { .. } => return true,
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::ClassDef { .. } => {}
            _ => pending.extend(child_statements(ast, current)),
        }
    }
    false
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let mut collector = Collector::default();
    let root = collector.new_scope(None);
    collector.collect(ast, module, root, false);

    let reject = |name: &str| {
        Err(Error::Rejected(format!(
            "{name}() must be called unconditionally in the direct body of module-level main"
        )))
    };
    for &call in &collector.calls {
        let Kind::Call { func, .. } = ast.kind(call) else { unreachable!() };
        let path = collector.path(ast, *func, &mut Vec::new());
        if let Some(name) = plot_name(&path)
            && !collector.allowed.contains(&call)
        {
            return reject(name);
        }
    }
    for &reference in &collector.references {
        let path = collector.path(ast, reference, &mut Vec::new());
        let Some(name) = plot_name(&path) else { continue };
        let parent = collector.parents[&reference];
        match ast.kind(parent) {
            Kind::Call { func, .. } if *func == reference => continue,
            Kind::Attribute { value, attr, .. }
                if *value == reference
                    && (attr == "plot"
                        || attr == "hline"
                        || attr.starts_with("style_")
                        || attr.starts_with("linestyle_")) =>
            {
                continue;
            }
            _ => {}
        }
        if let Some(&target) = collector.aliases.get(&reference) {
            let Kind::Name { id, .. } = ast.kind(target) else { unreachable!() };
            if let Some(scope) = collector.resolve(collector.owners[&target], id)
                && collector.scopes[scope].bindings[id].len() == 1
            {
                continue;
            }
        }
        return reject(name);
    }
    Ok(module)
}
