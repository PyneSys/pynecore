//! Port of `transformers/outer_write.py`: an object created outside a function may
//! not be modified inside one.
//!
//! The step changes nothing; it rejects a module whose function writes, directly,
//! into an object a module-level name holds. A rejected module answers
//! [`Error::Rejected`], and the Python host runs the Python step, which raises the
//! source-located `SyntaxError` the script author sees.

use std::collections::HashSet;

use pyne_ast::walk::{children, walk, walk_statements};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// Methods of the Python containers a script can hold that write the receiver.
const MUTATING_METHODS: [&str; 13] = [
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "sort",
    "reverse",
    "update",
    "setdefault",
    "popitem",
    "add",
    "discard",
];

/// Per collection namespace, the library functions that modify their first argument.
fn mutating_lib_calls(namespace: &str) -> Option<&'static [&'static str]> {
    match namespace {
        "array" => Some(&[
            "clear", "concat", "fill", "insert", "pop", "push", "remove", "reverse", "set",
            "shift", "sort", "unshift",
        ]),
        "matrix" => Some(&[
            "add_col",
            "add_row",
            "concat",
            "fill",
            "remove_col",
            "remove_row",
            "reshape",
            "reverse",
            "set",
            "sort",
            "swap_columns",
            "swap_rows",
        ]),
        "map" => Some(&["clear", "put", "put_all", "remove"]),
        _ => None,
    }
}

/// Prefix of the harness functions of a `@pyne` test script, exempt from the rule.
const TEST_PREFIX: &str = "__test_";

/// The base `Name` of an attribute / subscript chain, if it has one.
fn chain_root(ast: &Ast, node: NodeId) -> Option<&str> {
    let mut root = node;
    loop {
        match ast.kind(root) {
            Kind::Attribute { value, .. } | Kind::Subscript { value, .. } => root = *value,
            Kind::Name { id, .. } => return Some(id),
            _ => return None,
        }
    }
}

/// `(namespace, function)` of a `lib.<array|matrix|map>.<fn>` callee.
fn collection_call(ast: &Ast, func: NodeId) -> Option<(&str, &str)> {
    let Kind::Attribute { value: owner, attr: fname, .. } = ast.kind(func) else { return None };
    let Kind::Attribute { value: base, attr: namespace, .. } = ast.kind(*owner) else {
        return None;
    };
    mutating_lib_calls(namespace)?;
    match ast.kind(*base) {
        Kind::Name { id, .. } if id == "lib" => Some((namespace, fname)),
        _ => None,
    }
}

/// Names the module's top level binds to an object of its own (assignment targets only).
fn module_names(ast: &Ast, module: NodeId) -> HashSet<String> {
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let mut names = HashSet::new();
    for &stmt in body {
        let targets: &[NodeId] = match ast.kind(stmt) {
            Kind::Assign { targets, .. } => targets,
            Kind::AnnAssign { target, .. } => std::slice::from_ref(target),
            _ => &[],
        };
        for &target in targets {
            for sub in walk(ast, target) {
                if let Kind::Name { id, .. } = ast.kind(sub)
                    && !id.starts_with("__")
                {
                    names.insert(id.clone());
                }
            }
        }
    }
    names
}

/// Whether a node is one of the candidates [`OuterWrite::check_node`] can report.
fn is_write(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Global { .. }
            | Kind::Assign { .. }
            | Kind::AnnAssign { .. }
            | Kind::AugAssign { .. }
            | Kind::Delete { .. }
            | Kind::Call { .. }
    )
}

fn is_function(kind: &Kind) -> bool {
    matches!(kind, Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

/// The function definitions standing directly in a node's own scope; a class body
/// is followed, a `def` and a `lambda` are not.
fn nested_defs(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut found = Vec::new();
    let mut pending = children(ast, node);
    while let Some(child) = pending.pop() {
        match ast.kind(child) {
            kind if is_function(kind) => found.push(child),
            Kind::Lambda { .. } => {}
            _ => pending.extend(children(ast, child)),
        }
    }
    found
}

/// What one function body holds for the check: its local names, its write
/// candidates in pre-order, and its nested definitions in `nested_defs` order.
struct Scan {
    names: HashSet<String>,
    writes: Vec<NodeId>,
    defs: Vec<NodeId>,
}

fn scan_function(ast: &Ast, func: NodeId) -> Scan {
    let (Kind::FunctionDef { args, body, .. } | Kind::AsyncFunctionDef { args, body, .. }) =
        ast.kind(func)
    else {
        unreachable!()
    };
    let Kind::Arguments { posonlyargs, args: positional, vararg, kwonlyargs, kwarg, .. } =
        ast.kind(*args)
    else {
        unreachable!()
    };
    let mut names = HashSet::new();
    for &param in posonlyargs.iter().chain(positional).chain(kwonlyargs).chain(vararg).chain(kwarg)
    {
        let Kind::Arg { arg, .. } = ast.kind(param) else { unreachable!() };
        names.insert(arg.clone());
    }
    let mut writes = Vec::new();
    let mut defs = Vec::new();
    // Pre-order, left to right: (node, whether it stands in a class body)
    let mut pending: Vec<(NodeId, bool)> = body.iter().rev().map(|&stmt| (stmt, false)).collect();
    while let Some((node, mut in_class)) = pending.pop() {
        let kind = ast.kind(node);
        if is_function(kind) {
            defs.push(node);
            continue;
        }
        if matches!(kind, Kind::Lambda { .. }) {
            continue;
        }
        if matches!(kind, Kind::ClassDef { .. }) {
            in_class = true;
        } else if !in_class {
            match kind {
                Kind::Name { id, ctx } => {
                    if matches!(ctx, ExprContext::Store | ExprContext::Del) {
                        names.insert(id.clone());
                    }
                    continue;
                }
                Kind::Import { names: aliases } | Kind::ImportFrom { names: aliases, .. } => {
                    for &alias in aliases {
                        let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                        let bound = match asname {
                            Some(asname) if !asname.is_empty() => asname.clone(),
                            _ => name.split('.').next().unwrap_or_default().to_string(),
                        };
                        names.insert(bound);
                    }
                }
                kind if is_write(kind) => writes.push(node),
                _ => {}
            }
        }
        pending.extend(children(ast, node).into_iter().rev().map(|child| (child, in_class)));
    }
    // Definitions and classes bind their names in the scope they stand in, the
    // ones nested deeper in this function included
    for stmt in walk_statements(ast, func) {
        if stmt == func {
            continue;
        }
        match ast.kind(stmt) {
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                names.insert(name.clone());
            }
            _ => {}
        }
    }
    defs.reverse();
    Scan { names, writes, defs }
}

struct OuterWrite {
    outer: HashSet<String>,
}

impl OuterWrite {
    /// Check one function body and every definition nested in it.
    fn check_function(
        &self,
        ast: &Ast,
        func: NodeId,
        shadowed: &HashSet<String>,
    ) -> Result<(), Error> {
        let (Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. }) = ast.kind(func)
        else {
            unreachable!()
        };
        if name.starts_with(TEST_PREFIX) {
            return Ok(());
        }
        let Scan { names, writes, defs } = scan_function(ast, func);
        let locals_here: HashSet<String> = shadowed.union(&names).cloned().collect();
        for node in writes {
            self.check_node(ast, node, &locals_here)?;
        }
        for inner in defs {
            self.check_function(ast, inner, &locals_here)?;
        }
        Ok(())
    }

    /// Whether a chain root names a module-level binding the function does not shadow.
    fn is_outer(&self, root: Option<&str>, shadowed: &HashSet<String>) -> bool {
        root.is_some_and(|root| !shadowed.contains(root) && self.outer.contains(root))
    }

    /// Reject the node when it writes an object a module-level name holds.
    fn check_node(&self, ast: &Ast, node: NodeId, shadowed: &HashSet<String>) -> Result<(), Error> {
        let kind = ast.kind(node);
        if let Kind::Global { names } = kind {
            // Whatever a ``global`` names is module-level storage
            return Err(reject(&names[0], "is rebound"));
        }
        let targets: &[NodeId] = match kind {
            Kind::Assign { targets, .. } | Kind::Delete { targets } => targets,
            Kind::AnnAssign { target, .. } | Kind::AugAssign { target, .. } => {
                std::slice::from_ref(target)
            }
            _ => &[],
        };
        for &target in targets {
            if !matches!(ast.kind(target), Kind::Attribute { .. } | Kind::Subscript { .. }) {
                continue;
            }
            let root = chain_root(ast, target);
            if self.is_outer(root, shadowed) {
                return Err(reject(root.unwrap_or_default(), "is modified"));
            }
        }
        let Kind::Call { func, args, keywords } = kind else { return Ok(()) };
        let Kind::Attribute { value: receiver, attr, .. } = ast.kind(*func) else {
            return Ok(());
        };
        if let Some((namespace, fname)) = collection_call(ast, *func) {
            if !mutating_lib_calls(namespace).is_some_and(|calls| calls.contains(&fname)) {
                return Ok(());
            }
            let root = first_argument(ast, args, keywords).and_then(|first| chain_root(ast, first));
            if self.is_outer(root, shadowed) {
                return Err(reject(root.unwrap_or_default(), "is modified"));
            }
            return Ok(());
        }
        if !MUTATING_METHODS.contains(&attr.as_str()) {
            return Ok(());
        }
        let root = chain_root(ast, *receiver);
        if self.is_outer(root, shadowed) {
            return Err(reject(root.unwrap_or_default(), "is modified"));
        }
        Ok(())
    }
}

/// The collection a `lib.<array|matrix|map>.<fn>` call acts on: its first
/// positional argument, or the `id` keyword; an unpacked positional names nothing.
fn first_argument(ast: &Ast, args: &[NodeId], keywords: &[NodeId]) -> Option<NodeId> {
    if let Some(&first) = args.first() {
        return (!matches!(ast.kind(first), Kind::Starred { .. })).then_some(first);
    }
    keywords.iter().find_map(|&keyword| match ast.kind(keyword) {
        Kind::Keyword { arg: Some(arg), value } if arg == "id" => Some(*value),
        _ => None,
    })
}

fn reject(name: &str, verb: &str) -> Error {
    Error::Rejected(format!(
        "'{name}' {verb} inside a function: an object created outside a function cannot be \
         modified inside one. Module-level state is not restored when a bar is re-executed; \
         declare the value as 'Persistent' or 'IBPersistent' inside the function instead."
    ))
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("OuterWrite on a non-module".into()));
    }
    let check = OuterWrite { outer: module_names(ast, module) };
    for func in nested_defs(ast, module) {
        check.check_function(ast, func, &HashSet::new())?;
    }
    Ok(module)
}
