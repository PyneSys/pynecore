//! Port of `transformers/security_closed_shift_check.py`: exact verification of the
//! `closed_shift` flag on the LOWERED tree.
//!
//! After the series step a runtime history reference is exactly one of two forms:
//! `<state param>[slot][k]` with `k` an int constant >= 1, or `inline_series(expr, k)`
//! (while the module has not bound that spelling itself). A context whose lowered
//! `__sec_write__` expression is not built from those forms (or a tuple/list of
//! them, or a name certainly bound to one) cannot be `closed_shift`, and the flag is
//! cleared -- for the context AND every member of its merge group. The pass never
//! sets the flag to True.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{children, walk, walk_statements};
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId};

use crate::slot_layout::{DEFAULT_STATE_PARAM, ModuleLayout};
use crate::steps::security::contexts::{ctx_get, iter_contexts};
use crate::{Context, Error};

/// The modules the pipeline imports the history primitives FROM: an import of
/// `inline_series` out of one of them is the primitive, not a shadow of it.
const PRIMITIVE_MODULES: &[&str] = &["pynecore.core.series"];

/// Every name a lowered state-vector reference can be rooted at.
fn state_params(layout: &ModuleLayout) -> HashSet<String> {
    let mut names = HashSet::from([DEFAULT_STATE_PARAM.to_string()]);
    for scope in layout.scopes.values() {
        names.insert(scope.state_param.clone());
    }
    names
}

/// The value of an int constant that is no bool, `None` for anything else.
fn int_constant(ast: &Ast, node: NodeId) -> Option<&Int> {
    match ast.kind(node) {
        Kind::Constant { value: Constant::Int(value), .. } => Some(value),
        _ => None,
    }
}

/// Whether a node is an int constant >= 1 (a history offset).
fn is_history_index(ast: &Ast, node: NodeId) -> bool {
    match int_constant(ast, node) {
        Some(Int::Small(value)) => *value >= 1,
        Some(Int::Big(digits)) => !digits.starts_with('-'),
        None => false,
    }
}

/// Whether a node is a `<state param>[slot]` series-buffer reference.
fn is_slot_ref(ast: &Ast, node: NodeId, state_params: &HashSet<String>) -> bool {
    let Kind::Subscript { value, slice, .. } = ast.kind(node) else { return false };
    matches!(ast.kind(*value), Kind::Name { id, .. } if state_params.contains(id))
        && int_constant(ast, *slice).is_some()
}

/// Whether a node is the `inline_series(expr, k>=1)` history call, while the
/// spelling still means PyneCore's own primitive in this module.
fn is_inline_series(ast: &Ast, node: NodeId, primitive: bool) -> bool {
    if !primitive {
        return false;
    }
    let Kind::Call { func, args, keywords } = ast.kind(node) else { return false };
    if args.len() != 2 || !keywords.is_empty() {
        return false;
    }
    let name = match ast.kind(*func) {
        Kind::Name { id, .. } => id,
        Kind::Attribute { attr, .. } => attr,
        _ => return false,
    };
    name == "inline_series" && is_history_index(ast, args[1])
}

/// The parameters of a definition or lambda, as the Python pass lists them.
fn parameters(ast: &Ast, args: NodeId) -> Vec<NodeId> {
    let Kind::Arguments { posonlyargs, args, kwonlyargs, vararg, kwarg, .. } = ast.kind(args)
    else {
        unreachable!()
    };
    let mut out: Vec<NodeId> = posonlyargs.iter().chain(args).chain(kwonlyargs).copied().collect();
    out.extend(vararg.iter().chain(kwarg.iter()).copied());
    out
}

fn arg_name(ast: &Ast, arg: NodeId) -> &str {
    let Kind::Arg { arg, .. } = ast.kind(arg) else { unreachable!() };
    arg
}

/// Whether the module binds `name` itself instead of inheriting it (`_binds_name`).
fn binds_name(ast: &Ast, module: NodeId, name: &str) -> bool {
    for node in walk(ast, module) {
        let kind = ast.kind(node);
        if let Kind::FunctionDef { name: def_name, .. }
        | Kind::AsyncFunctionDef { name: def_name, .. } = kind
            && def_name == name
        {
            return true;
        }
        if let Kind::FunctionDef { args, .. }
        | Kind::AsyncFunctionDef { args, .. }
        | Kind::Lambda { args, .. } = kind
        {
            if parameters(ast, *args).into_iter().any(|arg| arg_name(ast, arg) == name) {
                return true;
            }
            continue;
        }
        if let Kind::Import { names } | Kind::ImportFrom { names, .. } = kind {
            for &alias in names {
                let Kind::Alias { name: alias_name, asname } = ast.kind(alias) else {
                    unreachable!()
                };
                let bound = match asname {
                    Some(asname) => asname.as_str(),
                    None => alias_name.split('.').next().unwrap_or_default(),
                };
                if bound != name {
                    continue;
                }
                if let Kind::ImportFrom { module: Some(module), level, .. } = kind
                    && level.unwrap_or(0) == 0
                    && PRIMITIVE_MODULES.contains(&module.as_str())
                    && asname.is_none()
                {
                    continue;
                }
                return true;
            }
            continue;
        }
        let targets: Vec<NodeId> = match kind {
            Kind::Assign { targets, .. } => targets.clone(),
            Kind::AnnAssign { target, .. } | Kind::AugAssign { target, .. } => vec![*target],
            Kind::For { target, .. }
            | Kind::AsyncFor { target, .. }
            | Kind::Comprehension { target, .. } => vec![*target],
            Kind::WithItem { optional_vars, .. } => optional_vars.iter().copied().collect(),
            _ => Vec::new(),
        };
        for target in targets {
            if walk(ast, target)
                .into_iter()
                .any(|sub| matches!(ast.kind(sub), Kind::Name { id, .. } if id == name))
            {
                return true;
            }
        }
    }
    false
}

/// A node and everything under it that belongs to the SAME scope: the walk stops at
/// a definition, the node itself included (`_own_nodes`).
fn own_nodes(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut pending = vec![node];
    while let Some(current) = pending.pop() {
        out.push(current);
        if matches!(
            ast.kind(current),
            Kind::FunctionDef { .. }
                | Kind::AsyncFunctionDef { .. }
                | Kind::ClassDef { .. }
                | Kind::Lambda { .. }
        ) {
            continue;
        }
        pending.extend(children(ast, current));
    }
    out
}

/// The statement list of a scope (the module or a definition).
fn scope_body(ast: &Ast, scope: NodeId) -> &[NodeId] {
    match ast.kind(scope) {
        Kind::Module { body, .. }
        | Kind::FunctionDef { body, .. }
        | Kind::AsyncFunctionDef { body, .. } => body,
        _ => unreachable!(),
    }
}

/// The value `name` certainly holds where `target` stands, or `None` (`_name_binding`).
///
/// The one shape this answers for is a single plain `name = <expr>` statement in the
/// scope's own statement list, standing in front of the statement that holds
/// `target`, with no other binding of the spelling anywhere in the scope.
fn name_binding(ast: &Ast, name: &str, scope: NodeId, target: NodeId) -> Option<NodeId> {
    if let Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. } = ast.kind(scope)
        && parameters(ast, *args).into_iter().any(|arg| arg_name(ast, arg) == name)
    {
        return None;
    }
    let body = scope_body(ast, scope);
    let mut bound: Option<NodeId> = None;
    let mut bound_index = 0;
    for (index, &stmt) in body.iter().enumerate() {
        let plain = match ast.kind(stmt) {
            Kind::Assign { targets, .. }
                if targets.len() == 1
                    && matches!(ast.kind(targets[0]), Kind::Name { id, .. } if id == name) =>
            {
                Some(targets[0])
            }
            _ => None,
        };
        for sub in walk(ast, stmt) {
            match ast.kind(sub) {
                Kind::FunctionDef { name: def_name, .. }
                | Kind::AsyncFunctionDef { name: def_name, .. }
                | Kind::ClassDef { name: def_name, .. }
                    if def_name == name =>
                {
                    return None;
                }
                Kind::Global { names } | Kind::Nonlocal { names }
                    if names.iter().any(|declared| declared == name) =>
                {
                    return None;
                }
                Kind::Name { id, ctx: ExprContext::Store | ExprContext::Del }
                    if id == name && Some(sub) != plain =>
                {
                    return None;
                }
                Kind::Arg { arg, .. } if arg == name => return None,
                _ => {}
            }
        }
        if plain.is_some() {
            if bound.is_some() {
                return None;
            }
            let Kind::Assign { value, .. } = ast.kind(stmt) else { unreachable!() };
            bound = Some(*value);
            bound_index = index;
        }
    }
    let bound = bound?;
    for &stmt in &body[bound_index + 1..] {
        if own_nodes(ast, stmt).contains(&target) {
            return Some(bound);
        }
    }
    None
}

/// What the closed-shift check reads besides the tree.
struct Check<'a> {
    state_params: &'a HashSet<String>,
    primitive: bool,
}

impl Check<'_> {
    /// Whether the lowered expression reads closed bars ONLY (`_is_history_form`).
    fn is_history_form(
        &self,
        ast: &Ast,
        node: NodeId,
        scope: Option<NodeId>,
        read: Option<NodeId>,
    ) -> bool {
        if let Kind::Tuple { elts, .. } | Kind::List { elts, .. } = ast.kind(node) {
            let read = read.or(Some(node));
            return !elts.is_empty()
                && elts.iter().all(|&elt| self.is_history_form(ast, elt, scope, read));
        }
        if is_inline_series(ast, node, self.primitive) {
            return true;
        }
        if let Kind::Subscript { value, slice, .. } = ast.kind(node)
            && is_history_index(ast, *slice)
            && is_slot_ref(ast, *value, self.state_params)
        {
            return true;
        }
        if let Kind::Name { id, .. } = ast.kind(node)
            && let Some(scope) = scope
        {
            let read = read.unwrap_or(node);
            // The binding stands in front of the read, so resolving its value cannot
            // come back to this name and the recursion terminates.
            if let Some(bound) = name_binding(ast, id, scope, read) {
                return self.is_history_form(ast, bound, Some(scope), Some(read));
            }
        }
        false
    }
}

/// The `__security_contexts__` dict literal of the module, if it has one.
fn find_contexts(ast: &Ast, module: NodeId) -> Option<NodeId> {
    for &stmt in scope_body(ast, module) {
        let Kind::Assign { targets, value, .. } = ast.kind(stmt) else { continue };
        if let [target] = targets[..]
            && matches!(ast.kind(target), Kind::Name { id, .. } if id == "__security_contexts__")
            && matches!(ast.kind(*value), Kind::Dict { .. })
        {
            return Some(*value);
        }
    }
    None
}

/// Every `__sec_write__(sid, expr)` expression of the module, by sid, with the scope
/// whose statement list the call stands in (`_collect_writes`).
fn collect_writes(ast: &Ast, module: NodeId) -> HashMap<String, Vec<(NodeId, NodeId)>> {
    let mut scopes = vec![module];
    // A definition is a statement: the statements alone hold every one.
    scopes.extend(walk_statements(ast, module).into_iter().filter(|&node| {
        matches!(ast.kind(node), Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
    }));
    let mut writes: HashMap<String, Vec<(NodeId, NodeId)>> = HashMap::new();
    for scope in scopes {
        for &stmt in scope_body(ast, scope) {
            for node in own_nodes(ast, stmt) {
                let Kind::Call { func, args, .. } = ast.kind(node) else { continue };
                if !matches!(ast.kind(*func), Kind::Name { id, .. } if id == "__sec_write__")
                    || args.len() != 2
                {
                    continue;
                }
                if let Kind::Constant { value: Constant::Str(sid), .. } = ast.kind(args[0]) {
                    writes.entry(sid.clone()).or_default().push((args[1], scope));
                }
            }
        }
    }
    writes
}

/// Whether a context's `closed_shift` entry is the constant True.
fn is_flag_true(ast: &Ast, ctx: NodeId) -> bool {
    ctx_get(ast, ctx, "closed_shift").is_some_and(|value| {
        matches!(ast.kind(value), Kind::Constant { value: Constant::Bool(true), .. })
    })
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("VerifyClosedShift on a non-module".into()));
    }
    let Some(contexts) = find_contexts(ast, module) else { return Ok(module) };
    let entries = iter_contexts(ast, contexts);
    let flagged: Vec<&String> =
        entries.iter().filter(|(_, ctx)| is_flag_true(ast, *ctx)).map(|(sid, _)| sid).collect();
    if flagged.is_empty() {
        return Ok(module);
    }

    let state_params = state_params(&ctx.slot_layout.borrow());
    let check =
        Check { state_params: &state_params, primitive: !binds_name(ast, module, "inline_series") };
    let writes = collect_writes(ast, module);
    let failed: HashSet<&String> = flagged
        .into_iter()
        .filter(|&sid| match writes.get(sid) {
            Some(exprs) if !exprs.is_empty() => !exprs
                .iter()
                .all(|&(expr, scope)| check.is_history_form(ast, expr, Some(scope), None)),
            _ => true,
        })
        .collect();
    if failed.is_empty() {
        return Ok(module);
    }

    // The merge groups: one child serves a whole group, so its round protocol must
    // be uniform
    let mut group_of: HashMap<&String, &Int> = HashMap::new();
    let mut members: HashMap<&Int, HashSet<&String>> = HashMap::new();
    for (sid, ctx) in &entries {
        if let Some(value) = ctx_get(ast, *ctx, "group")
            && let Some(group) = int_constant(ast, value)
        {
            group_of.insert(sid, group);
            members.entry(group).or_default().insert(sid);
        }
    }
    let mut cleared: HashSet<&String> = failed.clone();
    for sid in &failed {
        if let Some(group) = group_of.get(sid) {
            cleared.extend(members[group].iter().copied());
        }
    }

    let flags: Vec<NodeId> = entries
        .iter()
        .filter(|(sid, _)| cleared.contains(sid))
        .filter_map(|(_, ctx)| ctx_get(ast, *ctx, "closed_shift"))
        .collect();
    for value in flags {
        if let Kind::Constant { value: flag @ Constant::Bool(true), .. } = ast.kind_mut(value) {
            *flag = Constant::Bool(false);
        }
    }
    Ok(module)
}
