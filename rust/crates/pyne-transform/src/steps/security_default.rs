//! Port of `transformers/security_default.py`: the missing value of a `request.security`
//! read whose inferred type is bool, or a tuple holding bools, becomes a call of the
//! runtime-mode bool factory, imported once at the top of the module; a scalar missing
//! value of a tuple read becomes one per element. A raw lower-timeframe read of a tuple
//! gets the `__ltf_unzip__` transposition with the width of the written tuple. A tuple read
//! unpacked into fewer names than the tuple gives takes a `[:n]` slice: its leading values,
//! as on TradingView.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{
    Transformer, Visited, Walk, clone, copy_location, descendants, future_end, generic_visit, walk,
    walk_statements,
};
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId, Value};

use crate::pine_type_rules::{
    BOOL, FLOAT, OBJECT, STR, TY_ATTR, UNKNOWN, elements_of, is_numeric, is_tuple, set_ty,
    stamp_lowering, tuple_of,
};
use crate::{Context, Error};

/// The alias the bool factory is imported under
const BOOL_NA_NAME: &str = "__sec_bool_na·__";

/// A node's Pine type (`get_ty`), refusing a stamp that is no string.
fn ty_of(ast: &Ast, id: NodeId) -> Result<String, Error> {
    match ast.extra(id, TY_ATTR) {
        None => Ok(UNKNOWN.to_string()),
        Some(Value::Str(ty)) => Ok(ty.clone()),
        Some(other) => Err(Error::Unsupported(format!("a {TY_ATTR} stamp {other:?}"))),
    }
}

struct SecurityDefault {
    used: bool,
}

impl SecurityDefault {
    /// The default with its bool elements resolved under the runtime mode, one per
    /// element of a tuple read.
    fn default(&mut self, ast: &mut Ast, default: NodeId, ty: &str) -> NodeId {
        if ty == BOOL {
            self.used = true;
            let func = ast.add(Kind::Name { id: BOOL_NA_NAME.into(), ctx: ExprContext::Load });
            let call = ast.add(Kind::Call { func, args: Vec::new(), keywords: Vec::new() });
            let call = stamp_lowering(ast, call, BOOL);
            copy_location(ast, call, default);
            return call;
        }
        if is_tuple(ty) {
            let mut types: Vec<String> = elements_of(ty).into_iter().map(str::to_string).collect();
            let mut ty = ty.to_string();
            let values: Vec<NodeId> = match ast.kind(default) {
                // The security transform took the width from the unpack target, which may
                // name fewer values than the tuple gives: the leading ones
                Kind::Tuple { elts, .. } if elts.len() <= types.len() => {
                    let defaults = elts.clone();
                    types.truncate(defaults.len());
                    ty = tuple_of(&types);
                    let values: Vec<NodeId> = defaults
                        .iter()
                        .zip(&types)
                        .map(|(&value, item)| self.default(ast, value, item))
                        .collect();
                    if values == defaults {
                        return default;
                    }
                    values
                }
                Kind::Tuple { .. } | Kind::List { .. } => return default,
                // A scalar default of a tuple read: the security transform could not see
                // the width (the expression is a call of a tuple-returning function, and
                // the read stands in a `return`), the type pass did. One `na` per element,
                // or the unpack at the call site raises
                _ => types.iter().map(|item| self.element_default(ast, default, item)).collect(),
            };
            let tuple = ast.add(Kind::Tuple { elts: values, ctx: ExprContext::Load });
            let tuple = stamp_lowering(ast, tuple, &ty);
            copy_location(ast, tuple, default);
            return tuple;
        }
        default
    }

    /// The missing value of one element of a tuple read, from a scalar default: the na of
    /// the element's type, as the implicit conversion makes it from a typeless `na` in a
    /// tuple-literal default. Any other element keeps the typeless `na`.
    fn element_default(&mut self, ast: &mut Ast, default: NodeId, ty: &str) -> NodeId {
        if is_numeric(ty) {
            let lib = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
            let nan = ast.add(Kind::Attribute {
                value: lib,
                attr: "_na_float".into(),
                ctx: ExprContext::Load,
            });
            copy_location(ast, nan, default);
            return set_ty(ast, nan, FLOAT);
        }
        if ty == STR {
            let empty = ast.add(Kind::Constant { value: Constant::Str(String::new()), kind: None });
            copy_location(ast, empty, default);
            return set_ty(ast, empty, STR);
        }
        let value = clone(ast, default, &mut HashMap::new());
        self.default(ast, value, ty)
    }
}

/// Whether a statement is `__security_contexts__ = ...`.
fn is_registry(ast: &Ast, stmt: NodeId) -> bool {
    let Kind::Assign { targets, .. } = ast.kind(stmt) else { return false };
    targets.iter().any(|&target| {
        matches!(ast.kind(target), Kind::Name { id, .. } if id == "__security_contexts__")
    })
}

/// A two-argument `__sec_read__(...)` call.
fn is_read(ast: &Ast, id: NodeId) -> bool {
    let Kind::Call { func, args, .. } = ast.kind(id) else { return false };
    args.len() == 2 && matches!(ast.kind(*func), Kind::Name { id, .. } if id == "__sec_read__")
}

/// Whether a value is a security read, or a call of a function giving one (`is_requested`):
/// such a tuple may be unpacked into fewer names than it gives, which take its leading values.
pub(crate) fn is_requested(ast: &Ast, value: Option<NodeId>, functions: &HashSet<String>) -> bool {
    let Some(value) = value else { return false };
    let Kind::Call { func, .. } = ast.kind(value) else { return false };
    let Kind::Name { id, .. } = ast.kind(*func) else { return false };
    if id == "__sec_read__" {
        return is_read(ast, value);
    }
    if id == "__ltf_unzip__" {
        let Kind::Call { args, .. } = ast.kind(value) else { unreachable!() };
        return args.len() == 2 && is_requested(ast, Some(args[0]), functions);
    }
    functions.contains(id)
}

/// Whether the module body imports `__ltf_unzip__` from `pynecore.core.security`.
fn imports_unzip(ast: &Ast, module: NodeId) -> bool {
    let Kind::Module { body, .. } = ast.kind(module) else { return false };
    body.iter().any(|&stmt| match ast.kind(stmt) {
        Kind::ImportFrom { module: Some(name), names, .. } if name == "pynecore.core.security" => {
            names.iter().any(|&alias| {
                matches!(ast.kind(alias), Kind::Alias { name, asname: None } if name == "__ltf_unzip__")
            })
        }
        _ => false,
    })
}

/// Whether a node is a call of the bare name `name`.
fn calls(ast: &Ast, id: NodeId, name: &str) -> bool {
    let Kind::Call { func, .. } = ast.kind(id) else { return false };
    matches!(ast.kind(*func), Kind::Name { id, .. } if id == name)
}

/// The width of the tuple a raw lower-timeframe read gives, if it is one (`_lower_tf_width`).
fn lower_tf_width(ast: &Ast, id: NodeId, widths: &HashMap<String, Option<usize>>) -> Option<usize> {
    if !is_read(ast, id) {
        return None;
    }
    let Kind::Call { args, .. } = ast.kind(id) else { unreachable!() };
    let Kind::Constant { value: Constant::Str(sec_id), .. } = ast.kind(args[0]) else {
        return None;
    };
    if !matches!(ast.kind(args[1]), Kind::List { elts, .. } if elts.is_empty()) {
        return None;
    }
    widths.get(sec_id).copied().flatten()
}

/// The raw lower-timeframe reads of a tuple, wrapped in `__ltf_unzip__` (`_unzip_lower_tf`).
struct Unzip<'w> {
    widths: &'w HashMap<String, Option<usize>>,
    wrapped: bool,
}

impl Transformer for Unzip<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if calls(ast, id, "__ltf_unzip__") {
            return Visited::Node(id);
        }
        if let Some(width) = lower_tf_width(ast, id, self.widths) {
            let func = ast.add(Kind::Name { id: "__ltf_unzip__".into(), ctx: ExprContext::Load });
            let width = ast
                .add(Kind::Constant { value: Constant::Int(Int::Small(width as i64)), kind: None });
            let unzip = ast.add(Kind::Call { func, args: vec![id, width], keywords: Vec::new() });
            copy_location(ast, unzip, id);
            self.wrapped = true;
            return Visited::Node(stamp_lowering(ast, unzip, OBJECT));
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }
}

/// Transpose the raw lower-timeframe reads of a tuple into one array per element
/// (`_unzip_lower_tf`): whether a read was wrapped.
fn unzip_lower_tf(ast: &mut Ast, module: NodeId) -> Result<bool, Error> {
    let mut widths: HashMap<String, Option<usize>> = HashMap::new();
    for sub in walk(ast, module) {
        if !calls(ast, sub, "__sec_write__") {
            continue;
        }
        let Kind::Call { args, .. } = ast.kind(sub) else { unreachable!() };
        if args.len() != 2 {
            continue;
        }
        let Kind::Constant { value: Constant::Str(sec_id), .. } = ast.kind(args[0]) else {
            continue;
        };
        let ty = ty_of(ast, args[1])?;
        let width = if is_tuple(&ty) { Some(elements_of(&ty).len()) } else { None };
        let entry = widths.entry(sec_id.clone()).or_insert(width);
        if *entry != width {
            *entry = None;
        }
    }
    let mut pass = Unzip { widths: &widths, wrapped: false };
    generic_visit(&mut pass, ast, module);
    Ok(pass.wrapped)
}

/// The width of the tuple a requested value gives, where its type does not tell (`_width`).
fn width_of(
    ast: &Ast,
    value: NodeId,
    returned: &HashMap<String, usize>,
) -> Result<Option<usize>, Error> {
    let Kind::Call { func, args, .. } = ast.kind(value) else { return Ok(None) };
    let Kind::Name { id, .. } = ast.kind(*func) else { return Ok(None) };
    if id == "__ltf_unzip__" {
        if args.len() != 2 {
            return Ok(None);
        }
        return Ok(match ast.kind(args[1]) {
            Kind::Constant { value: Constant::Int(Int::Small(width)), .. } => {
                usize::try_from(*width).ok()
            }
            _ => None,
        });
    }
    let ty = ty_of(ast, value)?;
    if is_tuple(&ty) {
        return Ok(Some(elements_of(&ty).len()));
    }
    Ok(returned.get(id).copied())
}

/// The width of the tuple each requested function's every `return` gives
/// (`_returned_widths`), for the functions whose returns agree on one.
fn returned_widths(
    ast: &Ast,
    module: NodeId,
    functions: &HashSet<String>,
) -> Result<HashMap<String, usize>, Error> {
    let mut returns: Vec<(String, Vec<Option<NodeId>>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for sub in walk(ast, module) {
        let Kind::FunctionDef { name, .. } = ast.kind(sub) else { continue };
        if !functions.contains(name) {
            continue;
        }
        let stop = |kind: &Kind| {
            matches!(
                kind,
                Kind::FunctionDef { .. }
                    | Kind::AsyncFunctionDef { .. }
                    | Kind::Lambda { .. }
                    | Kind::ClassDef { .. }
            )
        };
        let slot = *index.entry(name.clone()).or_insert_with(|| {
            returns.push((name.clone(), Vec::new()));
            returns.len() - 1
        });
        for node in descendants(ast, sub, &stop) {
            if let Kind::Return { value } = ast.kind(node) {
                returns[slot].1.push(*value);
            }
        }
    }
    let mut widths: HashMap<String, usize> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (name, values) in &returns {
            if widths.contains_key(name) {
                continue;
            }
            let mut found: HashSet<Option<usize>> = HashSet::new();
            for &value in values {
                found.insert(match value {
                    Some(value) => width_of(ast, value, &widths)?,
                    None => None,
                });
            }
            if found.len() == 1
                && let Some(&Some(width)) = found.iter().next()
            {
                widths.insert(name.clone(), width);
                changed = true;
            }
        }
    }
    Ok(widths)
}

/// The names of the functions whose every `return` gives a security read, directly or
/// through another such function (`requested_functions`). A name defined more than once
/// counts only when every definition qualifies.
pub(crate) fn requested_functions(ast: &Ast, module: NodeId) -> HashSet<String> {
    let mut returns: Vec<(String, Vec<Vec<Option<NodeId>>>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for sub in walk(ast, module) {
        let Kind::FunctionDef { name, .. } = ast.kind(sub) else { continue };
        let stop = |kind: &Kind| {
            matches!(
                kind,
                Kind::FunctionDef { .. }
                    | Kind::AsyncFunctionDef { .. }
                    | Kind::Lambda { .. }
                    | Kind::ClassDef { .. }
            )
        };
        let values: Vec<Option<NodeId>> = descendants(ast, sub, &stop)
            .into_iter()
            .filter_map(|node| match ast.kind(node) {
                Kind::Return { value } => Some(*value),
                _ => None,
            })
            .collect();
        let slot = *index.entry(name.clone()).or_insert_with(|| {
            returns.push((name.clone(), Vec::new()));
            returns.len() - 1
        });
        returns[slot].1.push(values);
    }
    let mut requested: HashSet<String> = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (name, definitions) in &returns {
            if requested.contains(name) {
                continue;
            }
            let qualifies = definitions.iter().all(|values| {
                !values.is_empty()
                    && values.iter().all(|&value| is_requested(ast, value, &requested))
            });
            if qualifies {
                requested.insert(name.clone());
                changed = true;
            }
        }
    }
    requested
}

/// Slice the requested tuple of an unpack that names fewer values than it gives
/// (`_take_leading`).
fn take_leading(ast: &mut Ast, module: NodeId) -> Result<(), Error> {
    let requested = requested_functions(ast, module);
    let returned = returned_widths(ast, module, &requested)?;
    for stmt in walk_statements(ast, module) {
        let Kind::Assign { targets, value, .. } = ast.kind(stmt) else { continue };
        if targets.len() != 1 {
            continue;
        }
        let (target, value) = (targets[0], *value);
        let names = match ast.kind(target) {
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                if elts.iter().any(|&elt| matches!(ast.kind(elt), Kind::Starred { .. })) {
                    continue;
                }
                elts.len()
            }
            _ => continue,
        };
        if !is_requested(ast, Some(value), &requested) {
            continue;
        }
        let mut ty = ty_of(ast, value)?;
        let width = if is_tuple(&ty) {
            let width = elements_of(&ty).len();
            ty = tuple_of(&elements_of(&ty)[..names.min(width)]);
            Some(width)
        } else {
            width_of(ast, value, &returned)?
        };
        if width.is_none_or(|width| width <= names) {
            continue;
        }
        let upper =
            ast.add(Kind::Constant { value: Constant::Int(Int::Small(names as i64)), kind: None });
        let slice = ast.add(Kind::Slice { lower: None, upper: Some(upper), step: None });
        let sliced = ast.add(Kind::Subscript { value, slice, ctx: ExprContext::Load });
        copy_location(ast, sliced, value);
        let sliced = stamp_lowering(ast, sliced, &ty);
        let Kind::Assign { value, .. } = ast.kind_mut(stmt) else { unreachable!() };
        *value = sliced;
    }
    Ok(())
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("SecurityDefault on a non-module".into()));
    };
    // Security reads are only ever emitted together with the module's registry
    if !body.iter().any(|&stmt| is_registry(ast, stmt)) {
        return Ok(module);
    }
    let mut pass = SecurityDefault { used: false };
    // Lazy like `ast_walk.walk`: a read's children are queued before it is rewritten
    let mut walker = Walk::new(module);
    while let Some(sub) = walker.next(ast) {
        if !is_read(ast, sub) {
            continue;
        }
        let ty = ty_of(ast, sub)?;
        let Kind::Call { args, .. } = ast.kind(sub) else { unreachable!() };
        let default = args[1];
        let resolved = pass.default(ast, default, &ty);
        let Kind::Call { args, .. } = ast.kind_mut(sub) else { unreachable!() };
        args[1] = resolved;
    }
    if unzip_lower_tf(ast, module)? && !imports_unzip(ast, module) {
        let alias = ast.add(Kind::Alias { name: "__ltf_unzip__".into(), asname: None });
        let import = ast.add(Kind::ImportFrom {
            module: Some("pynecore.core.security".into()),
            names: vec![alias],
            level: Some(0),
            is_lazy: None,
        });
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let at = future_end(ast, body);
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        body.insert(at, import);
    }
    take_leading(ast, module)?;
    if !pass.used {
        return Ok(module);
    }
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let mut insert_at = 0;
    if let Some(&first) = body.first()
        && let Kind::Expr { value } = ast.kind(first)
        && matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
    {
        insert_at = 1;
    }
    while insert_at < body.len() {
        match ast.kind(body[insert_at]) {
            Kind::ImportFrom { module: Some(name), .. } if name == "__future__" => insert_at += 1,
            _ => break,
        }
    }
    let alias =
        ast.add(Kind::Alias { name: "new_bool_na".into(), asname: Some(BOOL_NA_NAME.into()) });
    let import = ast.add(Kind::ImportFrom {
        module: Some("pynecore.types.na".into()),
        names: vec![alias],
        level: Some(0),
        is_lazy: None,
    });
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(insert_at, import);
    Ok(module)
}
