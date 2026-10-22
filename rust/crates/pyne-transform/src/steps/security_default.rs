//! Port of `transformers/security_default.py`: the missing value of a `request.security`
//! read whose inferred type is bool, or a tuple holding bools, becomes a call of the
//! runtime-mode bool factory, imported once at the top of the module.

use pyne_ast::walk::{Walk, copy_location};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, Value};

use crate::pine_type_rules::{BOOL, TY_ATTR, UNKNOWN, elements_of, is_tuple, stamp_lowering};
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
    /// The default with its bool elements resolved under the runtime mode.
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
            let types: Vec<String> = elements_of(ty).into_iter().map(str::to_string).collect();
            let defaults = match ast.kind(default) {
                Kind::Tuple { elts, .. } if elts.len() == types.len() => elts.clone(),
                _ => vec![default; types.len()],
            };
            let values: Vec<NodeId> = defaults
                .iter()
                .zip(&types)
                .map(|(&value, item)| self.default(ast, value, item))
                .collect();
            if values.iter().zip(&defaults).any(|(value, old)| value != old) {
                let tuple = ast.add(Kind::Tuple { elts: values, ctx: ExprContext::Load });
                let tuple = stamp_lowering(ast, tuple, ty);
                copy_location(ast, tuple, default);
                return tuple;
            }
        }
        default
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
    });
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(insert_at, import);
    Ok(module)
}
