//! Port of `transformers/security_contexts.py`: reading the `__security_contexts__`
//! dict literal the security passes emit.

use pyne_ast::{Ast, Constant, Kind, NodeId};

/// The `(sid, context dict literal)` pairs of `__security_contexts__` (`iter_contexts`).
pub fn iter_contexts(ast: &Ast, contexts: NodeId) -> Vec<(String, NodeId)> {
    let Kind::Dict { keys, values } = ast.kind(contexts) else { return Vec::new() };
    let mut out = Vec::new();
    for (key, &value) in keys.iter().zip(values) {
        let Some(key) = key else { continue };
        if let Kind::Constant { value: Constant::Str(sid), .. } = ast.kind(*key)
            && matches!(ast.kind(value), Kind::Dict { .. })
        {
            out.push((sid.clone(), value));
        }
    }
    out
}

/// One entry of a context dict literal (`ctx_get`).
///
/// Python compares the key's value with `==`; a key of another type than `str`
/// never equals the name.
pub fn ctx_get(ast: &Ast, ctx: NodeId, name: &str) -> Option<NodeId> {
    let Kind::Dict { keys, values } = ast.kind(ctx) else { return None };
    for (key, &value) in keys.iter().zip(values) {
        let Some(key) = key else { continue };
        if matches!(ast.kind(*key), Kind::Constant { value: Constant::Str(text), .. } if text == name)
        {
            return Some(value);
        }
    }
    None
}
