//! Port of `transformers/security_calls.py`: `has_security_call`,
//! `is_security_call` and `is_security_lower_tf_call`.

use pyne_ast::walk::Walk;
use pyne_ast::{Ast, Kind, NodeId};

/// Attribute names of the `lib.request` functions the security passes lower
/// (`_REQUEST_FUNCS`).
const REQUEST_FUNCS: [&str; 2] = ["security", "security_lower_tf"];

/// Whether a callee is `lib.request.<attr>`.
fn is_request_attr(ast: &Ast, func: NodeId, attr: &str) -> bool {
    let Kind::Attribute { value, attr: name, .. } = ast.kind(func) else { return false };
    name == attr && is_lib_request(ast, *value)
}

/// Whether a node is the `lib.request` owner of a request function.
fn is_lib_request(ast: &Ast, owner: NodeId) -> bool {
    let Kind::Attribute { value, attr, .. } = ast.kind(owner) else { return false };
    attr == "request" && matches!(ast.kind(*value), Kind::Name { id, .. } if id == "lib")
}

/// Whether a call is `lib.request.security(...)`; false for a node that is no call.
pub fn is_security_call(ast: &Ast, call: NodeId) -> bool {
    matches!(ast.kind(call), Kind::Call { func, .. } if is_request_attr(ast, *func, "security"))
}

/// Whether a call is `lib.request.security_lower_tf(...)`; false for a node that is no
/// call.
pub fn is_security_lower_tf_call(ast: &Ast, call: NodeId) -> bool {
    matches!(
        ast.kind(call),
        Kind::Call { func, .. } if is_request_attr(ast, *func, "security_lower_tf")
    )
}

/// Whether a tree holds a `lib.request.security[_lower_tf](...)` call
/// (`has_security_call`). Every rewrite of the security passes starts at such a
/// call, and most scripts have none.
pub fn has_security_call(ast: &Ast, root: NodeId) -> bool {
    let mut walker = Walk::new(root);
    while let Some(node) = walker.next(ast) {
        let Kind::Call { func, .. } = ast.kind(node) else { continue };
        let Kind::Attribute { value, attr, .. } = ast.kind(*func) else { continue };
        if REQUEST_FUNCS.contains(&attr.as_str()) && is_lib_request(ast, *value) {
            return true;
        }
    }
    false
}
