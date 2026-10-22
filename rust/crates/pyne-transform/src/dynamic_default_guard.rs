//! Port of `transformers/dynamic_default_guard.py`: recognition of the
//! parameter-default guards the dynamic-default pass emits.

use pyne_ast::{Ast, CmpOperator, Kind, NodeId};

/// The name a dynamic parameter default is replaced with; its guard compares the
/// parameter against it
pub const SENTINEL_NAME: &str = "__dyn_default__";

/// Whether a statement is one of the dynamic-default pass's parameter-default
/// guards (`if <param> is __dyn_default__: ...`).
///
/// A later pass that prepends its own prologue has to land AFTER these: the guard
/// is what gives the parameter its value, so a statement placed above it reads the
/// sentinel instead.
pub fn is_dynamic_default_guard(ast: &Ast, stmt: NodeId) -> bool {
    let Kind::If { test, .. } = ast.kind(stmt) else { return false };
    let Kind::Compare { ops, comparators, .. } = ast.kind(*test) else { return false };
    if !matches!(ops[..], [CmpOperator::Is]) {
        return false;
    }
    let Some(&right) = comparators.first() else { return false };
    matches!(ast.kind(right), Kind::Name { id, .. } if id == SENTINEL_NAME)
}
