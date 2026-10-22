//! Stable per-node identifiers (`node_ids.py`), native.
//!
//! The numbering is the Python one: a pre-order walk over every `ast.AST` object
//! of the tree, the fieldless ones (an expression context, an operator) included.
//! Those are no nodes of the arena but fields of their parent, and Python shares
//! one instance of each between every parent, so an id stamped on one means
//! nothing; they take their id from the count all the same, so every real node
//! gets the id Python gives it.

use pyne_ast::{Ast, CLASSES, FieldRef, Int, NodeId, Value};

/// The attribute the id is stamped under (`ID_ATTR`).
pub const ID_ATTR: &str = "_pine_nid";

/// What the walk visits: a node of the arena, or a fieldless object of a parent.
enum Item {
    Node(NodeId),
    Fieldless,
}

/// Number every node of a tree in pre-order (`assign_node_ids`).
///
/// A node reached twice (shared by two parents) is numbered twice and keeps the
/// later id, as the Python walk overwrites it.
///
/// :return: The next unused id.
pub fn assign_node_ids(ast: &mut Ast, tree: NodeId, start: i64) -> i64 {
    let mut nid = start;
    let mut stack = vec![Item::Node(tree)];
    while let Some(item) = stack.pop() {
        let id = match item {
            Item::Node(id) => id,
            Item::Fieldless => {
                nid += 1;
                continue;
            }
        };
        ast.set_extra(id, ID_ATTR, Value::Int(Int::Small(nid)));
        nid += 1;
        // The children of `ast.iter_child_nodes`, pushed back to front so they
        // come off the stack in source order
        let kind = ast.kind(id);
        for index in (0..CLASSES[kind.class_index()].fields.len()).rev() {
            match kind.field(index) {
                FieldRef::Node(child) => stack.push(Item::Node(*child)),
                FieldRef::OptNode(child) => stack.extend(child.map(Item::Node)),
                FieldRef::Nodes(children) => {
                    stack.extend(children.iter().rev().map(|child| Item::Node(*child)))
                }
                FieldRef::OptNodes(children) => {
                    stack.extend(children.iter().rev().flatten().map(|child| Item::Node(*child)))
                }
                FieldRef::ExprContext(_)
                | FieldRef::Operator(_)
                | FieldRef::UnaryOperator(_)
                | FieldRef::BoolOperator(_) => stack.push(Item::Fieldless),
                FieldRef::CmpOperators(ops) => stack.extend(ops.iter().map(|_| Item::Fieldless)),
                FieldRef::Str(_)
                | FieldRef::OptStr(_)
                | FieldRef::Strs(_)
                | FieldRef::Int(_)
                | FieldRef::OptInt(_)
                | FieldRef::Constant(_) => {}
            }
        }
    }
    nid
}

/// A node's id, `None` when the tree was never numbered (`node_id`).
pub fn node_id(ast: &Ast, node: NodeId) -> Option<i64> {
    match ast.extra(node, ID_ATTR)? {
        Value::Int(Int::Small(nid)) => Some(*nid),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fieldless_objects_take_an_id_from_the_count() {
        // Module, Expr, BinOp, Name, Load, Add, Constant
        let (mut ast, module) = pyne_parse::parse_module("a + 1\n").expect("parses");
        assert_eq!(assign_node_ids(&mut ast, module, 10), 17);
        assert_eq!(node_id(&ast, module), Some(10));
        let pyne_ast::Kind::Module { body, .. } = ast.kind(module) else { panic!() };
        let statement = body[0];
        let pyne_ast::Kind::Expr { value } = *ast.kind(statement) else { panic!() };
        let pyne_ast::Kind::BinOp { left, right, .. } = *ast.kind(value) else { panic!() };
        assert_eq!(node_id(&ast, statement), Some(11));
        assert_eq!(node_id(&ast, value), Some(12));
        assert_eq!(node_id(&ast, left), Some(13));
        // Load is 14, Add is 15
        assert_eq!(node_id(&ast, right), Some(16));
    }
}
