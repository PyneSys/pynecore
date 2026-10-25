//! Traversal and copying, the counterparts of `transformers/ast_walk.py`.
//!
//! Every function visits the same nodes in the same order as its Python
//! namesake, so a ported pass that walks makes its decisions in the same order.

use std::collections::{HashMap, VecDeque};

use crate::{Ast, CLASSES, FieldKind, FieldMut, FieldRef, Kind, Loc, Node, NodeId, Value};

/// Whether a field kind can hold a node: the fields a walk enters.
fn holds_nodes(kind: FieldKind) -> bool {
    matches!(kind, FieldKind::Node | FieldKind::OptNode | FieldKind::Nodes | FieldKind::OptNodes)
}

/// The fields that hold statement lists, or the handlers and cases holding statements.
const STATEMENT_FIELDS: &[&str] = &["body", "orelse", "finalbody", "handlers", "cases"];

/// Push the direct children of a node, in field order, as `ast.iter_child_nodes` yields them.
pub fn push_children(kind: &Kind, out: &mut impl Extend<NodeId>) {
    let class = kind.class();
    for (index, field) in class.fields.iter().enumerate() {
        if !holds_nodes(field.kind) {
            continue;
        }
        match kind.field(index) {
            FieldRef::Node(id) => out.extend([*id]),
            FieldRef::OptNode(id) => out.extend(id.iter().copied()),
            FieldRef::Nodes(ids) => out.extend(ids.iter().copied()),
            FieldRef::OptNodes(ids) => out.extend(ids.iter().flatten().copied()),
            _ => unreachable!(),
        }
    }
}

/// The direct child nodes, as `ast.iter_child_nodes` yields them.
pub fn children(ast: &Ast, id: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    push_children(ast.kind(id), &mut out);
    out
}

/// Every node of a tree, breadth first, as `ast.walk` yields them.
///
/// Lazy like the Python generator: a node's children are read when the node is
/// handed out, so editing a yielded node does not change what is walked, while
/// editing a node not yet reached does.
pub struct Walk {
    todo: VecDeque<NodeId>,
}

impl Walk {
    pub fn new(root: NodeId) -> Walk {
        Walk { todo: VecDeque::from([root]) }
    }

    pub fn next(&mut self, ast: &Ast) -> Option<NodeId> {
        let id = self.todo.pop_front()?;
        push_children(ast.kind(id), &mut self.todo);
        Some(id)
    }
}

/// Every node of a tree, breadth first, collected at once.
pub fn walk(ast: &Ast, root: NodeId) -> Vec<NodeId> {
    let mut walker = Walk::new(root);
    let mut out = Vec::new();
    while let Some(id) = walker.next(ast) {
        out.push(id);
    }
    out
}

fn push_child_statements(kind: &Kind, out: &mut impl Extend<NodeId>) {
    let class = kind.class();
    for (index, field) in class.fields.iter().enumerate() {
        if !STATEMENT_FIELDS.contains(&field.name) {
            continue;
        }
        // ``Lambda`` and ``IfExp`` call an expression their ``body``
        if let FieldRef::Nodes(ids) = kind.field(index) {
            out.extend(ids.iter().copied());
        }
    }
}

/// The direct children that are or hold statements, in field order.
pub fn child_statements(ast: &Ast, id: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    push_child_statements(ast.kind(id), &mut out);
    out
}

/// The node, then every statement, handler and case under it, in `ast.walk` order.
pub fn walk_statements(ast: &Ast, root: NodeId) -> Vec<NodeId> {
    let mut todo = VecDeque::from([root]);
    let mut out = Vec::new();
    while let Some(id) = todo.pop_front() {
        push_child_statements(ast.kind(id), &mut todo);
        out.push(id);
    }
    out
}

/// Every descendant in pre-order, not yielding or entering nodes `stop` accepts.
pub fn descendants(ast: &Ast, root: NodeId, stop: &dyn Fn(&Kind) -> bool) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut stack: Vec<std::vec::IntoIter<NodeId>> = vec![children(ast, root).into_iter()];
    while let Some(top) = stack.last_mut() {
        match top.next() {
            Some(child) => {
                if stop(ast.kind(child)) {
                    continue;
                }
                out.push(child);
                stack.push(children(ast, child).into_iter());
            }
            None => {
                stack.pop();
            }
        }
    }
    out
}

/// What a transformer's visit hands back for a node: the stdlib's three answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Visited {
    /// The node that takes the visited node's place (the node itself when unchanged)
    Node(NodeId),
    /// `None`: the node is removed
    Remove,
    /// A list of nodes spliced into a list field in its place
    Many(Vec<NodeId>),
}

/// `ast.NodeTransformer`: `visit` dispatches, [`generic_visit`] walks the children
/// and writes back what the visits answered.
pub trait Transformer {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        generic_visit(self, ast, id);
        Visited::Node(id)
    }
}

/// Visit every child of a node and write the answers back, as the stdlib does.
///
/// A list field is rebuilt from the answers (a removed node dropped, a list
/// spliced in); a node field is replaced, or cleared when it is optional and the
/// answer was a removal.
pub fn generic_visit<T: Transformer + ?Sized>(t: &mut T, ast: &mut Ast, id: NodeId) {
    let count = ast.kind(id).class().fields.len();
    for index in 0..count {
        visit_field(t, ast, id, index);
    }
}

/// Visit one field of a node and write the result back, as `generic_visit` does.
pub fn visit_field<T: Transformer + ?Sized>(t: &mut T, ast: &mut Ast, id: NodeId, index: usize) {
    match ast.kind(id).field(index) {
        FieldRef::Node(child) => {
            let child = *child;
            let answer = t.visit(ast, child);
            match (answer, ast.kind_mut(id).field_mut(index)) {
                (Visited::Node(new), FieldMut::Node(slot)) => *slot = new,
                (answer, _) => panic!("{:?} for a required field of {}", answer, ast.name(id)),
            }
        }
        FieldRef::OptNode(Some(child)) => {
            let child = *child;
            let answer = t.visit(ast, child);
            let FieldMut::OptNode(slot) = ast.kind_mut(id).field_mut(index) else { unreachable!() };
            match answer {
                Visited::Node(new) => *slot = Some(new),
                Visited::Remove => *slot = None,
                Visited::Many(_) => panic!("a list for a node field"),
            }
        }
        FieldRef::Nodes(items) => {
            let items = items.clone();
            let mut new_items = Vec::with_capacity(items.len());
            for child in items {
                match t.visit(ast, child) {
                    Visited::Node(new) => new_items.push(new),
                    Visited::Remove => {}
                    Visited::Many(many) => new_items.extend(many),
                }
            }
            let FieldMut::Nodes(slot) = ast.kind_mut(id).field_mut(index) else { unreachable!() };
            *slot = new_items;
        }
        FieldRef::OptNodes(items) => {
            let items = items.clone();
            let mut new_items = Vec::with_capacity(items.len());
            for child in items {
                let Some(child) = child else {
                    new_items.push(None);
                    continue;
                };
                match t.visit(ast, child) {
                    Visited::Node(new) => new_items.push(Some(new)),
                    Visited::Remove => {}
                    Visited::Many(many) => new_items.extend(many.into_iter().map(Some)),
                }
            }
            let FieldMut::OptNodes(slot) = ast.kind_mut(id).field_mut(index) else {
                unreachable!()
            };
            *slot = new_items;
        }
        _ => {}
    }
}

/// `ast.NodeVisitor`: a read-only walk.
pub trait Visitor {
    fn visit(&mut self, ast: &Ast, id: NodeId) {
        generic_visit_ro(self, ast, id);
    }
}

pub fn generic_visit_ro<T: Visitor + ?Sized>(t: &mut T, ast: &Ast, id: NodeId) {
    for child in children(ast, id) {
        t.visit(ast, child);
    }
}

/// `ast.fix_missing_locations`: same nodes filled with the same values, in pre-order.
pub fn fix_missing_locations(ast: &mut Ast, root: NodeId) {
    let mut stack = vec![(root, 1i64, 0i64, 1i64, 0i64)];
    while let Some((id, mut lineno, mut col, mut end_lineno, mut end_col)) = stack.pop() {
        if ast.kind(id).class().located {
            let loc = &mut ast[id].loc;
            match loc.lineno {
                None => loc.lineno = Some(lineno),
                Some(value) => lineno = value,
            }
            match loc.end_lineno {
                None => loc.end_lineno = Some(end_lineno),
                Some(value) => end_lineno = value,
            }
            match loc.col_offset {
                None => loc.col_offset = Some(col),
                Some(value) => col = value,
            }
            match loc.end_col_offset {
                None => loc.end_col_offset = Some(end_col),
                Some(value) => end_col = value,
            }
        }
        for child in children(ast, id).into_iter().rev() {
            stack.push((child, lineno, col, end_lineno, end_col));
        }
    }
}

/// Index of the first statement after a leading docstring of `body` (`docstring_end`).
pub fn docstring_end(ast: &Ast, body: &[NodeId]) -> usize {
    usize::from(body.first().is_some_and(|&first| {
        matches!(ast.kind(first), Kind::Expr { value }
            if matches!(ast.kind(*value), Kind::Constant { value: crate::Constant::Str(_), .. }))
    }))
}

/// Index of the first statement after the docstring and the `__future__` block
/// (`future_end`).
pub fn future_end(ast: &Ast, body: &[NodeId]) -> usize {
    let mut index = docstring_end(ast, body);
    while index < body.len()
        && matches!(ast.kind(body[index]),
            Kind::ImportFrom { module: Some(name), .. } if name == "__future__")
    {
        index += 1;
    }
    index
}

/// `copy.deepcopy` of a subtree: a node referenced twice within one copy (one
/// `memo`) is copied once, and the stamped attributes are copied with the node,
/// including the nodes they reference.
pub fn clone(ast: &mut Ast, id: NodeId, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&copy) = memo.get(&id) {
        return copy;
    }
    let placeholder = ast.add(Kind::Pass);
    memo.insert(id, placeholder);
    let Node { mut kind, loc, extra } = ast[id].clone();
    let class = kind.class();
    for (index, field) in class.fields.iter().enumerate() {
        if !holds_nodes(field.kind) {
            continue;
        }
        match kind.field_mut(index) {
            FieldMut::Node(child) => *child = clone(ast, *child, memo),
            FieldMut::OptNode(Some(child)) => *child = clone(ast, *child, memo),
            FieldMut::Nodes(items) => {
                for child in items.iter_mut() {
                    *child = clone(ast, *child, memo);
                }
            }
            FieldMut::OptNodes(items) => {
                for child in items.iter_mut().flatten() {
                    *child = clone(ast, *child, memo);
                }
            }
            _ => {}
        }
    }
    let extra =
        extra.into_iter().map(|(name, value)| (name, clone_value(ast, value, memo))).collect();
    ast[placeholder] = Node { kind, loc, extra };
    placeholder
}

/// A stamped value of a copied node. A [`Value::Opaque`] or [`Value::Native`]
/// stays the same object where `copy.deepcopy` would copy it: the native side
/// cannot copy what it cannot see, and the copy is equal to the original, which
/// is what a consumer of the tree compares.
fn clone_value(ast: &mut Ast, value: Value, memo: &mut HashMap<NodeId, NodeId>) -> Value {
    match value {
        Value::Node(id) => Value::Node(clone(ast, id, memo)),
        Value::Tuple(items) => {
            Value::Tuple(items.into_iter().map(|v| clone_value(ast, v, memo)).collect())
        }
        Value::List(items) => {
            Value::List(items.into_iter().map(|v| clone_value(ast, v, memo)).collect())
        }
        Value::Dict(items) => Value::Dict(
            items
                .into_iter()
                .map(|(k, v)| (clone_value(ast, k, memo), clone_value(ast, v, memo)))
                .collect(),
        ),
        other => other,
    }
}

/// Copy only the location attributes of `src` onto `dst` (`ast.copy_location`).
pub fn copy_location(ast: &mut Ast, dst: NodeId, src: NodeId) {
    let from: Loc = ast[src].loc;
    let class = ast.kind(dst).class();
    if !class.located || !ast.kind(src).class().located {
        return;
    }
    let to = &mut ast[dst].loc;
    if from.lineno.is_some() {
        to.lineno = from.lineno;
    }
    if from.col_offset.is_some() {
        to.col_offset = from.col_offset;
    }
    // An end attribute is copied None included, and on 3.13+ every node has one
    // (the class default)
    to.end_lineno = from.end_lineno;
    to.end_col_offset = from.end_col_offset;
}

/// Every class in [`CLASSES`] whose name is `name`; for debugging and dumps.
pub fn class_named(name: &str) -> Option<&'static crate::ClassInfo> {
    CLASSES.iter().find(|class| class.name == name)
}
