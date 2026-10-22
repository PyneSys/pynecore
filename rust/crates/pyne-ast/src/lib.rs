//! The CPython `ast` tree, as an arena.
//!
//! A node is addressed by a [`NodeId`], the way a Python pass holds a reference to
//! an `ast.AST` object: a pass may keep ids in sets and maps, hand the same node
//! to two parents, mutate a node it reached through another one, or drop a
//! subtree, and the arena behaves as the Python object graph does. A dropped
//! node simply stays in the arena unreferenced.
//!
//! The node classes, their fields and the field order are generated from the
//! CPython `ast` module (see `gen/gen_nodes.py`), so a port of a Python pass can
//! read the same fields in the same order. What a pass stamps on a node besides
//! its fields (`_pine_ty`, `pine_exact`, ...) lives in the node's
//! [`Node::extra`] map under the same attribute name; an object that is no plain
//! value (the type pass's `PineTypeTable`) is either a host object handed through
//! ([`Value::Opaque`]) or an object the tree owns natively ([`Value::Native`]).

#[rustfmt::skip]
mod nodes;
pub mod dump;
pub mod locations;
pub mod walk;

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Index, IndexMut};

pub use nodes::*;

/// A reference to a node of an [`Ast`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Stands in a required node field until it is filled.
    pub const PLACEHOLDER: NodeId = NodeId(u32::MAX);
}

/// A Python `int`: machine-sized when it fits, its decimal spelling otherwise.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Int {
    Small(i64),
    /// Decimal digits with an optional leading `-`
    Big(String),
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int::Small(value) => write!(f, "{value}"),
            Int::Big(digits) => f.write_str(digits),
        }
    }
}

/// The value of an `ast.Constant` (or `ast.MatchSingleton`).
#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    None,
    Bool(bool),
    Int(Int),
    Float(f64),
    Complex { real: f64, imag: f64 },
    Str(String),
    Bytes(Vec<u8>),
    Ellipsis,
    Tuple(Vec<Constant>),
    FrozenSet(Vec<Constant>),
}

/// A value a pass stamped on a node as an attribute of its own.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(Int),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    /// Entries in insertion order, as a Python `dict` keeps them
    Dict(Vec<(Value, Value)>),
    /// A reference to another node of the same tree
    Node(NodeId),
    /// An object of the host that no native step looks into (the Python type
    /// pass's `PineTypeTable`): an index into the object list of the bridge that
    /// converted the tree, which hands the very same object back. Only that
    /// bridge makes one; a step may move, share or drop it, never build one.
    /// Two of them are equal when they hold the same object.
    Opaque(u32),
    /// An object the tree owns natively (a `PineTypeTable` the native type pass
    /// built, or one the bridge converted for a native step that reads it): an
    /// index into the tree's native objects ([`Ast::add_native`]). A copied node
    /// shares it, as a Python pass sharing the object would. Two of them are
    /// equal when they hold the same object.
    Native(u32),
}

/// What [`Value::Native`] holds: any cloneable value, reached back through
/// [`Ast::native`] by its concrete type.
pub trait NativeObject: Any + fmt::Debug {
    fn clone_object(&self) -> Box<dyn NativeObject>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any + Clone + fmt::Debug> NativeObject for T {
    fn clone_object(&self) -> Box<dyn NativeObject> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// One native object of an [`Ast`]; cloning the tree clones it.
#[derive(Debug)]
struct NativeSlot(Box<dyn NativeObject>);

impl Clone for NativeSlot {
    fn clone(&self) -> NativeSlot {
        NativeSlot(self.0.clone_object())
    }
}

/// Source location attributes; `None` is an attribute the node does not carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Loc {
    pub lineno: Option<i64>,
    pub col_offset: Option<i64>,
    pub end_lineno: Option<i64>,
    pub end_col_offset: Option<i64>,
}

impl Loc {
    pub fn new(lineno: i64, col_offset: i64, end_lineno: i64, end_col_offset: i64) -> Loc {
        Loc {
            lineno: Some(lineno),
            col_offset: Some(col_offset),
            end_lineno: Some(end_lineno),
            end_col_offset: Some(end_col_offset),
        }
    }
}

/// One node: its class and fields, its location and the attributes passes stamped on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: Kind,
    pub loc: Loc,
    /// Attributes beyond the fields and the location, by Python attribute name
    pub extra: BTreeMap<String, Value>,
}

/// The arena all nodes of one transform live in.
#[derive(Clone, Debug, Default)]
pub struct Ast {
    nodes: Vec<Node>,
    /// What the [`Value::Native`] values of the nodes index
    natives: Vec<NativeSlot>,
}

impl Ast {
    pub fn new() -> Ast {
        Ast::default()
    }

    /// Add a node without location attributes.
    pub fn add(&mut self, kind: Kind) -> NodeId {
        self.add_at(kind, Loc::default())
    }

    pub fn add_at(&mut self, kind: Kind, loc: Loc) -> NodeId {
        let id = NodeId(u32::try_from(self.nodes.len()).expect("arena overflow"));
        self.nodes.push(Node { kind, loc, extra: BTreeMap::new() });
        id
    }

    /// Number of nodes ever added (reachable or not).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn kind(&self, id: NodeId) -> &Kind {
        &self[id].kind
    }

    pub fn kind_mut(&mut self, id: NodeId) -> &mut Kind {
        &mut self[id].kind
    }

    /// A stamped attribute, `None` when the node does not carry it.
    pub fn extra(&self, id: NodeId, name: &str) -> Option<&Value> {
        self[id].extra.get(name)
    }

    pub fn set_extra(&mut self, id: NodeId, name: &str, value: Value) {
        self[id].extra.insert(name.to_string(), value);
    }

    /// The class name of a node.
    pub fn name(&self, id: NodeId) -> &'static str {
        self[id].kind.name()
    }

    /// Hand an object to the tree; the value to stamp it with.
    pub fn add_native<T: NativeObject>(&mut self, object: T) -> Value {
        let index = u32::try_from(self.natives.len()).expect("native object overflow");
        self.natives.push(NativeSlot(Box::new(object)));
        Value::Native(index)
    }

    /// The object a [`Value::Native`] holds, whatever its type.
    pub fn native_object(&self, index: u32) -> Option<&dyn NativeObject> {
        self.natives.get(index as usize).map(|slot| &*slot.0)
    }

    /// The object a value holds, when it is a [`Value::Native`] of type `T`.
    pub fn native<T: Any>(&self, value: &Value) -> Option<&T> {
        let Value::Native(index) = value else { return None };
        self.native_object(*index)?.as_any().downcast_ref()
    }

    /// The object a value holds, mutably, when it is a [`Value::Native`] of type `T`.
    pub fn native_mut<T: Any>(&mut self, value: &Value) -> Option<&mut T> {
        let Value::Native(index) = value else { return None };
        self.natives.get_mut(*index as usize)?.0.as_any_mut().downcast_mut()
    }
}

impl Index<NodeId> for Ast {
    type Output = Node;

    fn index(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }
}

impl IndexMut<NodeId> for Ast {
    fn index_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0 as usize]
    }
}

/// A construct a native step does not handle: the host runs the Python step instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}
