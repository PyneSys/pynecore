//! The result of the Pine type inference (`pine_type_table.py`), native.
//!
//! The inference stamps each expression's type on the node itself; what lives
//! here is the derived index a consumer wants: the per-scope bindings, the
//! function signatures, the per-call-site contexts a generic function was
//! analysed in, the call sites with their overload pin, and the diagnostics.
//!
//! Every type mirrors the Python dataclass of the same name field for field, in
//! the same order and with the same defaults, so the bridge converts a table
//! both ways without losing anything: a Python `dict` is an [`IndexMap`] (its
//! insertion order is part of the table), a `list` or `tuple` a `Vec`, a
//! `frozenset` a [`BTreeSet`], an `X | None` an `Option`. A node id (`_pine_nid`,
//! see [`crate::node_ids`]) is a number, as in Python, never a [`pyne_ast::NodeId`]:
//! the table describes the tree the inference walked, not the arena it lives in.
//! A Pine qualifier is a `u8` (`pine_qualifier`).

use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::pine_qualifier::{NOT_READ, SERIES};
use crate::pine_type_rules::{ImplSig, UNKNOWN};

/// The attribute the module node carries its table under (`TABLE_ATTR`).
pub const TABLE_ATTR: &str = "_pine_types";

/// Separator the scope-qualified ids are spelled with.
pub const SCOPE_SEP: &str = "·";

/// The scope-qualified identity of a name declared in one scope (`qualify`).
pub fn qualify(scope: &str, name: &str) -> String {
    if scope.is_empty() { name.to_string() } else { format!("{scope}{SCOPE_SEP}{name}") }
}

/// Why a type could not be determined, and where it was lost.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Unknown {
    /// Short machine-readable cause, e.g. `unannotated-param`
    pub reason: String,
    /// 1-based line the type was lost on
    pub line: i64,
    /// 0-based column offset
    pub col: i64,
    /// Human-readable detail, e.g. the offending name
    pub detail: String,
}

/// A name in a scope, with the type its assignments join to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub ty: String,
    /// 1-based line of the binding that first established the type
    pub line: i64,
    /// Provenance when `ty` is UNKNOWN
    pub unknown: Option<Unknown>,
    /// Declared as a series: its history `x[n]` is readable
    pub series: bool,
    /// Pine qualifier: the strongest over every assignment and the conditions of
    /// every reassignment
    pub qual: u8,
    /// How many conditions enclosed the first binding
    pub depth: i64,
    /// Declared `Persistent` (Pine's `var`)
    pub persistent: bool,
    /// Node id of the statement that declared it
    pub decl: Option<i64>,
    /// Lowest qualifier a read of it has seen in the running walk, NOT_READ when
    /// none has read it
    pub min_read: u8,
}

impl Binding {
    /// A binding with the dataclass defaults.
    pub fn new(name: impl Into<String>) -> Binding {
        Binding {
            name: name.into(),
            ty: UNKNOWN.to_string(),
            line: 0,
            unknown: None,
            series: false,
            qual: SERIES,
            depth: 0,
            persistent: false,
            decl: None,
            min_read: NOT_READ,
        }
    }
}

/// A user function's inferred signature: the join over every context it was
/// analysed in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncSig {
    pub name: String,
    pub params: Vec<String>,
    pub ret: String,
    pub line: i64,
}

impl FuncSig {
    /// A signature with the dataclass defaults.
    pub fn new(name: impl Into<String>) -> FuncSig {
        FuncSig { name: name.into(), params: Vec::new(), ret: UNKNOWN.to_string(), line: 0 }
    }
}

/// How a context is addressed (`ContextKey`, a tuple in Python).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContextKey {
    /// The callee's scope id
    pub key: String,
    /// The parameter types it was instantiated with
    pub params: Vec<String>,
    /// The context its caller was running in
    pub origin: i64,
    /// Node id of the definition analysed
    pub def: Option<i64>,
    /// (name, type, qualifier) the enclosing scopes held for the names its body reads
    pub env: Vec<(String, String, u8)>,
    /// The qualifier of each parameter
    pub quals: Vec<u8>,
}

/// One analysis of one function body under one parameter-type tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextResult {
    /// Identity of this context within the module; 0 is the module body itself
    pub cid: i64,
    /// Scope-qualified id of the function analysed
    pub key: String,
    /// Type of each parameter, positional first then keyword-only
    pub params: Vec<String>,
    pub ret: String,
    /// Pine qualifier of each parameter, in the order of `params`
    pub quals: Vec<u8>,
    /// Pine qualifier of the value the function returns
    pub ret_qual: u8,
    /// Call node id -> the overload pin this context justified there
    pub pins: IndexMap<i64, Option<String>>,
}

impl ContextResult {
    /// A context with the dataclass defaults.
    pub fn new(cid: i64, key: impl Into<String>, params: Vec<String>) -> ContextResult {
        ContextResult {
            cid,
            key: key.into(),
            params,
            ret: UNKNOWN.to_string(),
            quals: Vec::new(),
            ret_qual: SERIES,
            pins: IndexMap::new(),
        }
    }
}

/// One call, with the overload pin the inference decided on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSite {
    /// Dotted callee as written, e.g. `ta.highest`
    pub callee: String,
    pub line: i64,
    pub col: i64,
    /// Arguments passed, keyword ones included; None when an unpacking hides the count
    pub argc: Option<i64>,
    pub ty: String,
    pub pin: Option<String>,
}

/// One typing complaint, in the shape both modes report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diag {
    pub message: String,
    pub line: i64,
    pub col: i64,
    /// Where the type was lost, when that differs from where it was noticed
    pub origin: Option<Unknown>,
    /// Concrete remedy
    pub fix: String,
    /// End of the construct the complaint is about; 0 = a point
    pub end_line: i64,
    pub end_col: i64,
}

/// One module another module's types were derived from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepRecord {
    /// Resolved source path of the dependency
    pub path: String,
    pub mtime_ns: i64,
    pub size: i64,
    /// The dependency's interface digest, or `module_interface.SOURCE_DIGEST`
    pub digest: String,
    /// The dependency's routes digest, `""` when it was never settled
    pub routes: String,
}

/// One name a module publishes, as a consumer of the module reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportSig {
    pub name: String,
    /// `function` for a plain definition, `group` for an `@overload` group
    pub kind: String,
    /// Type of each positional parameter, in declaration order
    pub params: Vec<String>,
    /// How many of them a call has to pass
    pub required: i64,
    /// Whether a `*args` lets it take any number of them
    pub open_ended: bool,
    pub ret: String,
    /// Whether every positional parameter carries an annotation the pass can read
    pub annotated: bool,
    /// Every implementation of a group, in declaration order; empty otherwise
    pub impls: Vec<ImplSig>,
    pub line: i64,
    /// Name of each positional parameter
    pub names: Vec<String>,
}

/// One user-defined type, as everything that reads a value of it needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassSig {
    /// The class name, as an annotation of its own module spells it
    pub name: String,
    /// `<module key>#<name>`, the identity a shaped type carries
    pub id: String,
    /// Field name -> its declared type, in declaration order
    pub fields: IndexMap<String, String>,
    /// How many of them a constructor call has to pass
    pub required: i64,
    /// Method name -> its signature
    pub methods: IndexMap<String, ExportSig>,
}

/// Everything the inference learned about one module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PineTypeTable {
    /// Absolute source path of the module
    pub module_path: String,
    /// scope id -> name -> binding; the module scope is the empty string
    pub bindings: IndexMap<String, IndexMap<String, Binding>>,
    /// scope-qualified function id -> signature
    pub funcs: IndexMap<String, FuncSig>,
    /// What each context found
    pub contexts: IndexMap<ContextKey, ContextResult>,
    pub calls: Vec<CallSite>,
    pub diags: Vec<Diag>,
    /// Scope-qualified id of every `@overload` group -> its implementations
    pub groups: IndexMap<String, Vec<ImplSig>>,
    /// Resolved path -> the state of every module this one's types were derived from
    pub deps: IndexMap<String, DepRecord>,
    /// Every class name an annotation in this module may name -> its class id
    pub classes: IndexMap<String, String>,
    /// The classes this module declares, by id
    pub class_sigs: IndexMap<String, ClassSig>,
    /// The methods this module declares on classes it does not declare, by class
    /// id and method name
    pub extensions: IndexMap<String, IndexMap<String, ExportSig>>,
    /// Why no call site of the module carries an overload pin, when none does
    pub pins_suppressed: Option<Diag>,
    /// Module-scope names an importer is guaranteed to receive a definition under
    pub exportable: BTreeSet<String>,
    /// Call node id -> the source position of that call
    pub call_pos: IndexMap<i64, (i64, i64)>,
}

impl PineTypeTable {
    /// Look one name up in one scope, without walking outwards (`binding`).
    pub fn binding(&self, scope: &str, name: &str) -> Option<&Binding> {
        self.bindings.get(scope)?.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualify_joins_with_the_scope_separator() {
        assert_eq!(qualify("", "x"), "x");
        assert_eq!(qualify("main", "x"), "main·x");
    }

    #[test]
    fn a_binding_is_found_in_its_own_scope_only() {
        let mut table = PineTypeTable::default();
        table.bindings.entry("main".to_string()).or_default().insert("x".into(), Binding::new("x"));
        assert_eq!(table.binding("main", "x").map(|b| b.qual), Some(SERIES));
        assert_eq!(table.binding("", "x"), None);
        assert_eq!(table.binding("main", "y"), None);
    }
}
