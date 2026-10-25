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
    /// Method name -> its signature, for the methods declared in the class body,
    /// receiver first
    pub members: IndexMap<String, ExportSig>,
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

/// What the inference reads of the interface an imported module publishes
/// (`ModuleInterface`), with the dependency record a dependent bakes for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInterface {
    /// Resolved source path of the module
    pub path: String,
    pub exports: IndexMap<String, ExportSig>,
    /// The module's literal `__all__`, `None` when it spells none
    pub all: Option<Vec<String>>,
    /// The classes the module publishes, by name
    pub classes: IndexMap<String, ClassSig>,
    /// The methods it declares on another module's class, by class id and name
    pub extensions: IndexMap<String, IndexMap<String, ExportSig>>,
    /// Every module this one's types were derived from or that it imports
    pub deps: IndexMap<String, DepRecord>,
    /// Why no pin may be built on anything it publishes, `""` when nothing says so
    pub suppressed: String,
    /// The exports bound to an `Exported` proxy
    pub proxies: Vec<String>,
    /// `module_interface.dep_record(interface)`
    pub record: DepRecord,
}

fn json_str(value: &serde_json::Value, field: &str) -> Result<String, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("no string {field}"))
}

fn json_int(value: &serde_json::Value, field: &str) -> Result<i64, String> {
    value.get(field).and_then(serde_json::Value::as_i64).ok_or_else(|| format!("no int {field}"))
}

fn json_bool(value: &serde_json::Value, field: &str) -> Result<bool, String> {
    value.get(field).and_then(serde_json::Value::as_bool).ok_or_else(|| format!("no bool {field}"))
}

fn json_strs(value: &serde_json::Value, field: &str) -> Result<Vec<String>, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("no list {field}"))?
        .iter()
        .map(|item| item.as_str().map(str::to_string).ok_or_else(|| format!("{field} item")))
        .collect()
}

fn json_map<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    value.get(field).and_then(serde_json::Value::as_object).ok_or_else(|| format!("no map {field}"))
}

impl ExportSig {
    /// An `ExportSig` as `dataclasses.asdict` writes it.
    pub fn from_json(value: &serde_json::Value) -> Result<ExportSig, String> {
        let impls = value
            .get("impls")
            .and_then(serde_json::Value::as_array)
            .ok_or("no impls")?
            .iter()
            .map(|item| {
                Ok(ImplSig {
                    params: json_strs(item, "params")?,
                    required: usize::try_from(json_int(item, "required")?)
                        .map_err(|_| "negative required")?,
                    open_ended: json_bool(item, "open_ended")?,
                    ret: json_str(item, "ret")?,
                    fits: json_str(item, "fits")?,
                    names: json_strs(item, "names")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(ExportSig {
            name: json_str(value, "name")?,
            kind: json_str(value, "kind")?,
            params: json_strs(value, "params")?,
            required: json_int(value, "required")?,
            open_ended: json_bool(value, "open_ended")?,
            ret: json_str(value, "ret")?,
            annotated: json_bool(value, "annotated")?,
            impls,
            line: json_int(value, "line")?,
            names: json_strs(value, "names")?,
        })
    }
}

impl ClassSig {
    /// A `ClassSig` as `dataclasses.asdict` writes it.
    pub fn from_json(value: &serde_json::Value) -> Result<ClassSig, String> {
        let fields = json_map(value, "fields")?
            .iter()
            .map(|(name, ty)| Ok((name.clone(), ty.as_str().ok_or("field type")?.to_string())))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let methods = json_map(value, "methods")?
            .iter()
            .map(|(name, sig)| Ok((name.clone(), ExportSig::from_json(sig)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let members = json_map(value, "members")?
            .iter()
            .map(|(name, sig)| Ok((name.clone(), ExportSig::from_json(sig)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        Ok(ClassSig {
            name: json_str(value, "name")?,
            id: json_str(value, "id")?,
            fields,
            required: json_int(value, "required")?,
            methods,
            members,
        })
    }
}

impl DepRecord {
    /// A `DepRecord` as `dataclasses.asdict` writes it.
    pub fn from_json(value: &serde_json::Value) -> Result<DepRecord, String> {
        Ok(DepRecord {
            path: json_str(value, "path")?,
            mtime_ns: json_int(value, "mtime_ns")?,
            size: json_int(value, "size")?,
            digest: json_str(value, "digest")?,
            routes: json_str(value, "routes")?,
        })
    }
}

impl ModuleInterface {
    /// The interface a host serialized as `{"interface": asdict(interface), "record":
    /// asdict(dep_record(interface))}`, dict order kept.
    pub fn from_json(text: &str) -> Result<ModuleInterface, String> {
        let document: serde_json::Value =
            serde_json::from_str(text).map_err(|err| err.to_string())?;
        let interface = document.get("interface").ok_or("no interface")?;
        let exports = json_map(interface, "exports")?
            .iter()
            .map(|(name, sig)| Ok((name.clone(), ExportSig::from_json(sig)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let classes = json_map(interface, "classes")?
            .iter()
            .map(|(name, sig)| Ok((name.clone(), ClassSig::from_json(sig)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let extensions = json_map(interface, "extensions")?
            .iter()
            .map(|(cid, methods)| {
                let methods = methods
                    .as_object()
                    .ok_or("extension methods")?
                    .iter()
                    .map(|(name, sig)| Ok((name.clone(), ExportSig::from_json(sig)?)))
                    .collect::<Result<IndexMap<_, _>, String>>()?;
                Ok((cid.clone(), methods))
            })
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let deps = json_map(interface, "deps")?
            .iter()
            .map(|(path, record)| Ok((path.clone(), DepRecord::from_json(record)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let all = match interface.get("all") {
            None | Some(serde_json::Value::Null) => None,
            Some(_) => Some(json_strs(interface, "all")?),
        };
        Ok(ModuleInterface {
            path: json_str(interface, "path")?,
            exports,
            all,
            classes,
            extensions,
            deps,
            suppressed: json_str(interface, "suppressed")?,
            proxies: json_strs(interface, "proxies")?,
            record: DepRecord::from_json(document.get("record").ok_or("no record")?)?,
        })
    }
}

/// What the call-site isolation reads of the interface an imported module
/// publishes: where a call through its names goes, and the dependency record a
/// route read off it keeps (`ModuleInterface.call_routes`, `.reexports`, `.deps`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallInterface {
    /// Resolved source path of the module
    pub path: String,
    /// The module's literal `__all__`, `None` when it spells none
    pub all: Option<Vec<String>>,
    /// Dotted path -> route word
    pub call_routes: IndexMap<String, String>,
    /// Name -> (level, module, attribute) of the import binding it
    pub reexports: IndexMap<String, (i64, String, String)>,
    /// Every module this one's types were derived from or that it imports
    pub deps: IndexMap<String, DepRecord>,
    /// `module_interface.dep_record(interface)`
    pub record: DepRecord,
}

impl CallInterface {
    /// The interface a host serialized as for [`ModuleInterface::from_json`], read
    /// for what the isolation needs.
    pub fn from_json(text: &str) -> Result<CallInterface, String> {
        let document: serde_json::Value =
            serde_json::from_str(text).map_err(|err| err.to_string())?;
        let interface = document.get("interface").ok_or("no interface")?;
        let all = match interface.get("all") {
            None | Some(serde_json::Value::Null) => None,
            Some(_) => Some(json_strs(interface, "all")?),
        };
        let call_routes = json_map(interface, "call_routes")?
            .iter()
            .map(|(path, route)| Ok((path.clone(), route.as_str().ok_or("route")?.to_string())))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let reexports = json_map(interface, "reexports")?
            .iter()
            .map(|(name, target)| {
                let target = target.as_array().filter(|items| items.len() == 3);
                let target = target.ok_or("reexport")?;
                let level = target[0].as_i64().ok_or("reexport level")?;
                let module = target[1].as_str().ok_or("reexport module")?.to_string();
                let attr = target[2].as_str().ok_or("reexport attribute")?.to_string();
                Ok((name.clone(), (level, module, attr)))
            })
            .collect::<Result<IndexMap<_, _>, String>>()?;
        let deps = json_map(interface, "deps")?
            .iter()
            .map(|(path, record)| Ok((path.clone(), DepRecord::from_json(record)?)))
            .collect::<Result<IndexMap<_, _>, String>>()?;
        Ok(CallInterface {
            path: json_str(interface, "path")?,
            all,
            call_routes,
            reexports,
            deps,
            record: DepRecord::from_json(document.get("record").ok_or("no record")?)?,
        })
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
    fn a_call_interface_reads_the_routes_and_the_record() {
        let record = r#"{"path": "/m.py", "mtime_ns": 1, "size": 2, "digest": "d", "routes": "r"}"#;
        let text = format!(
            r#"{{"interface": {{"path": "/m.py", "all": null, "exports": {{"g": {{"impls": [[]]}}}},
                "call_routes": {{"f": "fast", "P.new": "skip"}},
                "reexports": {{"g": [1, "leaf", "g"], "*": [0, "star", "*"]}},
                "deps": {{"/d.py": {record}}}}}, "record": {record}}}"#
        );
        let interface = CallInterface::from_json(&text).expect("parses");
        assert_eq!(interface.path, "/m.py");
        assert_eq!(interface.all, None);
        assert_eq!(interface.call_routes["P.new"], "skip");
        assert_eq!(interface.reexports["g"], (1, "leaf".to_string(), "g".to_string()));
        assert_eq!(interface.reexports["*"].2, "*");
        assert_eq!(interface.deps["/d.py"].routes, "r");
        assert_eq!(interface.record.digest, "d");
        assert!(CallInterface::from_json(&text.replace(r#""call_routes""#, r#""other""#)).is_err());
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
