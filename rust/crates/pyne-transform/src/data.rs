//! The data files of `pynecore/transformers`, compiled in.
//!
//! They are part of a step's source closure (see `transformers/native.py`): a
//! regenerated file changes the digest of every step that reads it, so a binary
//! built with the old copy is out of use until it is rebuilt.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::{Map, Value};

/// `module_properties.json`: module -> member name -> its `"type"` (`"property"` or
/// `"variable"`; `module_property.module_properties`).
pub fn module_properties() -> &'static HashMap<String, HashMap<String, String>> {
    static REGISTRY: OnceLock<HashMap<String, HashMap<String, String>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let text = include_str!("../../../../src/pynecore/transformers/module_properties.json");
        let document: HashMap<String, HashMap<String, HashMap<String, String>>> =
            serde_json::from_str(text).expect("module_properties.json");
        document
            .into_iter()
            .map(|(module, members)| {
                let members = members
                    .into_iter()
                    .map(|(name, mut entry)| (name, entry.remove("type").expect("type")))
                    .collect();
                (module, members)
            })
            .collect()
    })
}

/// `lib_types.json`, the generated lib registry (`lib_registry.lib_registry`).
fn lib_registry() -> &'static Map<String, Value> {
    static DOCUMENT: OnceLock<Map<String, Value>> = OnceLock::new();
    DOCUMENT.get_or_init(|| {
        let text = include_str!("../../../../src/pynecore/transformers/lib_types.json");
        serde_json::from_str(text).expect("lib_types.json")
    })
}

/// A top-level section of the lib registry.
fn section(name: &str) -> &'static Value {
    lib_registry().get(name).unwrap_or_else(|| panic!("lib_types.json has no {name}"))
}

/// `lib_types.json`'s `star_exports`: lib module -> its `__all__` (`lib_registry.lib_star_exports`).
pub fn star_exports() -> &'static HashMap<String, Vec<String>> {
    static EXPORTS: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    EXPORTS.get_or_init(|| {
        serde_json::from_value(section("star_exports").clone()).expect("star_exports")
    })
}

/// `lib_types.json`'s `routes`: lib callee path (`lib.ta.sma`) -> its call-site route
/// (`lib_registry.lib_routes`).
pub fn routes() -> &'static HashMap<String, String> {
    static ROUTES: OnceLock<HashMap<String, String>> = OnceLock::new();
    ROUTES.get_or_init(|| serde_json::from_value(section("routes").clone()).expect("routes"))
}

/// `lib_types.json`'s `classes`: lib class name -> its fields and their types, in
/// declaration order.
pub fn classes() -> &'static HashMap<String, Vec<(String, String)>> {
    static CLASSES: OnceLock<HashMap<String, Vec<(String, String)>>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        let Value::Object(classes) = section("classes") else { panic!("classes") };
        classes
            .iter()
            .map(|(name, fields)| {
                let Value::Object(fields) = fields else { panic!("classes.{name}") };
                let fields = fields
                    .iter()
                    .map(|(field, ty)| {
                        let ty = ty.as_str().unwrap_or_else(|| panic!("classes.{name}.{field}"));
                        (field.clone(), ty.to_string())
                    })
                    .collect();
                (name.clone(), fields)
            })
            .collect()
    })
}

/// `lib_types.json`'s `scalar_classes`: lib class name -> the scalar type it is.
pub fn scalar_classes() -> &'static HashMap<String, String> {
    static SCALARS: OnceLock<HashMap<String, String>> = OnceLock::new();
    SCALARS.get_or_init(|| {
        serde_json::from_value(section("scalar_classes").clone()).expect("scalar_classes")
    })
}

/// An entry of `lib_types.json`'s `names` (`lib_registry.lib_types()[name]`): the
/// fields a native step reads so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibName {
    /// `value`, `function` or `overloads`
    pub kind: String,
    /// A value's type (`entry.get('ty')`)
    pub ty: Option<String>,
    /// Whether a value can be called as well (`entry.get('callable', False)`)
    pub callable: bool,
    /// A function's return type (`entry.get('ret')`)
    pub ret: Option<String>,
    /// The return type of every implementation of an overload group, in order
    /// (`[impl.get('ret') for impl in entry['impls']]`)
    pub impl_rets: Vec<Option<String>>,
}

/// `lib_types.json`'s `names`: registry key (`ta.sma`, `syminfo.mintick`) -> its entry
/// (`lib_registry.lib_types`).
pub fn lib_names() -> &'static HashMap<String, LibName> {
    static NAMES: OnceLock<HashMap<String, LibName>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let Value::Object(names) = section("names") else { panic!("names") };
        let text = |entry: &Value, field: &str| {
            entry.get(field).and_then(Value::as_str).map(str::to_string)
        };
        names
            .iter()
            .map(|(name, entry)| {
                let impl_rets = entry
                    .get("impls")
                    .and_then(Value::as_array)
                    .map(|impls| impls.iter().map(|item| text(item, "ret")).collect())
                    .unwrap_or_default();
                let entry = LibName {
                    kind: text(entry, "kind").unwrap_or_else(|| panic!("names.{name}.kind")),
                    ty: text(entry, "ty"),
                    callable: entry.get("callable").and_then(Value::as_bool).unwrap_or(false),
                    ret: text(entry, "ret"),
                    impl_rets,
                };
                (name.clone(), entry)
            })
            .collect()
    })
}

/// A whole entry of `lib_types.json`'s `names` (`lib_registry.lib_types()[name]`), or one
/// implementation of an overload group: every field the type pass reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibEntry {
    /// `value`, `function` or `overloads`; empty for an implementation
    pub kind: String,
    /// A value's type (`entry.get('ty')`)
    pub ty: Option<String>,
    /// Whether a value can be called as well (`entry.get('callable', False)`)
    pub callable: bool,
    /// What a function or an implementation returns (`entry.get('ret')`)
    pub ret: Option<String>,
    /// Type of each positional parameter (`entry.get('params')`)
    pub params: Option<Vec<String>>,
    /// Name of each positional parameter (`entry.get('names')`)
    pub names: Option<Vec<String>>,
    /// How many trailing positional parameters have a default (`entry['defaults']`)
    pub defaults: i64,
    /// The type of each default (`entry.get('default_ty', ())`)
    pub default_ty: Vec<String>,
    /// Whether each literal-`None` default is taken by the annotation
    /// (`entry.get('default_none_ok')`)
    pub default_none_ok: Option<Vec<bool>>,
    /// Whether a `*args` follows (`entry.get('vararg') is not None`)
    pub vararg: bool,
    /// Whether a `**kwargs` follows (`bool(entry.get('kwarg'))`)
    pub kwarg: bool,
    /// The implementations of an overload group, in order
    pub impls: Vec<LibEntry>,
}

fn lib_entry(entry: &Value) -> LibEntry {
    let text = |field: &str| entry.get(field).and_then(Value::as_str).map(str::to_string);
    let texts = |field: &str| {
        entry.get(field).and_then(Value::as_array).map(|items| {
            items.iter().map(|item| item.as_str().unwrap_or("").to_string()).collect::<Vec<_>>()
        })
    };
    LibEntry {
        kind: text("kind").unwrap_or_default(),
        ty: text("ty"),
        callable: entry.get("callable").and_then(Value::as_bool).unwrap_or(false),
        ret: text("ret"),
        params: texts("params"),
        names: texts("names"),
        defaults: entry.get("defaults").and_then(Value::as_i64).unwrap_or(0),
        default_ty: texts("default_ty").unwrap_or_default(),
        default_none_ok: entry
            .get("default_none_ok")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(|item| item.as_bool().unwrap_or(false)).collect()),
        vararg: entry.get("vararg").is_some_and(|value| !value.is_null()),
        kwarg: entry.get("kwarg").is_some_and(|value| match value {
            Value::Null => false,
            Value::Bool(flag) => *flag,
            Value::String(text) => !text.is_empty(),
            _ => true,
        }),
        impls: entry
            .get("impls")
            .and_then(Value::as_array)
            .map(|impls| impls.iter().map(lib_entry).collect())
            .unwrap_or_default(),
    }
}

/// Every entry of `lib_types.json`'s `names`, whole (`lib_registry.lib_types`).
pub fn lib_entries() -> &'static HashMap<String, LibEntry> {
    static ENTRIES: OnceLock<HashMap<String, LibEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        let Value::Object(names) = section("names") else { panic!("names") };
        names.iter().map(|(name, entry)| (name.clone(), lib_entry(entry))).collect()
    })
}

/// Every dotted path the registry's names hang off (`pine_type_infer.lib_namespaces`).
pub fn lib_namespaces() -> &'static std::collections::HashSet<String> {
    static NAMESPACES: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    NAMESPACES.get_or_init(|| {
        let mut out = std::collections::HashSet::new();
        for key in lib_entries().keys() {
            let parts: Vec<&str> = key.split('.').collect();
            for depth in 1..parts.len() {
                out.insert(parts[..depth].join("."));
            }
        }
        out
    })
}
