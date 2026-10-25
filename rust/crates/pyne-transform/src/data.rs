//! The data files of `pynecore/transformers`, compiled in.
//!
//! They are part of a step's source closure (see `transformers/native.py`): a
//! regenerated file changes the digest of every step that reads it, so a binary
//! built with the old copy is out of use until it is rebuilt.

use std::collections::{HashMap, HashSet};
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

/// A function of `call_inline_templates.json` (`call_inline.inline_templates()['functions']`).
#[derive(Clone, Debug, PartialEq)]
pub enum InlineFunction {
    /// The body's own expression: the parameter names and the encoded expression
    /// (`{"param": i}` for a read of the i-th parameter, `{"_type": class, field:
    /// value, ..., "arg": true}` for any other node)
    Derive { params: Vec<String>, body: Value },
    /// The fixed-arity expansion of `math.max` / `math.min`: the builtin the body
    /// ends in, and the `inline_support` anchors of `_na_of_operands` and `builtins`
    Varargs { builtin: String, na_anchor: String, builtins_anchor: String },
}

/// `call_inline_templates.json`, the generated inline template registry
/// (`call_inline.inline_templates`).
#[derive(Clone, Debug, PartialEq)]
pub struct InlineTemplates {
    /// Function path (`pynecore.lib.math.abs`) -> its template
    pub functions: HashMap<String, InlineFunction>,
    /// Callee path, as the import map resolves it -> the function path
    pub callees: HashMap<String, String>,
    /// The paths of `pine_range`
    pub loop_ranges: Vec<String>,
    /// The positional arities the varargs expansion is written for
    pub varargs_arities: Vec<usize>,
}

/// `call_inline_templates.json`, parsed once.
pub fn inline_templates() -> &'static InlineTemplates {
    static TEMPLATES: OnceLock<InlineTemplates> = OnceLock::new();
    TEMPLATES.get_or_init(|| {
        let text = include_str!("../../../../src/pynecore/transformers/call_inline_templates.json");
        let document: Map<String, Value> =
            serde_json::from_str(text).expect("call_inline_templates.json");
        let section = |name: &str| {
            document
                .get(name)
                .unwrap_or_else(|| panic!("call_inline_templates.json has no {name}"))
                .clone()
        };
        let text = |entry: &Value, field: &str| {
            entry
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("call_inline_templates.json: no {field}"))
                .to_string()
        };
        let Value::Object(functions) = section("functions") else { panic!("functions") };
        let functions = functions
            .iter()
            .map(|(name, entry)| {
                let function = match entry.get("kind").and_then(Value::as_str) {
                    Some("derive") => InlineFunction::Derive {
                        params: serde_json::from_value(entry["params"].clone())
                            .unwrap_or_else(|_| panic!("functions.{name}.params")),
                        body: entry["body"].clone(),
                    },
                    Some("varargs") => InlineFunction::Varargs {
                        builtin: text(entry, "builtin"),
                        na_anchor: text(entry, "na_anchor"),
                        builtins_anchor: text(entry, "builtins_anchor"),
                    },
                    other => panic!("functions.{name}.kind {other:?}"),
                };
                (name.clone(), function)
            })
            .collect();
        InlineTemplates {
            functions,
            callees: serde_json::from_value(section("callees")).expect("callees"),
            loop_ranges: serde_json::from_value(section("loop_ranges")).expect("loop_ranges"),
            varargs_arities: serde_json::from_value(section("varargs_arities"))
                .expect("varargs_arities"),
        }
    })
}

/// `python_names.json` (`python_names`): the builtins, keywords and standard
/// library modules of every supported interpreter, and per interpreter the code
/// points `str.isprintable` rejects.
#[derive(Debug)]
pub struct PythonNames {
    pub builtins: HashSet<String>,
    pub keywords: HashSet<String>,
    pub stdlib: HashSet<String>,
    /// Per interpreter, the code points `str.isprintable` rejects
    pub unprintable: Vec<PrintTable>,
}

/// An interpreter's `(major, minor)` and the inclusive code point ranges, ascending,
/// whose characters its `str.isprintable` rejects.
pub type PrintTable = ((u32, u32), Vec<(u32, u32)>);

/// The compiled-in `python_names.json`.
pub fn python_names() -> &'static PythonNames {
    static NAMES: OnceLock<PythonNames> = OnceLock::new();
    NAMES.get_or_init(|| {
        let text = include_str!("../../../../src/pynecore/transformers/python_names.json");
        let mut document: HashMap<String, serde_json::Value> =
            serde_json::from_str(text).expect("python_names.json");
        let mut section = |name: &str| document.remove(name).expect("python_names.json section");
        let names = |value: serde_json::Value| -> HashSet<String> {
            serde_json::from_value(value).expect("python_names.json names")
        };
        let builtins = names(section("builtins"));
        let keywords = names(section("keywords"));
        let stdlib = names(section("stdlib"));
        let tables: HashMap<String, Vec<(u32, u32)>> =
            serde_json::from_value(section("unprintable")).expect("python_names.json unprintable");
        let unprintable = tables
            .into_iter()
            .map(|(version, ranges)| {
                let (major, minor) = version.split_once('.').expect("major.minor");
                ((major.parse().expect("major"), minor.parse().expect("minor")), ranges)
            })
            .collect();
        PythonNames { builtins, keywords, stdlib, unprintable }
    })
}

/// The newest interpreter `python_names.json` knows: the one a host without an
/// interpreter of its own follows.
pub fn newest_python() -> (u32, u32) {
    python_names().unprintable.iter().map(|(version, _)| *version).max().expect("a version")
}

/// `str.isprintable` of one character under an interpreter; `None` for an
/// interpreter `python_names.json` does not know.
pub fn is_printable(c: char, python: (u32, u32)) -> Option<bool> {
    let (_, ranges) = python_names().unprintable.iter().find(|(version, _)| *version == python)?;
    let code = c as u32;
    let index = ranges.partition_point(|&(_, last)| last < code);
    Some(!ranges.get(index).is_some_and(|&(first, _)| first <= code))
}

/// Whether a module's top-level package is part of the standard library
/// (`python_names.is_stdlib`).
pub fn is_stdlib(module: &str) -> bool {
    python_names().stdlib.contains(module.split('.').next().unwrap_or(module))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_python_names_answer_like_an_interpreter() {
        let names = python_names();
        assert!(names.builtins.contains("len") && names.builtins.contains("__import__"));
        assert!(names.keywords.contains("lambda") && !names.keywords.contains("match"));
        assert!(is_stdlib("os.path") && is_stdlib("collections") && !is_stdlib("pynecore"));
    }

    #[test]
    fn printability_follows_the_interpreter() {
        let python = newest_python();
        assert_eq!(is_printable('a', python), Some(true));
        assert_eq!(is_printable(' ', python), Some(true));
        assert_eq!(is_printable('\n', python), Some(false));
        assert_eq!(is_printable('\u{a0}', python), Some(false));
        assert_eq!(is_printable('\u{25b2}', python), Some(true));
        assert_eq!(is_printable('\u{200b}', python), Some(false));
        assert_eq!(is_printable('a', (2, 7)), None);
    }
}
