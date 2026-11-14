//! The data files of `pynecore/transformers`, compiled in as `static` tables
//! (`build.rs` writes them from the JSON): nothing is parsed when the crate runs.
//!
//! They are part of a step's source closure (see `transformers/native.py`): a
//! regenerated file changes the digest of every step that reads it, so a binary
//! built with the old copy is out of use until it is rebuilt.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde_json::Value;

mod tables {
    use super::{InlineStatic, JsonStatic as J, LibStatic};

    /// Per interpreter, its `str.isprintable` rejections ([`super::PrintTable`]).
    type Unprintable = &'static [((u32, u32), &'static [(u32, u32)])];

    include!(concat!(env!("OUT_DIR"), "/data_tables.rs"));
}

pub(crate) use tables::PORTED_SHA256;

/// The `u16` that stands for `None` of an optional string or list (`build.rs`).
const NONE: u16 = u16::MAX;

/// A string of the tables, by its index.
fn s(id: u16) -> &'static str {
    let id = usize::from(id);
    let start = if id == 0 { 0 } else { tables::ENDS[id - 1] as usize };
    &tables::TEXT[start..tables::ENDS[id] as usize]
}

fn opt_s(id: u16) -> Option<String> {
    (id != NONE).then(|| s(id).to_string())
}

/// A list of strings, `(start, len)` in `POOL`.
fn list((start, len): (u16, u16)) -> Vec<String> {
    let start = usize::from(start);
    tables::POOL[start..start + usize::from(len)].iter().map(|&id| s(id).to_string()).collect()
}

fn opt_list(range: (u16, u16)) -> Option<Vec<String>> {
    (range.0 != NONE).then(|| list(range))
}

fn pairs(items: &[(u16, u16)]) -> HashMap<String, String> {
    items.iter().map(|&(key, value)| (s(key).to_string(), s(value).to_string())).collect()
}

/// An entry of `lib_types.json`'s `names`, or one implementation of an overload
/// group, as `build.rs` writes it: strings by index, lists as ranges.
pub(crate) struct LibStatic {
    kind: u16,
    ty: u16,
    callable: bool,
    ret: u16,
    params: (u16, u16),
    names: (u16, u16),
    defaults: u8,
    default_ty: (u16, u16),
    /// A range of `BOOLS`
    default_none_ok: (u16, u16),
    vararg: bool,
    kwarg: bool,
    /// A range of `IMPLS`
    impls: (u16, u16),
}

impl LibStatic {
    fn impls(&self) -> &'static [LibStatic] {
        let start = usize::from(self.impls.0);
        &tables::IMPLS[start..start + usize::from(self.impls.1)]
    }
}

/// A function of `call_inline_templates.json` as `build.rs` writes it.
pub(crate) enum InlineStatic {
    Derive { params: (u16, u16), body: &'static JsonStatic },
    Varargs { builtin: u16, na_anchor: u16, builtins_anchor: u16 },
}

/// A JSON value as `build.rs` writes it: an inline template's body.
#[allow(dead_code)] // the variants the current templates hold vary
pub(crate) enum JsonStatic {
    Null,
    Bool(bool),
    U(u64),
    I(i64),
    /// A float by its bits
    F(u64),
    S(u16),
    A(&'static [JsonStatic]),
    O(&'static [(u16, JsonStatic)]),
}

impl JsonStatic {
    fn value(&self) -> Value {
        match self {
            JsonStatic::Null => Value::Null,
            JsonStatic::Bool(flag) => Value::Bool(*flag),
            JsonStatic::U(n) => Value::from(*n),
            JsonStatic::I(n) => Value::from(*n),
            JsonStatic::F(bits) => Value::from(f64::from_bits(*bits)),
            JsonStatic::S(id) => Value::String(s(*id).to_string()),
            JsonStatic::A(items) => Value::Array(items.iter().map(JsonStatic::value).collect()),
            JsonStatic::O(map) => Value::Object(
                map.iter().map(|(key, item)| (s(*key).to_string(), item.value())).collect(),
            ),
        }
    }
}

/// `module_properties.json`: module -> member name -> its `"type"` (`"property"` or
/// `"variable"`; `module_property.module_properties`).
pub fn module_properties() -> &'static HashMap<String, HashMap<String, String>> {
    static REGISTRY: OnceLock<HashMap<String, HashMap<String, String>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        tables::MODULE_PROPERTIES
            .iter()
            .map(|&(module, members)| (s(module).to_string(), pairs(members)))
            .collect()
    })
}

/// `lib_types.json`'s `star_exports`: lib module -> its `__all__` (`lib_registry.lib_star_exports`).
pub fn star_exports() -> &'static HashMap<String, Vec<String>> {
    static EXPORTS: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    EXPORTS.get_or_init(|| {
        tables::STAR_EXPORTS
            .iter()
            .map(|&(module, names)| (s(module).to_string(), list(names)))
            .collect()
    })
}

/// `lib_types.json`'s `routes`: lib callee path (`lib.ta.sma`) -> its call-site route
/// (`lib_registry.lib_routes`).
pub fn routes() -> &'static HashMap<String, String> {
    static ROUTES: OnceLock<HashMap<String, String>> = OnceLock::new();
    ROUTES.get_or_init(|| pairs(tables::ROUTES))
}

/// `lib_types.json`'s `classes`: lib class name -> its fields and their types, in
/// declaration order.
pub fn classes() -> &'static HashMap<String, Vec<(String, String)>> {
    static CLASSES: OnceLock<HashMap<String, Vec<(String, String)>>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        tables::CLASSES
            .iter()
            .map(|&(name, fields)| {
                let fields =
                    fields.iter().map(|&(field, ty)| (s(field).to_string(), s(ty).to_string()));
                (s(name).to_string(), fields.collect())
            })
            .collect()
    })
}

/// `lib_types.json`'s `scalar_classes`: lib class name -> the scalar type it is.
pub fn scalar_classes() -> &'static HashMap<String, String> {
    static SCALARS: OnceLock<HashMap<String, String>> = OnceLock::new();
    SCALARS.get_or_init(|| pairs(tables::SCALAR_CLASSES))
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
        tables::NAMES
            .iter()
            .map(|(name, entry)| {
                let entry = LibName {
                    kind: s(entry.kind).to_string(),
                    ty: opt_s(entry.ty),
                    callable: entry.callable,
                    ret: opt_s(entry.ret),
                    impl_rets: entry.impls().iter().map(|item| opt_s(item.ret)).collect(),
                };
                (s(*name).to_string(), entry)
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

fn lib_entry(entry: &LibStatic) -> LibEntry {
    let (start, len) = entry.default_none_ok;
    LibEntry {
        kind: s(entry.kind).to_string(),
        ty: opt_s(entry.ty),
        callable: entry.callable,
        ret: opt_s(entry.ret),
        params: opt_list(entry.params),
        names: opt_list(entry.names),
        defaults: i64::from(entry.defaults),
        default_ty: list(entry.default_ty),
        default_none_ok: (start != NONE).then(|| {
            tables::BOOLS[usize::from(start)..usize::from(start) + usize::from(len)].to_vec()
        }),
        vararg: entry.vararg,
        kwarg: entry.kwarg,
        impls: entry.impls().iter().map(lib_entry).collect(),
    }
}

/// Every entry of `lib_types.json`'s `names`, whole (`lib_registry.lib_types`).
pub fn lib_entries() -> &'static HashMap<String, LibEntry> {
    static ENTRIES: OnceLock<HashMap<String, LibEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        tables::NAMES.iter().map(|(name, entry)| (s(*name).to_string(), lib_entry(entry))).collect()
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
        let functions = tables::INLINE_FUNCTIONS
            .iter()
            .map(|(name, function)| {
                let function = match function {
                    InlineStatic::Derive { params, body } => {
                        InlineFunction::Derive { params: list(*params), body: body.value() }
                    }
                    InlineStatic::Varargs { builtin, na_anchor, builtins_anchor } => {
                        InlineFunction::Varargs {
                            builtin: s(*builtin).to_string(),
                            na_anchor: s(*na_anchor).to_string(),
                            builtins_anchor: s(*builtins_anchor).to_string(),
                        }
                    }
                };
                (s(*name).to_string(), function)
            })
            .collect();
        InlineTemplates {
            functions,
            callees: pairs(tables::INLINE_CALLEES),
            loop_ranges: list(tables::LOOP_RANGES),
            varargs_arities: tables::VARARGS_ARITIES.to_vec(),
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
        let names = |range| list(range).into_iter().collect();
        PythonNames {
            builtins: names(tables::BUILTINS),
            keywords: names(tables::KEYWORDS),
            stdlib: names(tables::STDLIB),
            unprintable: tables::UNPRINTABLE
                .iter()
                .map(|(version, ranges)| (*version, ranges.to_vec()))
                .collect(),
        }
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
