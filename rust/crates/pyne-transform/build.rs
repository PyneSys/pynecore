//! Writes the data files of `pynecore/transformers` as `static` tables
//! (`data.rs` includes them), so the crate reads no JSON when it runs. It also
//! writes the digest of `ported.json`, which a host takes as the pipeline's.
//!
//! Every string is written once, into one text (`TEXT`, its ends in `ENDS`),
//! and a table names it by its index; a list of strings is a range of
//! `POOL`. Indices compress far better than a pointer per string.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

#[path = "src/sha256.rs"]
#[allow(dead_code)]
mod sha256;

/// The `u16` that stands for `None` of an optional string or list.
const NONE: u16 = u16::MAX;

fn read(path: &Path) -> String {
    println!("cargo:rerun-if-changed={}", path.display());
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn parse(path: &Path) -> Map<String, Value> {
    serde_json::from_str(&read(path)).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn object<'a>(value: &'a Value, what: &str) -> &'a Map<String, Value> {
    value.as_object().unwrap_or_else(|| panic!("{what} is not an object"))
}

fn string<'a>(value: &'a Value, what: &str) -> &'a str {
    value.as_str().unwrap_or_else(|| panic!("{what} is not a string"))
}

fn section<'a>(document: &'a Map<String, Value>, name: &str) -> &'a Value {
    document.get(name).unwrap_or_else(|| panic!("no section {name}"))
}

fn strings<'a>(value: &'a Value, what: &str) -> Vec<&'a str> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{what} is not a list"))
        .iter()
        .map(|item| string(item, what))
        .collect()
}

/// A JSON array of strings, a non-string item read as `""` (as `data.rs`
/// always read it).
fn texts<'a>(entry: &'a Value, field: &str) -> Option<Vec<&'a str>> {
    entry
        .get(field)
        .and_then(Value::as_array)
        .map(|items| items.iter().map(|item| item.as_str().unwrap_or("")).collect())
}

/// The shared text, the lists and the implementations the tables index.
#[derive(Default)]
struct Gen {
    text: String,
    ends: Vec<u32>,
    ids: HashMap<String, u16>,
    pool: Vec<u16>,
    bools: Vec<bool>,
    impls: Vec<String>,
}

fn index(n: usize, what: &str) -> u16 {
    u16::try_from(n).ok().filter(|&n| n != NONE).unwrap_or_else(|| panic!("too many {what}"))
}

impl Gen {
    /// A string's index.
    fn s(&mut self, text: &str) -> u16 {
        if let Some(&id) = self.ids.get(text) {
            return id;
        }
        let id = index(self.ends.len(), "strings");
        self.text.push_str(text);
        self.ends.push(u32::try_from(self.text.len()).expect("a text under 4 GiB"));
        self.ids.insert(text.to_string(), id);
        id
    }

    fn opt_s(&mut self, text: Option<&str>) -> u16 {
        text.map_or(NONE, |text| self.s(text))
    }

    /// A list of strings as its range of `POOL`, `(start, len)`.
    fn list(&mut self, items: &[&str]) -> String {
        let ids: Vec<u16> = items.iter().map(|item| self.s(item)).collect();
        let start = index(self.pool.len(), "pool entries");
        self.pool.extend(ids);
        format!("({start},{})", index(items.len(), "list items"))
    }

    fn opt_list(&mut self, items: Option<&[&str]>) -> String {
        items.map_or(format!("({NONE},0)"), |items| self.list(items))
    }

    /// `name -> string` as `&[(u16, u16)]`, in the file's order.
    fn pairs(&mut self, map: &Map<String, Value>, what: &str) -> String {
        let mut out = String::from("&[");
        for (key, value) in map {
            let key = self.s(key);
            let value = self.s(string(value, what));
            write!(out, "({key},{value}),").unwrap();
        }
        out.push(']');
        out
    }

    /// An entry of `lib_types.json`'s `names`, or an implementation of one, as a
    /// `LibStatic` literal.
    fn lib_entry(&mut self, entry: &Value) -> String {
        let text = |field: &str| entry.get(field).and_then(Value::as_str);
        let kind = self.s(text("kind").unwrap_or(""));
        let ty = self.opt_s(text("ty"));
        let callable = entry.get("callable").and_then(Value::as_bool).unwrap_or(false);
        let ret = self.opt_s(text("ret"));
        let params = self.opt_list(texts(entry, "params").as_deref());
        let names = self.opt_list(texts(entry, "names").as_deref());
        let defaults = entry.get("defaults").and_then(Value::as_i64).unwrap_or(0);
        let defaults = u8::try_from(defaults).expect("a default count of 0..=255");
        let default_ty = self.list(&texts(entry, "default_ty").unwrap_or_default());
        let default_none_ok = match entry.get("default_none_ok").and_then(Value::as_array) {
            None => format!("({NONE},0)"),
            Some(items) => {
                let start = index(self.bools.len(), "bools");
                self.bools.extend(items.iter().map(|item| item.as_bool().unwrap_or(false)));
                format!("({start},{})", index(items.len(), "bools"))
            }
        };
        let vararg = entry.get("vararg").is_some_and(|value| !value.is_null());
        let kwarg = entry.get("kwarg").is_some_and(|value| match value {
            Value::Null => false,
            Value::Bool(flag) => *flag,
            Value::String(text) => !text.is_empty(),
            _ => true,
        });
        let impls: Vec<String> = entry
            .get("impls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|item| self.lib_entry(item))
            .collect();
        let start = index(self.impls.len(), "implementations");
        let count = index(impls.len(), "implementations");
        self.impls.extend(impls);
        format!(
            "LibStatic {{kind: {kind},ty: {ty},callable: {callable},ret: {ret},params: {params},\
             names: {names},defaults: {defaults},default_ty: {default_ty},\
             default_none_ok: {default_none_ok},vararg: {vararg},kwarg: {kwarg},\
             impls: ({start},{count})}}"
        )
    }

    /// A JSON value as a `JsonStatic` literal (`data.rs` turns it into a
    /// `serde_json::Value`).
    fn json(&mut self, out: &mut String, value: &Value) {
        match value {
            Value::Null => out.push_str("J::Null"),
            Value::Bool(flag) => write!(out, "J::Bool({flag})").unwrap(),
            Value::Number(number) => {
                if let Some(n) = number.as_u64() {
                    write!(out, "J::U({n})").unwrap();
                } else if let Some(n) = number.as_i64() {
                    write!(out, "J::I({n})").unwrap();
                } else {
                    let bits = number.as_f64().expect("a JSON number").to_bits();
                    write!(out, "J::F({bits:#x})").unwrap();
                }
            }
            Value::String(text) => {
                let id = self.s(text);
                write!(out, "J::S({id})").unwrap();
            }
            Value::Array(items) => {
                out.push_str("J::A(&[");
                for item in items {
                    self.json(out, item);
                    out.push(',');
                }
                out.push_str("])");
            }
            Value::Object(map) => {
                out.push_str("J::O(&[");
                for (key, item) in map {
                    let key = self.s(key);
                    write!(out, "({key},").unwrap();
                    self.json(out, item);
                    out.push_str("),");
                }
                out.push_str("])");
            }
        }
    }

    fn module_properties(&mut self, out: &mut String, document: &Map<String, Value>) {
        out.push_str("pub(super) static MODULE_PROPERTIES: &[(u16, &[(u16, u16)])] = &[");
        for (module, members) in document {
            let id = self.s(module);
            write!(out, "({id},&[").unwrap();
            for (name, entry) in object(members, module) {
                let entry = object(entry, name);
                assert!(
                    entry.len() == 1,
                    "module_properties.json: {module}.{name} has more than a type"
                );
                let name = self.s(name);
                let ty = self.s(string(section(entry, "type"), "type"));
                write!(out, "({name},{ty}),").unwrap();
            }
            out.push_str("]),");
        }
        out.push_str("];\n");
    }

    fn lib_types(&mut self, out: &mut String, document: &Map<String, Value>) {
        out.push_str("pub(super) static STAR_EXPORTS: &[(u16, (u16, u16))] = &[");
        for (module, names) in object(section(document, "star_exports"), "star_exports") {
            let id = self.s(module);
            let names = self.list(&strings(names, module));
            write!(out, "({id},{names}),").unwrap();
        }
        out.push_str("];\n");

        let routes = self.pairs(object(section(document, "routes"), "routes"), "routes");
        writeln!(out, "pub(super) static ROUTES: &[(u16, u16)] = {routes};").unwrap();

        out.push_str("pub(super) static CLASSES: &[(u16, &[(u16, u16)])] = &[");
        for (name, fields) in object(section(document, "classes"), "classes") {
            let id = self.s(name);
            let fields = self.pairs(object(fields, name), name);
            write!(out, "({id},{fields}),").unwrap();
        }
        out.push_str("];\n");

        let scalars = section(document, "scalar_classes");
        let scalars = self.pairs(object(scalars, "scalar_classes"), "scalar_classes");
        writeln!(out, "pub(super) static SCALAR_CLASSES: &[(u16, u16)] = {scalars};").unwrap();

        out.push_str("pub(super) static NAMES: &[(u16, LibStatic)] = &[");
        for (name, entry) in object(section(document, "names"), "names") {
            assert!(entry.get("kind").and_then(Value::as_str).is_some(), "names.{name}.kind");
            let id = self.s(name);
            let entry = self.lib_entry(entry);
            write!(out, "({id},{entry}),").unwrap();
        }
        out.push_str("];\n");
    }

    fn inline_templates(&mut self, out: &mut String, document: &Map<String, Value>) {
        let functions = object(section(document, "functions"), "functions");
        out.push_str("pub(super) static INLINE_FUNCTIONS: &[(u16, InlineStatic)] = &[");
        for (name, entry) in functions {
            let id = self.s(name);
            write!(out, "({id},").unwrap();
            let entry = object(entry, name);
            match entry.get("kind").and_then(Value::as_str) {
                Some("derive") => {
                    let params = self.list(&strings(section(entry, "params"), name));
                    write!(out, "InlineStatic::Derive {{params: {params},body: &").unwrap();
                    self.json(out, section(entry, "body"));
                    out.push('}');
                }
                Some("varargs") => {
                    let mut field = |field: &str| self.s(string(section(entry, field), field));
                    let builtin = field("builtin");
                    let na_anchor = field("na_anchor");
                    let builtins_anchor = field("builtins_anchor");
                    write!(
                        out,
                        "InlineStatic::Varargs {{builtin: {builtin},na_anchor: {na_anchor},\
                         builtins_anchor: {builtins_anchor}}}"
                    )
                    .unwrap();
                }
                other => panic!("functions.{name}.kind {other:?}"),
            }
            out.push_str("),");
        }
        out.push_str("];\n");

        let callees = self.pairs(object(section(document, "callees"), "callees"), "callees");
        writeln!(out, "pub(super) static INLINE_CALLEES: &[(u16, u16)] = {callees};").unwrap();

        let loop_ranges = self.list(&strings(section(document, "loop_ranges"), "loop_ranges"));
        writeln!(out, "pub(super) static LOOP_RANGES: (u16, u16) = {loop_ranges};").unwrap();

        out.push_str("pub(super) static VARARGS_ARITIES: &[usize] = &[");
        for arity in section(document, "varargs_arities").as_array().expect("varargs_arities") {
            write!(out, "{},", arity.as_u64().expect("an arity")).unwrap();
        }
        out.push_str("];\n");
    }

    fn python_names(&mut self, out: &mut String, document: &Map<String, Value>) {
        for (name, table) in
            [("builtins", "BUILTINS"), ("keywords", "KEYWORDS"), ("stdlib", "STDLIB")]
        {
            let names = self.list(&strings(section(document, name), name));
            writeln!(out, "pub(super) static {table}: (u16, u16) = {names};").unwrap();
        }
        out.push_str("pub(super) static UNPRINTABLE: Unprintable = &[");
        for (version, ranges) in object(section(document, "unprintable"), "unprintable") {
            let (major, minor) = version.split_once('.').expect("major.minor");
            let major: u32 = major.parse().expect("major");
            let minor: u32 = minor.parse().expect("minor");
            write!(out, "(({major}, {minor}), &[").unwrap();
            for range in ranges.as_array().expect("ranges") {
                let [first, last] = range.as_array().expect("a range").as_slice() else {
                    panic!("a range of two")
                };
                let first = first.as_u64().expect("u32");
                let last = last.as_u64().expect("u32");
                write!(out, "({first}, {last}),").unwrap();
            }
            out.push_str("]),");
        }
        out.push_str("];\n");
    }

    /// The shared tables the others index.
    fn finish(self, out: &mut String) {
        writeln!(out, "pub(super) static TEXT: &str = {:?};", self.text).unwrap();
        out.push_str("pub(super) static ENDS: &[u32] = &[");
        for end in &self.ends {
            write!(out, "{end},").unwrap();
        }
        out.push_str("];\npub(super) static POOL: &[u16] = &[");
        for id in &self.pool {
            write!(out, "{id},").unwrap();
        }
        out.push_str("];\npub(super) static BOOLS: &[bool] = &[");
        for flag in &self.bools {
            write!(out, "{flag},").unwrap();
        }
        out.push_str("];\npub(super) static IMPLS: &[LibStatic] = &[");
        for entry in &self.impls {
            write!(out, "{entry},").unwrap();
        }
        out.push_str("];\n");
    }
}

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let data = crate_dir.join("../../../src/pynecore/transformers");
    let mut out = String::new();
    let mut tables = Gen::default();
    tables.module_properties(&mut out, &parse(&data.join("module_properties.json")));
    tables.lib_types(&mut out, &parse(&data.join("lib_types.json")));
    tables.inline_templates(&mut out, &parse(&data.join("call_inline_templates.json")));
    tables.python_names(&mut out, &parse(&data.join("python_names.json")));
    tables.finish(&mut out);
    let ported = read(&crate_dir.join("ported.json"));
    writeln!(
        out,
        "pub(crate) const PORTED_SHA256: &str = {:?};",
        sha256::sha256_hex(ported.as_bytes())
    )
    .unwrap();
    let dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(dir.join("data_tables.rs"), out).expect("write the data tables");
}
