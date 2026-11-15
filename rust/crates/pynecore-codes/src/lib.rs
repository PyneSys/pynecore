//! PyneCore's diagnostic codes: the errors that stop a script's run, the rules
//! of Pyne code and of the Pyne Edge profile, and PyneCore's internal error.
//!
//! The constants, [`ALL`] and [`RUNTIME`] are generated from
//! `codes/registry.toml` by `codes/generate.py`. A code carries what a
//! reported diagnostic needs; the long explanation and the introducing release
//! are in the generated docs. PyneComp's code crate re-exports these codes next
//! to the compiler's own, with this crate's types.
//!
//! `no_std` with `alloc`: the runtime crates a compiled script runs on name
//! their errors' codes with it ([`Raised`]).

#![no_std]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

mod registry;

pub use registry::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Code {
    /// A domain letter and four digits, such as `R0001`.
    pub code: &'static str,
    pub severity: Severity,
    pub title: &'static str,
    /// The message with `{param}` placeholders; `{{` and `}}` are literal braces.
    pub message: &'static str,
    /// The placeholders in order of first appearance: [`Code::render`]'s argument order.
    pub params: &'static [&'static str],
    /// The edit that removes the cause, when one edit does.
    pub fix: Option<Fix>,
}

/// A quick fix: replace the diagnostic's range with `replacement`. Both are
/// templates of the code's parameters.
#[derive(Debug, PartialEq, Eq)]
pub struct Fix {
    pub title: &'static str,
    pub replacement: &'static str,
}

impl Code {
    /// The message with `args[i]` in place of `params[i]`. A message whose
    /// last parameter is empty (`Assertion failed. {message}` of a bare
    /// `assert`) ends without the space before it.
    pub fn render(&self, args: &[&str]) -> String {
        debug_assert_eq!(
            args.len(),
            self.params.len(),
            "{}: one argument per parameter",
            self.code
        );
        let mut out = self.fill(self.message, args);
        out.truncate(out.trim_end().len());
        out
    }

    /// The quick fix's title and replacement for `args`.
    pub fn fix(&self, args: &[&str]) -> Option<(String, String)> {
        let fix = self.fix.as_ref()?;
        Some((self.fill(fix.title, args), self.fill(fix.replacement, args)))
    }

    fn fill(&self, template: &str, args: &[&str]) -> String {
        let mut out = String::with_capacity(template.len());
        let mut rest = template;
        while let Some(at) = rest.find(['{', '}']) {
            out.push_str(&rest[..at]);
            let tail = &rest[at..];
            if tail.starts_with("{{") || tail.starts_with("}}") {
                out.push_str(&tail[..1]);
                rest = &tail[2..];
                continue;
            }
            // The generator only lets a whole `{name}` through.
            let end = tail.find('}').unwrap_or(tail.len() - 1);
            let name = &tail[1..end];
            let index = self.params.iter().position(|p| *p == name);
            out.push_str(index.and_then(|i| args.get(i)).copied().unwrap_or(""));
            rest = &tail[end + 1..];
        }
        out.push_str(rest);
        out
    }
}

impl Code {
    /// The code as one number, as a compiled script raises it: the domain
    /// letter's ASCII value times 10000 plus the four digits (`R0010` is
    /// `820010`).
    pub const fn id(&self) -> i32 {
        match id_of(self.code) {
            Some(id) => id,
            None => panic!("a code is a domain letter and four digits"),
        }
    }
}

/// [`Code::id`] of the code spelled `code`, such as `R0010`; `None` when
/// `code` is not a domain letter and four digits. It reads the spelling
/// alone, so a caller holding only the code's text links none of the
/// registry.
pub const fn id_of(code: &str) -> Option<i32> {
    let b = code.as_bytes();
    if b.len() != 5 || !b[0].is_ascii_uppercase() {
        return None;
    }
    let mut n = 0;
    let mut i = 1;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            return None;
        }
        n = n * 10 + (b[i] - b'0') as i32;
        i += 1;
    }
    Some(b[0] as i32 * 10000 + n)
}

/// The run-time code ([`RUNTIME`]) a compiled script raises as `id`
/// ([`Code::id`]).
pub fn runtime(id: i32) -> Option<&'static Code> {
    runtime_in(RUNTIME, id)
}

/// The code of `table` (sorted by `code`, as [`RUNTIME`] is) whose
/// [`Code::id`] is `id`.
pub fn runtime_in(table: &[&'static Code], id: i32) -> Option<&'static Code> {
    table.binary_search_by(|c| c.id().cmp(&id)).ok().map(|i| table[i])
}

/// A raised code with the values of its parameters, in their order: the
/// error of a run, before the host says where it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Raised {
    pub code: &'static Code,
    pub args: Vec<String>,
}

impl Raised {
    #[cold]
    #[inline(never)]
    pub fn new<A: Into<String>>(code: &'static Code, args: impl IntoIterator<Item = A>) -> Raised {
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        debug_assert_eq!(
            args.len(),
            code.params.len(),
            "{}: one argument per parameter",
            code.code
        );
        Raised { code, args }
    }

    /// The code's message with the arguments filled in.
    pub fn message(&self) -> String {
        let args: Vec<&str> = self.args.iter().map(String::as_str).collect();
        self.code.render(&args)
    }
}

/// The reported code `code`, such as `R0001`.
pub fn lookup(code: &str) -> Option<&'static Code> {
    lookup_in(ALL, code)
}

/// The code of `table` (sorted by `code`, as [`ALL`] is) spelled `code`.
pub fn lookup_in(table: &[&'static Code], code: &str) -> Option<&'static Code> {
    table.binary_search_by(|c| c.code.cmp(code)).ok().map(|i| table[i])
}

/// A number as an error message shows it, the same way PyneCore's
/// `format_arg` does: `na`, a whole number without a fraction, any other in
/// its shortest exact form written out in full (`1e-7` is `0.0000001`). Both
/// run the vectors of `codes/format_arg.toml`.
pub fn format_arg(v: f64) -> String {
    if v.is_nan() {
        String::from("na")
    } else if v > -1e18 && v < 1e18 && (v as i64) as f64 == v {
        (v as i64).to_string()
    } else {
        v.to_string()
    }
}
