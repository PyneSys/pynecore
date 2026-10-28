//! The native transform as a WebAssembly module, for hosts without Python.
//!
//! The interface is plain C ABI over linear memory, no bindings generator: the
//! host allocates a buffer with `pyne_alloc`, writes a UTF-8 JSON request into
//! it, calls `pyne_transform`, and reads the JSON response from the pointer and
//! length packed into the returned `u64` (pointer in the high 32 bits), then
//! hands both buffers back with `pyne_free`.
//!
//! Request: `{"source": str, "path": str, "pyne_mode": str | null,
//! "user_code": bool, "na_bool": bool, "sequential": bool,
//! "security_slice_disabled": bool, "emit_layout": bool, "steps": [str],
//! "include_attributes": bool, "python": "major.minor", "analyse": bool,
//! "package": str, "modules": {dotted: path | null},
//! "origins": [[name, importer, level, path | null]],
//! "files": {path: null | {"resolved": str, "mtime_ns": int, "size": int,
//! "source": str | null}}}`.
//!
//! A step name may also be a phase's (its rules in one traversal, or one pass each
//! with `sequential`) or one rule's. `python` names the interpreter whose `repr`
//! and `ast.unparse` the output follows, the newest one known when left out.
//!
//! The rest is what the modules around this one look like (see
//! `pyne_transform::resolver`): `modules` answers where a dotted name's source is,
//! `origins` the enum capture's own module search, `files` every file the transform
//! reads or tests for (`null`: there is no such file; a `null` source: it cannot be
//! read), and `package` is the directory of the `pynecore` package. A Pyne
//! dependency is transformed natively into the interface it publishes, unless
//! `analyse` is false; any other module the transform routes a call into is read
//! for what it binds. No module is run.
//!
//! Response: `{"dump": str}` (the `ast.dump` of the transformed module, as the
//! interpreter `python` names writes it),
//! `{"rejected": str}` (the module is invalid), `{"unsupported": str}` (the
//! native port does not cover it), `{"error": str}` (a malformed request), or
//! `{"needs": [need]}` when the request left a question about the modules around it
//! open: `{"module": dotted}`, `{"origin": [name, importer, level]}` or
//! `{"file": path}`. The transform runs on
//! as if each were answered "none" to find every one it reaches, so a host answers
//! them all and sends the request again.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pyne_ast::Unsupported;
use pyne_transform::resolver::{File, Files, Options, Resolver};
use pyne_transform::slot_layout::ModuleLayout;
use serde_json::{Value, json};

type R<T> = Result<T, Unsupported>;

/// What the request says about the modules around the one it transforms.
struct RequestFiles {
    python: (u32, u32),
    package: String,
    modules: HashMap<String, Option<String>>,
    origins: Vec<(String, String, i64, Option<String>)>,
    files: HashMap<String, Option<File>>,
    /// The questions the request leaves open, in the order they were asked
    needs: RefCell<Vec<Value>>,
}

impl RequestFiles {
    fn need(&self, need: Value) {
        let mut needs = self.needs.borrow_mut();
        if !needs.contains(&need) {
            needs.push(need);
        }
    }
}

impl Files for RequestFiles {
    fn python(&self) -> (u32, u32) {
        self.python
    }

    fn package(&self) -> &str {
        &self.package
    }

    fn spec_origin(&self, dotted: &str) -> R<Option<String>> {
        match self.modules.get(dotted) {
            Some(path) => Ok(path.clone()),
            None => {
                self.need(json!({ "module": dotted }));
                Ok(None)
            }
        }
    }

    fn module_origin(&self, name: &str, importer: &str, level: i64) -> R<Option<String>> {
        let found = self
            .origins
            .iter()
            .find(|(known, from, at, _)| known == name && from == importer && *at == level);
        match found {
            Some((.., path)) => Ok(path.clone()),
            None => {
                self.need(json!({ "origin": [name, importer, level] }));
                Ok(None)
            }
        }
    }

    fn file(&self, path: &str) -> R<Option<File>> {
        match self.files.get(path) {
            Some(file) => Ok(file.clone()),
            None => {
                self.need(json!({ "file": path }));
                Ok(None)
            }
        }
    }
}

/// Parse, run the requested steps, dump: the whole request in one call.
pub fn transform_json(request: &str) -> String {
    let response = match transform(request) {
        Ok(response) => response,
        Err(error) => error,
    };
    response.to_string()
}

fn malformed(what: &str) -> Value {
    json!({ "error": format!("malformed {what}") })
}

fn optional_path(value: &Value) -> Result<Option<String>, Value> {
    match value {
        Value::Null => Ok(None),
        Value::String(path) => Ok(Some(path.clone())),
        _ => Err(malformed("path")),
    }
}

fn request_files(request: &Value, python: (u32, u32)) -> Result<RequestFiles, Value> {
    let object = |name: &str| request.get(name).and_then(Value::as_object);
    let list = |name: &str| request.get(name).and_then(Value::as_array);
    let mut modules = HashMap::new();
    for (dotted, path) in object("modules").into_iter().flatten() {
        modules.insert(dotted.clone(), optional_path(path)?);
    }
    let mut origins = Vec::new();
    for entry in list("origins").into_iter().flatten() {
        let entry =
            entry.as_array().filter(|entry| entry.len() == 4).ok_or_else(|| malformed("origin"))?;
        let name = entry[0].as_str().ok_or_else(|| malformed("origin name"))?;
        let importer = entry[1].as_str().ok_or_else(|| malformed("origin importer"))?;
        let level = entry[2].as_i64().ok_or_else(|| malformed("origin level"))?;
        origins.push((name.to_string(), importer.to_string(), level, optional_path(&entry[3])?));
    }
    let mut files = HashMap::new();
    for (path, file) in object("files").into_iter().flatten() {
        let file = match file {
            Value::Null => None,
            file => {
                let int = |name: &str| {
                    file.get(name).and_then(Value::as_i64).ok_or_else(|| malformed(name))
                };
                let resolved = file.get("resolved").and_then(Value::as_str).unwrap_or(path);
                let data = match file.get("source") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(source)) => Some(Rc::from(source.as_bytes())),
                    Some(_) => return Err(malformed("source")),
                };
                Some(File {
                    resolved: resolved.to_string(),
                    mtime_ns: int("mtime_ns")?,
                    size: int("size")?,
                    data,
                })
            }
        };
        files.insert(path.clone(), file);
    }
    Ok(RequestFiles {
        python,
        package: request.get("package").and_then(Value::as_str).unwrap_or_default().to_string(),
        modules,
        origins,
        files,
        needs: RefCell::new(Vec::new()),
    })
}

fn transform(request: &str) -> Result<Value, Value> {
    let request: Value =
        serde_json::from_str(request).map_err(|err| json!({ "error": err.to_string() }))?;
    let text = |name: &str| request.get(name).and_then(Value::as_str).map(str::to_string);
    let flag =
        |name: &str, default: bool| request.get(name).and_then(Value::as_bool).unwrap_or(default);
    let source = text("source").ok_or_else(|| json!({ "error": "no source" }))?;
    let python = match text("python") {
        None => pyne_transform::data::newest_python(),
        Some(version) => version
            .split_once('.')
            .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
            .ok_or_else(|| json!({ "error": format!("python {version:?} is no major.minor") }))?,
    };
    let options = Options {
        analyse: flag("analyse", true),
        sequential: flag("sequential", false),
        security_slice_disabled: flag("security_slice_disabled", false),
    };
    let host = Resolver::new(request_files(&request, python)?, options);
    let ctx = pyne_transform::Context {
        path: text("path").unwrap_or_default(),
        pyne_mode: text("pyne_mode"),
        source: source.clone(),
        user_code: flag("user_code", true),
        na_bool: flag("na_bool", false),
        sequential: options.sequential,
        security_slice_disabled: options.security_slice_disabled,
        emit_layout: flag("emit_layout", true),
        host: &host,
        // A lib module's series are na-compacted windows, as the import hook sets it
        slot_layout: RefCell::new(ModuleLayout::new(text("pyne_mode").as_deref() == Some("lib"))),
    };
    let steps: Vec<&str> = request
        .get("steps")
        .and_then(Value::as_array)
        .map(|steps| steps.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let include_attributes = flag("include_attributes", false);
    let parsed = pyne_parse::parse_module_as(&source, python);
    let outcome = parsed.map_err(pyne_transform::Error::from).and_then(|(mut ast, module)| {
        let module = pyne_transform::run_steps(&mut ast, module, &steps, &ctx)?;
        Ok(pyne_transform::ast_dump::dump(&ast, module, include_attributes, python)?)
    });
    let needs = host.files.needs.take();
    if !needs.is_empty() {
        return Ok(json!({ "needs": needs }));
    }
    match outcome {
        Ok(dump) => Ok(json!({ "dump": dump })),
        Err(pyne_transform::Error::Unsupported(message)) => Err(json!({ "unsupported": message })),
        Err(pyne_transform::Error::Rejected(message)) => Err(json!({ "rejected": message })),
    }
}

/// A buffer of `len` bytes for the host to write into.
#[unsafe(no_mangle)]
pub extern "C" fn pyne_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Give back a buffer from `pyne_alloc` or a response of `pyne_transform`.
///
/// # Safety
///
/// `ptr` and `len` must be exactly one such buffer, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pyne_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

/// Run one JSON request; the response's pointer and length are packed into the result.
///
/// # Safety
///
/// `ptr` must point to `len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pyne_transform(ptr: *const u8, len: usize) -> u64 {
    let request = unsafe { std::slice::from_raw_parts(ptr, len) };
    let response = match std::str::from_utf8(request) {
        Ok(request) => transform_json(request),
        Err(err) => json!({ "error": err.to_string() }).to_string(),
    };
    let mut bytes = response.into_bytes();
    bytes.shrink_to_fit();
    let (out_ptr, out_len) = (bytes.as_mut_ptr(), bytes.len());
    std::mem::forget(bytes);
    ((out_ptr as u64) << 32) | out_len as u64
}
