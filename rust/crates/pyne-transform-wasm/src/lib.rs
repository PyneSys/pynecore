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
//! "include_attributes": bool, "libraries": {module: [served name]},
//! "python": "major.minor"}`. A step name may also be a phase's (its rules in one
//! traversal, or one pass each with `sequential`) or one rule's. `libraries` stands
//! in for the Python host's imports: a library is importable when it is listed, and
//! serves the names listed for it. `python` names the interpreter whose `repr` and
//! `ast.unparse` the output follows, the newest one known when left out.
//! Response: `{"dump": str}` (the `ast.dump` of the transformed module), or
//! `{"rejected": str}` (the module is invalid), `{"unsupported": str}` (the
//! native port does not cover it) or `{"error": str}` (a malformed request).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use pyne_ast::Unsupported;
use pyne_transform::slot_layout::ModuleLayout;
use serde_json::{Value, json};

/// The host of a request: what the browser was handed instead of an interpreter.
struct RequestHost {
    libraries: HashMap<String, HashSet<String>>,
    python: (u32, u32),
}

impl pyne_transform::Host for RequestHost {
    fn python(&self) -> (u32, u32) {
        self.python
    }

    fn import_library(&self, module: &str) -> Result<bool, Unsupported> {
        Ok(self.libraries.contains_key(module))
    }

    fn library_serves(&self, module: &str, attr: &str) -> Result<bool, Unsupported> {
        Ok(self.libraries.get(module).is_some_and(|names| names.contains(attr)))
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

fn transform(request: &str) -> Result<Value, Value> {
    let request: Value =
        serde_json::from_str(request).map_err(|err| json!({ "error": err.to_string() }))?;
    let text = |name: &str| request.get(name).and_then(Value::as_str).map(str::to_string);
    let source = text("source").ok_or_else(|| json!({ "error": "no source" }))?;
    let libraries = request
        .get("libraries")
        .and_then(Value::as_object)
        .map(|libraries| {
            libraries
                .iter()
                .map(|(module, names)| {
                    let names = names
                        .as_array()
                        .map(|names| {
                            names.iter().filter_map(Value::as_str).map(str::to_string).collect()
                        })
                        .unwrap_or_default();
                    (module.clone(), names)
                })
                .collect()
        })
        .unwrap_or_default();
    let python = match text("python") {
        None => pyne_transform::data::newest_python(),
        Some(version) => version
            .split_once('.')
            .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
            .ok_or_else(|| json!({ "error": format!("python {version:?} is no major.minor") }))?,
    };
    let host = RequestHost { libraries, python };
    let ctx = pyne_transform::Context {
        path: text("path").unwrap_or_default(),
        pyne_mode: text("pyne_mode"),
        source: source.clone(),
        user_code: request.get("user_code").and_then(Value::as_bool).unwrap_or(true),
        na_bool: request.get("na_bool").and_then(Value::as_bool).unwrap_or(false),
        sequential: request.get("sequential").and_then(Value::as_bool).unwrap_or(false),
        security_slice_disabled: request
            .get("security_slice_disabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        emit_layout: request.get("emit_layout").and_then(Value::as_bool).unwrap_or(true),
        host: &host,
        // A lib module's series are na-compacted windows, as the import hook sets it
        slot_layout: RefCell::new(ModuleLayout::new(text("pyne_mode").as_deref() == Some("lib"))),
    };
    let steps: Vec<&str> = request
        .get("steps")
        .and_then(Value::as_array)
        .map(|steps| steps.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let include_attributes =
        request.get("include_attributes").and_then(Value::as_bool).unwrap_or(false);
    let unsupported = |err: Unsupported| json!({ "unsupported": err.0 });
    let (mut ast, module) = pyne_parse::parse_module(&source).map_err(unsupported)?;
    let module =
        pyne_transform::run_steps(&mut ast, module, &steps, &ctx).map_err(|err| match err {
            pyne_transform::Error::Unsupported(message) => json!({ "unsupported": message }),
            pyne_transform::Error::Rejected(message) => json!({ "rejected": message }),
        })?;
    Ok(json!({ "dump": pyne_ast::dump::dump(&ast, module, include_attributes) }))
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
