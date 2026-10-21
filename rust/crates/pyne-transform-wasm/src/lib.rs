//! The native transform as a WebAssembly module, for hosts without Python.
//!
//! The interface is plain C ABI over linear memory, no bindings generator: the
//! host allocates a buffer with `pyne_alloc`, writes a UTF-8 JSON request into
//! it, calls `pyne_transform`, and reads the JSON response from the pointer and
//! length packed into the returned `u64` (pointer in the high 32 bits), then
//! hands both buffers back with `pyne_free`.
//!
//! Request: `{"source": str, "path": str, "pyne_mode": str | null,
//! "user_code": bool, "steps": [str], "include_attributes": bool}`.
//! Response: `{"dump": str}` (the `ast.dump` of the transformed module), or
//! `{"unsupported": str}`, or `{"error": str}`.

use serde_json::{Value, json};

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
    let ctx = pyne_transform::Context {
        path: text("path").unwrap_or_default(),
        pyne_mode: text("pyne_mode"),
        source: source.clone(),
        user_code: request.get("user_code").and_then(Value::as_bool).unwrap_or(true),
    };
    let steps: Vec<&str> = request
        .get("steps")
        .and_then(Value::as_array)
        .map(|steps| steps.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let include_attributes =
        request.get("include_attributes").and_then(Value::as_bool).unwrap_or(false);
    let unsupported = |err: pyne_ast::Unsupported| json!({ "unsupported": err.0 });
    let (mut ast, module) = pyne_parse::parse_module(&source).map_err(unsupported)?;
    let module = pyne_transform::run_steps(&mut ast, module, &steps, &ctx).map_err(unsupported)?;
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
