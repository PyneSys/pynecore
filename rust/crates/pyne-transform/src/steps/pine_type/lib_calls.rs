//! Calls into the lib registry: their shape, their type through the measured
//! override forms, and the overload pins and implicit conversions they decide.

use std::collections::HashSet;

use indexmap::IndexMap;
use pyne_ast::{Kind, NodeId, Unsupported};

use super::facts::{
    self, HLINE, WIDEN_HARMLESS, bound_arg, call_arity, fits, holds_na, keyword_parts, unchecked,
};
use super::{Engine, R};
use crate::data::{self, LibEntry};
use crate::pine_qualifier::{SERIES, SERIES_LENGTH_KEYWORD, WINDOW_CALLS, window_length};
use crate::pine_type_rules::{
    BOOL, CONTAINER_NA_WIDENS, CallRule, FIT_REQUIRED, FLOAT, INT, ImplSig, NUMERIC, OBJECT,
    PIN_ANY, SCALARS, STR, TYPELESS, UNKNOWN, VOID, WIDEN_ARRAY_NA, WIDEN_BOOL, WIDEN_FLOAT,
    WIDEN_MAP_NA, WIDEN_NA_TEXT, WIDEN_NUMBER_NA, WIDEN_STRING_NA, array_of, class_of, default_fit,
    element_of, get_ty, is_array, is_map, is_matrix, is_na_text_param, is_na_widen,
    is_not_pine_string, join, key_of, lib_type_override, matrix_of, object_ty, overload_binding,
    overload_choice, overload_result, override_param_names, pin_chars, pin_for, render_ty,
    set_na_class, set_pin, set_pins, set_widen, shape_conflict, value_of,
};
use crate::pine_type_table::{CallSite, Diag, Unknown};

/// Constructors whose element type comes from the values they are built from
/// (`_CONSTRUCTOR_ELEMENTS`); empty is every positional argument.
fn constructor_elements(callee: &str) -> Option<&'static [usize]> {
    match callee {
        "array.new" => Some(&[1]),
        "matrix.new" => Some(&[2]),
        "array.from_items" => Some(&[]),
        _ => None,
    }
}

/// Constructors that fill the container they build with one value (`_FILL_POSITIONS`).
fn fill_positions(callee: &str) -> Option<&'static [usize]> {
    match callee {
        "array.new" | "array.new_float" | "array.new_int" => Some(&[1]),
        "matrix.new" => Some(&[2]),
        _ => None,
    }
}

/// The map calls that look a key up (`_MAP_KEY_LOOKUPS`).
const MAP_KEY_LOOKUPS: &[&str] = &["map.get", "map.contains", "map.remove"];

/// The shape-reading override forms, in `_SHAPE_FORMS`' order.
const SHAPE_FORMS: &[&str] = &[
    "array_of_map_keys",
    "array_of_map_values",
    "array_of_elem",
    "array_of_arg",
    "matrix_of_arg",
    "map_key",
    "map_value",
    "elem",
    "same_array",
    "same_matrix",
    "same_map",
];

fn shape_form(form: &str, ty: &str) -> String {
    match form {
        "array_of_map_keys" => array_of(key_of(ty)),
        "array_of_map_values" => array_of(value_of(ty)),
        "array_of_elem" => array_of(element_of(ty)),
        "array_of_arg" => array_of(ty),
        "matrix_of_arg" => matrix_of(ty),
        "map_key" => key_of(ty).to_string(),
        "map_value" => value_of(ty).to_string(),
        "elem" => element_of(ty).to_string(),
        "same_array" => if is_array(ty) { ty } else { UNKNOWN }.to_string(),
        "same_matrix" => if is_matrix(ty) { ty } else { UNKNOWN }.to_string(),
        _ => if is_map(ty) { ty } else { UNKNOWN }.to_string(),
    }
}

/// Whether a text is a non-empty run of decimal digits (`str.isdigit`, ASCII).
fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// Whether an overload implementation can take this many positional arguments
/// (`_arity_fits`).
pub(super) fn arity_fits(entry: &LibEntry, argc: i64) -> bool {
    if entry.vararg {
        return true;
    }
    let params = entry.params.as_ref().map_or(0, Vec::len) as i64;
    let required = params - entry.defaults;
    if entry.kwarg {
        return required <= argc;
    }
    required <= argc && argc <= params
}

/// Whether a lib parameter may take an argument (`_lib_param_takes`).
fn lib_param_takes(declared: &str, passed: &str) -> bool {
    if unchecked(passed) {
        return true;
    }
    if SCALARS.contains(&declared) {
        return fits(declared, passed);
    }
    if declared.starts_with("o:") {
        return !SCALARS.contains(&passed);
    }
    true
}

/// One registry implementation, as the static selection reads it (`_lib_impl_sig`).
pub(super) fn lib_impl_sig(entry: &LibEntry) -> ImplSig {
    let params: Vec<String> = entry.params.clone().unwrap_or_default();
    let required = (params.len() as i64 - entry.defaults).max(0) as usize;
    let none_ok = entry
        .default_none_ok
        .clone()
        .filter(|flags| !flags.is_empty())
        .unwrap_or_else(|| vec![false; entry.default_ty.len()]);
    let mut fits: String = std::iter::repeat_n(FIT_REQUIRED, required).collect();
    for ((declared, ty), takes_none) in
        params.iter().skip(required).zip(&entry.default_ty).zip(none_ok)
    {
        fits.push(default_fit(declared, Some(ty), takes_none));
    }
    ImplSig {
        params,
        required,
        open_ended: entry.vararg,
        ret: entry.ret.clone().unwrap_or_else(|| UNKNOWN.to_string()),
        fits,
        names: entry.names.clone().unwrap_or_default(),
    }
}

impl Engine<'_, '_> {
    /// Type a call to a lib name and record its call site (`_lib_call`).
    pub(super) fn lib_call(&mut self, node: NodeId, callee: &str) -> R<String> {
        let argc = call_arity(self.ast, node);
        let entry = data::lib_entries().get(callee);
        if entry.is_none() && lib_type_override(callee).is_none() {
            if data::lib_namespaces().contains(callee) {
                self.node_diag(
                    &format!("'{callee}' is a lib namespace, not a function"),
                    node,
                    "unknown-lib",
                    callee,
                    &format!("call a name in '{callee}'"),
                );
            } else {
                self.node_diag(
                    &format!("'{callee}' is not a lib name"),
                    node,
                    "unknown-lib",
                    callee,
                    "call a lib name that exists",
                );
            }
        }
        let (line, col) = (self.line(node), self.col(node));
        if let Some(entry) = entry
            && argc.is_some()
            && !self.lib_call_fits(callee, entry, node, None)?
        {
            self.calls.push(CallSite {
                callee: callee.to_string(),
                line,
                col,
                argc,
                ty: UNKNOWN.to_string(),
                pin: None,
            });
            return Ok(UNKNOWN.to_string());
        }
        let is_group = entry.is_some_and(|entry| entry.kind == "overloads");
        let pin = self.pin(node, is_group)?;
        let (_, args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let (args, keywords) = (args.to_vec(), keywords.to_vec());
        let ty = self.lib_call_type(callee, &args, &keywords, argc, pin.as_deref())?;
        self.calls.push(CallSite {
            callee: callee.to_string(),
            line,
            col,
            argc,
            ty: ty.clone(),
            pin: pin.clone(),
        });
        if self.qualify_windows && WINDOW_CALLS.contains(&callee) {
            self.record_window(node, callee);
        }
        if let Some(entry) = entry {
            self.record_lib_na_args(callee, entry, &args, &keywords)?;
        }
        if let Some(positions) = fill_positions(callee)
            && (is_array(&ty) || is_matrix(&ty))
            && NUMERIC.contains(&element_of(&ty))
        {
            for arg in self.element_args(callee, node, positions) {
                if self.ty_of(arg) == TYPELESS {
                    self.record_widen(arg, Some(WIDEN_NUMBER_NA.to_string()))?;
                }
            }
        }
        Ok(ty)
    }

    /// Mark the typeless `na` literals a lib call passes to its string parameters
    /// (`_record_lib_na_args`).
    pub(super) fn record_lib_na_args(
        &mut self,
        callee: &str,
        entry: &LibEntry,
        args: &[NodeId],
        keywords: &[NodeId],
    ) -> R<()> {
        for (name, declared, arg) in self.lib_bound_args(callee, entry, args, keywords) {
            if self.ty_of(arg) != TYPELESS {
                continue;
            }
            match name {
                Some(name) if is_na_text_param(callee, &name) => {
                    self.record_widen(arg, Some(WIDEN_NA_TEXT.to_string()))?;
                }
                name if declared == STR
                    && !name.as_deref().is_some_and(|name| is_not_pine_string(callee, name)) =>
                {
                    self.record_widen(arg, Some(WIDEN_STRING_NA.to_string()))?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Bind a lib call's arguments to the callee's parameters (`_lib_bound_args`).
    fn lib_bound_args(
        &self,
        callee: &str,
        entry: &LibEntry,
        args: &[NodeId],
        keywords: &[NodeId],
    ) -> Vec<(Option<String>, String, NodeId)> {
        let positional: Vec<NodeId> =
            args.iter().copied().filter(|&arg| !facts::is_starred(self.ast, arg)).collect();
        if positional.len() != args.len() {
            return Vec::new();
        }
        let impls: Vec<&LibEntry> =
            match entry.kind.as_str() {
                "function" => vec![entry],
                "overloads" => {
                    let argc = (positional.len() + keywords.len()) as i64;
                    entry
                        .impls
                        .iter()
                        .filter(|imp| {
                            arity_fits(imp, argc)
                                && imp.params.as_deref().unwrap_or(&[]).iter().zip(&positional).all(
                                    |(declared, &arg)| lib_param_takes(declared, &self.ty_of(arg)),
                                )
                        })
                        .collect()
                }
                _ => return Vec::new(),
            };
        let mut bound = Vec::new();
        for (index, &arg) in positional.iter().enumerate() {
            if callee == "fill" || callee == "alertcondition" {
                let name = if callee == "fill" {
                    self.fill_arg_name(&positional, index)
                } else {
                    ["condition", "title", "message"].get(index).map(|name| name.to_string())
                };
                bound.push((name, UNKNOWN.to_string(), arg));
                continue;
            }
            let mut seen: Vec<(String, String)> = Vec::new();
            for imp in &impls {
                let names = imp.names.as_deref().unwrap_or(&[]);
                let params = imp.params.as_deref().unwrap_or(&[]);
                if index < names.len() {
                    let pair =
                        (names[index].clone(), params.get(index).cloned().unwrap_or_default());
                    if !seen.contains(&pair) {
                        seen.push(pair);
                    }
                }
            }
            if seen.len() == 1 {
                let (name, declared) = seen.remove(0);
                bound.push((Some(name), declared, arg));
            } else {
                bound.push((None, UNKNOWN.to_string(), arg));
            }
        }
        for &keyword in keywords {
            let (name, value) = keyword_parts(self.ast, keyword);
            let mut seen: Vec<String> = Vec::new();
            if let Some(name) = name {
                for imp in &impls {
                    let names = imp.names.as_deref().unwrap_or(&[]);
                    if let Some(at) = names.iter().position(|known| known == name) {
                        let declared = imp.params.as_deref().unwrap_or(&[]).get(at).cloned();
                        let declared = declared.unwrap_or_default();
                        if !seen.contains(&declared) {
                            seen.push(declared);
                        }
                    }
                }
            }
            let declared = if seen.len() == 1 { seen.remove(0) } else { UNKNOWN.to_string() };
            bound.push((name.map(str::to_string), declared, value));
        }
        bound
    }

    /// The parameter a positional `fill` argument binds to (`_fill_arg_name`).
    fn fill_arg_name(&self, positional: &[NodeId], index: usize) -> Option<String> {
        let first = positional.first().map_or(VOID.to_string(), |&arg| self.ty_of(arg));
        if first == HLINE {
            return (index == 3).then(|| "title".to_string());
        }
        let third = positional.get(2).map_or(VOID.to_string(), |&arg| self.ty_of(arg));

        let gradient = if third == TYPELESS && positional.len() > 4 {
            let fifth = self.ty_of(positional[4]);
            if fifth != TYPELESS && unchecked(&fifth) {
                return None;
            }
            fifth != BOOL
        } else if third == UNKNOWN || third == OBJECT {
            return None;
        } else {
            NUMERIC.contains(&third.as_str())
        };
        if gradient && unchecked(&first) {
            return None;
        }
        (index == if gradient { 6 } else { 3 }).then(|| "title".to_string())
    }

    /// Record whether a window call's length is series in the running context
    /// (`_record_window`).
    fn record_window(&mut self, node: NodeId, callee: &str) {
        let Some((_, _, keywords)) = facts::call_parts(self.ast, node) else { return };
        if keywords
            .iter()
            .any(|&keyword| keyword_parts(self.ast, keyword).0 == Some(SERIES_LENGTH_KEYWORD))
        {
            return;
        }
        let length = window_length(self.ast, callee, node);
        let (Some(length), Some(nid)) = (length, self.nid(node)) else { return };
        let series = self.qual(length) == SERIES;
        let context = self.context();
        self.series_len.entry(nid).or_default().insert(context, series);
        self.window_nodes.insert(nid, node);
    }

    /// Whether a lib call has the shape and the argument types its callee takes
    /// (`_lib_call_fits`).
    pub(super) fn lib_call_fits(
        &mut self,
        callee: &str,
        entry: &LibEntry,
        node: NodeId,
        args: Option<&[NodeId]>,
    ) -> R<bool> {
        let (_, own_args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let positional_args: Vec<NodeId> = args.unwrap_or(own_args).to_vec();
        let keywords = keywords.to_vec();
        let single = [entry.clone()];
        let impls: &[LibEntry] = match entry.kind.as_str() {
            "overloads" => &entry.impls,
            "function" => &single,
            "value"
                if entry.ty.as_deref().is_some_and(|ty| SCALARS.contains(&ty))
                    && !entry.callable =>
            {
                self.node_diag(
                    &format!("'{callee}' is a lib value, not a function"),
                    node,
                    "unknown-lib",
                    callee,
                    &format!("read '{callee}' without calling it"),
                );
                return Ok(false);
            }
            _ => return Ok(true),
        };
        let argc = (positional_args.len() + keywords.len()) as i64;
        if !impls.iter().any(|imp| arity_fits(imp, argc)) {
            let mut counts: Vec<i64> = impls
                .iter()
                .map(|imp| imp.params.as_ref().map_or(0, Vec::len) as i64 - imp.defaults)
                .collect();
            counts.sort();
            let least = counts.first().copied().unwrap_or(0);
            self.bad_call(
                node,
                callee,
                &format!("'{callee}' does not take {argc} argument(s) (at least {least} needed)"),
            );
            return Ok(false);
        }
        for &keyword in &keywords {
            let name = keyword_parts(self.ast, keyword).0;
            let taken = impls.iter().any(|imp| {
                name.is_some_and(|name| {
                    imp.names.as_deref().unwrap_or(&[]).iter().any(|n| n == name)
                }) || imp.kwarg
            });
            if !taken {
                let spelled = name.unwrap_or("None").to_string();
                self.bad_call(node, callee, &format!("'{callee}' has no parameter '{spelled}'"));
                return Ok(false);
            }
        }
        if entry.kind != "function" {
            return Ok(true);
        }
        let params = entry.params.clone().unwrap_or_default();
        let names = entry.names.clone().unwrap_or_default();
        let mut given: Vec<(String, String, NodeId)> = Vec::new();
        for (index, &arg) in positional_args.iter().take(params.len()).enumerate() {
            let name = names.get(index).cloned().unwrap_or_else(|| index.to_string());
            given.push((name, params[index].clone(), arg));
        }
        for &keyword in &keywords {
            let (name, value) = keyword_parts(self.ast, keyword);
            if let Some(name) = name
                && let Some(at) = names.iter().position(|known| known == name)
            {
                given.push((name.to_string(), params[at].clone(), value));
            }
        }
        for (pname, declared, arg) in given {
            if !SCALARS.contains(&declared.as_str()) {
                continue;
            }
            let passed = get_ty(self.ast, arg).to_string();
            if declared == BOOL {
                // A number goes into a bool parameter by Pine's tolerant
                // float-to-bool rule; the lib function itself tests the value with
                // Python's truthiness, which takes a nan for true
                let numeric = NUMERIC.contains(&passed.as_str());
                let kind = if numeric { WIDEN_BOOL } else { WIDEN_HARMLESS };
                self.record_widen(arg, Some(kind.to_string()))?;
                if numeric {
                    continue;
                }
            }
            if unchecked(&passed) || fits(&declared, &passed) {
                continue;
            }
            self.bad_call(
                node,
                callee,
                &format!(
                    "'{callee}' takes {} for '{pname}', {} passed",
                    render_ty(&declared),
                    render_ty(&passed)
                ),
            );
            return Ok(false);
        }
        Ok(true)
    }

    /// The overload pin this call site justifies in the running context (`_pin`).
    pub(super) fn pin(&mut self, node: NodeId, is_group: bool) -> R<Option<String>> {
        let mut pin = None;
        if let Some((_, args, keywords)) = facts::call_parts(self.ast, node)
            && is_group
            && keywords.is_empty()
            && !args.iter().any(|&arg| facts::is_starred(self.ast, arg))
        {
            let types: Vec<String> = args.iter().map(|&arg| self.ty_of(arg)).collect();
            pin = pin_for(&types);
        }
        self.record_pin(node, pin)
    }

    /// Merge this context's verdict into the node's pin and stamp both forms
    /// (`_record_pin`).
    pub(super) fn record_pin(&mut self, node: NodeId, pin: Option<String>) -> R<Option<String>> {
        let Some(nid) = self.nid(node) else {
            set_pin(self.ast, node, pin.as_deref());
            set_pins(self.ast, node, None);
            return Ok(pin);
        };
        let at = (self.line(node), self.col(node));
        self.call_pos.insert(nid, at);
        let context = self.context();
        let seen = self.pins.entry(nid).or_default();
        seen.insert(context, pin.clone());
        let seen = seen.clone();
        if let Some(Some(sink)) = self.pin_sink.last() {
            self.results[*sink].pins.insert(nid, pin.clone());
        }
        let distinct: HashSet<&Option<String>> = seen.values().collect();
        let agreed = distinct.len() == 1;
        set_pin(self.ast, node, if agreed { pin.as_deref() } else { None });
        let pairs: Vec<(i64, Option<String>)> =
            seen.iter().map(|(&cid, pin)| (cid, pin.clone())).collect();
        set_pins(self.ast, node, if agreed { None } else { Some(&pairs) });
        if !agreed && self.pin_diags.insert(nid) {
            let mut values: Vec<&Option<String>> = distinct.into_iter().collect();
            values.sort_by_key(|value| match value {
                None => "None".to_string(),
                Some(text) => text.clone(),
            });
            let mut rendered = Vec::new();
            for value in values {
                rendered.push(match value {
                    None => "None".to_string(),
                    Some(text) => crate::unparse::str_repr(text, self.host.python())?,
                });
            }
            let func = facts::call_parts(self.ast, node).map(|(func, _, _)| func);
            let spelled = func.and_then(|func| self.dotted(func)).unwrap_or_default();
            let origin = self.unknown("context-dependent-pin", node, &spelled);
            self.diag(
                &format!("overload pin differs per call context: [{}]", rendered.join(", ")),
                node,
                Some(origin),
                "annotate the parameters so every call site agrees",
            );
        }
        Ok(pin)
    }

    /// Merge this context's conversion verdict for a value into its stamp
    /// (`_record_widen`).
    pub(super) fn record_widen(&mut self, node: NodeId, kind: Option<String>) -> R<()> {
        let mut kind = kind;
        if is_na_widen(kind.as_deref()) && !holds_na(self.ast, node) {
            kind = None;
        }
        if let Some(found) = kind.clone()
            && (class_of(&found).is_some() || CONTAINER_NA_WIDENS.contains(&found.as_str()))
        {
            let spelled = self.na_spelling(&found);
            match spelled {
                Some(spelled)
                    if !self.hides_module_name(spelled.split('.').next().unwrap_or("")) =>
                {
                    set_na_class(self.ast, node, &spelled);
                }
                _ => kind = None,
            }
        }
        let Some(nid) = self.nid(node) else {
            let stamped = kind.filter(|kind| kind != WIDEN_HARMLESS);
            set_widen(self.ast, node, stamped.as_deref());
            return Ok(());
        };
        let context = self.context();
        let seen = self.widens.entry(nid).or_default();
        seen.insert(context, kind);
        let mut needed: Vec<&Option<String>> = Vec::new();
        for value in seen.values() {
            if value.as_deref() != Some(WIDEN_HARMLESS) && !needed.contains(&value) {
                needed.push(value);
            }
        }
        let stamped = if needed.len() == 1 { needed[0].clone() } else { None };
        self.widen_nodes.insert(nid, node);
        set_widen(self.ast, node, stamped.as_deref());
        Ok(())
    }

    /// Whether a scope the walk stands in binds a name, hiding the module's
    /// (`_hides_module_name`).
    fn hides_module_name(&self, name: &str) -> bool {
        self.stack[1..].iter().any(|&frame| {
            self.frames[frame].names.contains_key(name)
                || self.frames[frame].local_names.contains(name)
        })
    }

    /// How this module spells the class whose na a conversion builds (`_na_spelling`).
    fn na_spelling(&self, kind: &str) -> Option<String> {
        if let Some(cid) = class_of(kind) {
            return self.class_spelling(cid);
        }
        if kind == WIDEN_ARRAY_NA {
            return Some("list".to_string());
        }
        if kind == WIDEN_MAP_NA {
            return Some("dict".to_string());
        }
        self.type_imports.get("Matrix").cloned()
    }

    /// How this module spells a class in value position (`_class_spelling`).
    fn class_spelling(&self, cid: &str) -> Option<String> {
        if let Some(spelled) = self.class_spellings.get(cid) {
            return Some(spelled.clone());
        }
        let (module, name) = cid.rsplit_once('#').unwrap_or(("", cid));
        if module == "lib" { self.type_imports.get(name).cloned() } else { None }
    }

    /// The type a container constructor builds where a declaration says what it
    /// holds (`_declared_constructor`).
    pub(super) fn declared_constructor(
        &mut self,
        declared: &str,
        value: NodeId,
        value_ty: &str,
    ) -> R<String> {
        if value_ty == TYPELESS {
            let kind = facts::na_widen(declared);
            self.record_widen(value, kind.clone())?;
            return Ok(if kind.is_some() && holds_na(self.ast, value) {
                declared.to_string()
            } else {
                value_ty.to_string()
            });
        }
        let held = element_of(value_ty);
        let element = element_of(declared);
        let convertible = (held == INT && element == FLOAT)
            || (held == TYPELESS && facts::na_widen(element).is_some());
        let same_kind = (is_array(declared) && is_array(value_ty))
            || (is_matrix(declared) && is_matrix(value_ty));
        let Kind::Call { func, .. } = *self.kind(value) else { return Ok(value_ty.to_string()) };
        if value_ty == declared || !convertible || !same_kind {
            return Ok(value_ty.to_string());
        }
        let Some(callee) = self.lib_name(func) else { return Ok(value_ty.to_string()) };
        let Some(positions) = constructor_elements(&callee) else {
            return Ok(value_ty.to_string());
        };
        for arg in self.element_args(&callee, value, positions) {
            let arg_ty = self.ty_of(arg);
            if arg_ty == INT {
                self.record_widen(arg, Some(WIDEN_FLOAT.to_string()))?;
            } else if arg_ty == TYPELESS {
                self.record_widen(arg, facts::na_widen(element_of(declared)))?;
            }
        }
        if let Some(nid) = self.nid(value) {
            self.ty.insert(nid, declared.to_string());
        }
        crate::pine_type_rules::set_ty(self.ast, value, declared);
        Ok(declared.to_string())
    }

    /// The arguments a container constructor places as elements (`_element_args`).
    fn element_args(&self, callee: &str, call: NodeId, positions: &[usize]) -> Vec<NodeId> {
        let Some((_, args, keywords)) = facts::call_parts(self.ast, call) else { return vec![] };
        if positions.is_empty() {
            return args.iter().copied().filter(|&arg| !facts::is_starred(self.ast, arg)).collect();
        }
        let names = self.param_names(callee);
        positions
            .iter()
            .filter_map(|&at| bound_arg(self.ast, args, keywords, at, names.as_deref()))
            .collect()
    }

    /// The callee's declared parameter order, the registry's or the override table's.
    fn param_names(&self, callee: &str) -> Option<Vec<String>> {
        match data::lib_entries().get(callee).and_then(|entry| entry.names.clone()) {
            Some(names) => Some(names),
            None => override_param_names(callee)
                .map(|names| names.iter().map(|name| name.to_string()).collect()),
        }
    }

    /// Result type of a call to a lib name (`_lib_call_type`).
    pub(super) fn lib_call_type(
        &mut self,
        callee: &str,
        args: &[NodeId],
        keywords: &[NodeId],
        argc: Option<i64>,
        pin: Option<&str>,
    ) -> R<String> {
        let entry = data::lib_entries().get(callee);
        if MAP_KEY_LOOKUPS.contains(&callee) {
            let names = entry.and_then(|entry| entry.names.clone());
            self.record_key_widen(args, keywords, names.as_deref())?;
        }
        if let Some(rule) = lib_type_override(callee) {
            let names = self.param_names(callee);
            if let Some(resolved) =
                self.apply_override(rule, args, keywords, argc, names.as_deref())?
            {
                return Ok(resolved);
            }
        }
        let Some(entry) = entry else { return Ok(UNKNOWN.to_string()) };
        match entry.kind.as_str() {
            "value" => return Ok(entry.ty.clone().unwrap_or_else(|| UNKNOWN.to_string())),
            "function" => return Ok(entry.ret.clone().unwrap_or_else(|| UNKNOWN.to_string())),
            _ => {}
        }
        let Some(argc) = argc else { return Ok(UNKNOWN.to_string()) };
        let impls: Vec<ImplSig> = entry.impls.iter().map(lib_impl_sig).collect();
        let shapes = self.shapes(args);
        let kw_shapes = self.keyword_shapes(keywords);
        if let Some(choice) = self.choice(&impls, pin, shapes.as_deref(), kw_shapes.as_ref()) {
            return Ok(choice.ret);
        }
        let returns: Vec<String> = entry
            .impls
            .iter()
            .filter(|imp| arity_fits(imp, argc))
            .map(|imp| imp.ret.clone().unwrap_or_else(|| UNKNOWN.to_string()))
            .collect();
        Ok(overload_result(&returns).to_string())
    }

    /// Resolve one entry of the measured override table (`_apply_override`).
    pub(super) fn apply_override(
        &mut self,
        rule: CallRule,
        args: &[NodeId],
        keywords: &[NodeId],
        argc: Option<i64>,
        names: Option<&[String]>,
    ) -> R<Option<String>> {
        let form = match rule {
            CallRule::FirstOf(rules) => {
                for &candidate in rules {
                    if let Some(resolved) =
                        self.apply_override(candidate, args, keywords, argc, names)?
                    {
                        return Ok(Some(resolved));
                    }
                }
                return Ok(None);
            }
            CallRule::ByArity(table) => {
                let Some(argc) = argc else { return Ok(Some(UNKNOWN.to_string())) };
                let picked = table.iter().find(|(count, _)| *count as i64 == argc);
                return match picked {
                    None => Ok(None),
                    Some(&(_, picked)) => {
                        self.apply_override(picked, args, keywords, Some(argc), names)
                    }
                };
            }
            CallRule::Is(form) => form,
        };
        match form {
            "na_arg" => {
                let Some(argument) = bound_arg(self.ast, args, keywords, 0, names) else {
                    return Ok(None);
                };
                if let Some(built) = self.names_class(argument)? {
                    return Ok(Some(object_ty(&built)));
                }
                if !self.names_type(argument)? {
                    return Ok(None);
                }
                Ok(Some(self.annotation_type(Some(argument))?))
            }
            "join_args" => {
                let Some(passed) = self.passed(args, keywords, argc) else {
                    return Ok(Some(UNKNOWN.to_string()));
                };
                let Some((first, rest)) = passed.split_first() else { return Ok(None) };
                Ok(Some(rest.iter().fold(first.clone(), |result, ty| join(&result, ty))))
            }
            "all_int" => {
                let Some(passed) = self.passed(args, keywords, argc) else {
                    return Ok(Some(UNKNOWN.to_string()));
                };
                if passed.is_empty() {
                    return Ok(None);
                }
                if passed.iter().any(|ty| !NUMERIC.contains(&ty.as_str())) {
                    return Ok(Some(UNKNOWN.to_string()));
                }
                Ok(Some(if passed.iter().all(|ty| ty == INT) { INT } else { FLOAT }.to_string()))
            }
            "array_of_join_args" => {
                let passed = self.passed(args, keywords, argc).unwrap_or_default();
                let Some((first, rest)) = passed.split_first() else { return Ok(None) };
                let result = rest.iter().fold(first.clone(), |result, ty| join(&result, ty));
                for &arg in args {
                    if !facts::is_starred(self.ast, arg) {
                        let kind = facts::widen_for("array", &result, &self.ty_of(arg));
                        self.record_widen(arg, kind)?;
                    }
                }
                Ok(if result != UNKNOWN { Some(array_of(&result)) } else { None })
            }
            form if form.starts_with("merge_") || form.starts_with("put_") => {
                self.merge_override(form, args, keywords, names)
            }
            "matrix_mult" => {
                let base = self.arg_ty(args, keywords, 0, names);
                let other = self.arg_ty(args, keywords, 1, names);
                if element_of(&base) == UNKNOWN {
                    return Ok(None);
                }
                if is_array(&other) {
                    return Ok(Some(array_of(element_of(&base))));
                }
                if is_matrix(&other) || SCALARS.contains(&other.as_str()) {
                    return Ok(Some(base));
                }
                Ok(None)
            }
            form => Ok(self.shape_override(form, args, keywords, names)),
        }
    }

    /// Record the conversion of the key a map call is handed (`_record_key_widen`).
    fn record_key_widen(
        &mut self,
        args: &[NodeId],
        keywords: &[NodeId],
        names: Option<&[String]>,
    ) -> R<()> {
        let Some(key) = bound_arg(self.ast, args, keywords, 1, names) else { return Ok(()) };
        let base = self.arg_ty(args, keywords, 0, names);
        let kind = (is_map(&base) && key_of(&base) == STR && self.ty_of(key) == TYPELESS)
            .then(|| WIDEN_STRING_NA.to_string());
        self.record_widen(key, kind)
    }

    /// Resolve a container operation that takes a second container or puts one
    /// element in (`_merge_override`).
    fn merge_override(
        &mut self,
        form: &str,
        args: &[NodeId],
        keywords: &[NodeId],
        names: Option<&[String]>,
    ) -> R<Option<String>> {
        let putting = form.starts_with("put_");
        let put_kind = if putting { form[4..].split(':').next().unwrap_or("") } else { "" };
        let kind: fn(&str) -> bool = if putting {
            match put_kind {
                "array" => is_array,
                "map" => is_map,
                "matrix" | "matrix_array" => is_matrix,
                _ => return Err(Unsupported(format!("put form {form}"))),
            }
        } else if form == "merge_array" {
            is_array
        } else {
            is_matrix
        };
        let base = self.arg_ty(args, keywords, 0, names);
        if !kind(&base) {
            return Ok(None);
        }
        let mut operand_at: usize = if putting {
            form.split_once(':')
                .and_then(|(_, at)| at.parse().ok())
                .ok_or_else(|| Unsupported(format!("put form {form}")))?
        } else {
            1
        };
        if putting && bound_arg(self.ast, args, keywords, operand_at, names).is_none() {
            return Ok(Some(VOID.to_string()));
        }
        let mut other = self.arg_ty(args, keywords, operand_at, names);
        let element =
            if put_kind == "map" { value_of(&base) } else { element_of(&base) }.to_string();

        let fits_now: bool = if putting {
            let lenient = other != UNKNOWN && unchecked(&other);
            let mut fit = if put_kind == "matrix_array" {
                lenient
                    || (is_array(&other)
                        && element_of(&other) != UNKNOWN
                        && fits(&element, element_of(&other)))
            } else {
                lenient || (other != UNKNOWN && fits(&element, &other))
            };
            if fit && put_kind == "map" {
                let key = self.arg_ty(args, keywords, 1, names);
                fit = (key != UNKNOWN && unchecked(&key))
                    || (key != UNKNOWN && fits(key_of(&base), &key));
                if fit {
                    self.record_key_widen(args, keywords, names)?;
                } else {
                    operand_at = 1;
                    other = key;
                }
            }
            fit
        } else if kind(&other) {
            let held = element_of(&other);
            held != UNKNOWN
                && shape_conflict(&element, held).is_none()
                && join(&element, held) == element
        } else if form == "merge_matrix_or_scalar" && NUMERIC.contains(&other.as_str()) {
            join(&element, &other) == element
        } else {
            false
        };
        let operand = bound_arg(self.ast, args, keywords, operand_at, names);
        if let Some(operand) = operand {
            let conversion = if fits_now {
                facts::widen_for(if putting { put_kind } else { form }, &element, &other)
            } else {
                None
            };
            self.record_widen(operand, conversion)?;
        }
        if fits_now {
            if putting {
                return Ok(Some(
                    if put_kind == "map" { value_of(&base) } else { VOID }.to_string(),
                ));
            }
            return Ok(Some(base));
        }
        let receiver = bound_arg(self.ast, args, keywords, 0, names);
        let place = operand
            .or(receiver)
            .ok_or_else(|| Unsupported("a merge with neither operand nor receiver".into()))?;
        let nid = self.nid(place);
        let already = nid.is_some_and(|nid| self.shape_diags.contains(&nid));
        let unknown: Unknown;
        if already {
            unknown = self.unknown("unknown-value", place, &render_ty(&base));
        } else {
            if let Some(nid) = nid {
                self.shape_diags.insert(nid);
            }
            if other == UNKNOWN || (kind(&other) && element_of(&other) == UNKNOWN) {
                unknown = self.unknown("unknown-value", place, &render_ty(&base));
                self.diag(
                    &format!(
                        "{} takes an operand whose type is not known here, so what it holds \
                         afterwards is not known either",
                        render_ty(&base)
                    ),
                    place,
                    Some(unknown.clone()),
                    "give the operand a known type",
                );
            } else {
                unknown = Unknown {
                    reason: "shape-mismatch".into(),
                    line: self.line(place),
                    col: self.col(place),
                    detail: format!("{} vs {}", render_ty(&base), render_ty(&other)),
                };
                self.diag(
                    &format!(
                        "{} cannot take {}: the elements are different types",
                        render_ty(&base),
                        render_ty(&other)
                    ),
                    place,
                    Some(unknown.clone()),
                    &format!("pass {}", render_ty(&base)),
                );
            }
        }
        if self.pins_suppressed.is_none() {
            let found: Option<Diag> = self.diag_at(place).or_else(|| self.diags.last().cloned());
            self.pins_suppressed = Some(
                found.ok_or_else(|| Unsupported("pins suppressed with no diagnostic".into()))?,
            );
        }
        if let Some(receiver) = receiver
            && let Kind::Name { id, .. } = self.kind(receiver).clone()
            && let Some(found) = self.lookup_mut(&id)
            && found.ty != UNKNOWN
        {
            found.ty = UNKNOWN.to_string();
            found.unknown = Some(unknown);
        }
        Ok(Some(UNKNOWN.to_string()))
    }

    /// The diagnostic already standing at a node's position (`_diag_at`).
    fn diag_at(&self, node: NodeId) -> Option<Diag> {
        let (line, col) = (self.line(node), self.col(node));
        self.diags.iter().find(|diag| diag.line == line && diag.col == col).cloned()
    }

    /// Resolve the forms that read one argument (`_shape_override`).
    fn shape_override(
        &self,
        form: &str,
        args: &[NodeId],
        keywords: &[NodeId],
        names: Option<&[String]>,
    ) -> Option<String> {
        if let Some(index) = form.strip_prefix("arg")
            && is_digits(index)
        {
            let at: usize = index.parse().ok()?;
            return Some(match bound_arg(self.ast, args, keywords, at, names) {
                None => UNKNOWN.to_string(),
                Some(argument) => self.ty_of(argument),
            });
        }
        for &shape in SHAPE_FORMS {
            if let Some(index) = form.strip_prefix(shape)
                && is_digits(index)
            {
                let at: usize = index.parse().ok()?;
                let ty = self.arg_ty(args, keywords, at, names);
                let answer = shape_form(shape, &ty);
                return (answer != UNKNOWN && answer != OBJECT).then_some(answer);
            }
        }
        Some(form.to_string())
    }

    /// The type of the argument bound to one declared position (`_arg_ty`).
    fn arg_ty(
        &self,
        args: &[NodeId],
        keywords: &[NodeId],
        index: usize,
        names: Option<&[String]>,
    ) -> String {
        match bound_arg(self.ast, args, keywords, index, names) {
            None => UNKNOWN.to_string(),
            Some(argument) => self.ty_of(argument),
        }
    }

    /// The type of every argument a call passes (`_passed`).
    fn passed(
        &self,
        args: &[NodeId],
        keywords: &[NodeId],
        argc: Option<i64>,
    ) -> Option<Vec<String>> {
        argc?;
        let mut passed: Vec<String> = args.iter().map(|&arg| self.ty_of(arg)).collect();
        for &keyword in keywords {
            let (name, value) = keyword_parts(self.ast, keyword);
            if name.is_some() {
                passed.push(self.ty_of(value));
            }
        }
        Some(passed)
    }

    /// The static type of each positional argument (`_shapes`).
    pub(super) fn shapes(&self, args: &[NodeId]) -> Option<Vec<String>> {
        if args.iter().any(|&arg| facts::is_starred(self.ast, arg)) {
            return None;
        }
        Some(args.iter().map(|&arg| self.ty_of(arg)).collect())
    }

    /// The static type of each keyword argument, by parameter name (`_keyword_shapes`).
    pub(super) fn keyword_shapes(&self, keywords: &[NodeId]) -> Option<IndexMap<String, String>> {
        let mut out = IndexMap::new();
        for &keyword in keywords {
            let (name, value) = keyword_parts(self.ast, keyword);
            let name = name?;
            out.insert(name.to_string(), self.ty_of(value));
        }
        Some(out)
    }

    /// The implementation the selector's exact pass settles on (`_choice`).
    pub(super) fn choice(
        &self,
        impls: &[ImplSig],
        pin: Option<&str>,
        shapes: Option<&[String]>,
        kw_shapes: Option<&IndexMap<String, String>>,
    ) -> Option<ImplSig> {
        let (shapes, kw_shapes) = (shapes?, kw_shapes?);
        let any = |chars: String| chars.replace(FLOAT, &PIN_ANY.to_string());
        if !kw_shapes.is_empty() {
            let chars = any(pin_chars(shapes));
            let kw_chars: Vec<(String, String)> = kw_shapes
                .iter()
                .map(|(name, ty)| (name.clone(), any(pin_chars(std::slice::from_ref(ty)))))
                .collect();
            let kw_types: Vec<(String, String)> =
                kw_shapes.iter().map(|(name, ty)| (name.clone(), ty.clone())).collect();
            return overload_binding(impls, &chars, shapes, &kw_chars, &kw_types)
                .map(|index| impls[index].clone());
        }
        let chars = match pin {
            Some(pin) => pin.to_string(),
            None => any(pin_chars(shapes)),
        };
        overload_choice(impls, &chars, shapes).map(|index| impls[index].clone())
    }
}
