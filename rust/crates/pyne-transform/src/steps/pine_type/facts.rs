//! What the walk reads off the tree without typing anything: scopes, bindings and
//! their positions, the branch structure of the module body, and the small pure
//! rules of `pine_type_infer.py`'s module level.

use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use indexmap::IndexMap;
use pyne_ast::walk::{children, walk};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, Unsupported};

use crate::data;
use crate::node_ids::node_id;
use crate::pine_type_rules::{
    BOOL, FLOAT, INT, NUMERIC, OBJECT, SCALARS, STR, TYPELESS, UNKNOWN, VOID, WIDEN_ARRAY_NA,
    WIDEN_FLOAT, WIDEN_ITEMS, WIDEN_MAP_NA, WIDEN_MATRIX, WIDEN_MATRIX_NA, WIDEN_NUMBER_NA,
    WIDEN_STRING_NA, builtin_class_id, class_of, dotted, element_of, elements_of,
    forward_reference, get_pins, get_widen_site, get_widens, is_array, is_map, is_matrix, join,
    object_ty, shape_conflict,
};

/// Values no fit check has a say about (`_UNCHECKED`).
pub const UNCHECKED: &[&str] = &[UNKNOWN, TYPELESS, OBJECT, VOID];

pub fn unchecked(ty: &str) -> bool {
    UNCHECKED.contains(&ty)
}

/// A conversion verdict that never reaches a stamp (`_WIDEN_HARMLESS`).
pub const WIDEN_HARMLESS: &str = "=";

/// The type of an `hline` object (`_HLINE`).
pub const HLINE: &str = "o:lib#HLine";

/// The receivers a method call never asks for a Python attribute (`_SCALAR_RECEIVERS`).
pub const SCALAR_RECEIVERS: &[&str] = &[INT, FLOAT, BOOL, STR];

/// `dir(list)` of the interpreter the reference runs on (`_LIST_ATTRS`).
const LIST_ATTRS: &[&str] = &[
    "__add__",
    "__class__",
    "__class_getitem__",
    "__contains__",
    "__delattr__",
    "__delitem__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__getitem__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__imul__",
    "__init__",
    "__init_subclass__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__mul__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__reversed__",
    "__rmul__",
    "__setattr__",
    "__setitem__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "append",
    "clear",
    "copy",
    "count",
    "extend",
    "index",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
];

/// `dir(dict)` of the interpreter the reference runs on (`_DICT_ATTRS`).
const DICT_ATTRS: &[&str] = &[
    "__class__",
    "__class_getitem__",
    "__contains__",
    "__delattr__",
    "__delitem__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__getitem__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__ior__",
    "__iter__",
    "__le__",
    "__len__",
    "__lt__",
    "__ne__",
    "__new__",
    "__or__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__reversed__",
    "__ror__",
    "__setattr__",
    "__setitem__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "clear",
    "copy",
    "fromkeys",
    "get",
    "items",
    "keys",
    "pop",
    "popitem",
    "setdefault",
    "update",
    "values",
];

/// Annotation heads that give a name a history (`_SERIES_HEADS`).
const SERIES_HEADS: &[&str] = &["Series", "PersistentSeries", "IBPersistentSeries"];

/// Annotation heads that declare a variable `var` (`_PERSISTENT_HEADS`).
const PERSISTENT_HEADS: &[&str] =
    &["Persistent", "PersistentSeries", "IBPersistent", "IBPersistentSeries"];

/// A source position, `(line, column)`.
pub type Pos = (i64, i64);

/// A node's line, 0 for a synthetic one (`_line`).
pub fn line(ast: &Ast, node: NodeId) -> i64 {
    ast[node].loc.lineno.unwrap_or(0)
}

/// A node's column, 0 for a synthetic one (`_col`).
pub fn col(ast: &Ast, node: NodeId) -> i64 {
    ast[node].loc.col_offset.unwrap_or(0)
}

/// `(line, col)` of a node.
pub fn pos(ast: &Ast, node: NodeId) -> Pos {
    (line(ast, node), col(ast, node))
}

/// The source range one statement covers (`_span`); Python's `or` reads a 0 end
/// as missing.
pub fn span(ast: &Ast, node: NodeId) -> (Pos, Pos) {
    let start = pos(ast, node);
    let loc = ast[node].loc;
    let end_line = loc.end_lineno.filter(|&value| value != 0).unwrap_or(start.0);
    let end_col = loc.end_col_offset.filter(|&value| value != 0).unwrap_or(start.1);
    (start, (end_line, end_col))
}

/// The pre-order id of a node (`node_id`).
pub fn nid(ast: &Ast, node: NodeId) -> Option<i64> {
    node_id(ast, node)
}

pub fn is_def(ast: &Ast, node: NodeId) -> bool {
    matches!(ast.kind(node), Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

pub fn is_scope(ast: &Ast, node: NodeId) -> bool {
    matches!(
        ast.kind(node),
        Kind::FunctionDef { .. }
            | Kind::AsyncFunctionDef { .. }
            | Kind::Lambda { .. }
            | Kind::ClassDef { .. }
    )
}

/// The name of a definition or a class, empty for a lambda.
pub fn def_name(ast: &Ast, node: NodeId) -> &str {
    match ast.kind(node) {
        Kind::FunctionDef { name, .. }
        | Kind::AsyncFunctionDef { name, .. }
        | Kind::ClassDef { name, .. } => name,
        _ => "",
    }
}

/// The parts of a definition's `arguments`.
pub struct Args<'a> {
    pub posonlyargs: &'a [NodeId],
    pub args: &'a [NodeId],
    pub vararg: Option<NodeId>,
    pub kwonlyargs: &'a [NodeId],
    pub kw_defaults: &'a [Option<NodeId>],
    pub kwarg: Option<NodeId>,
    pub defaults: &'a [NodeId],
}

impl Args<'_> {
    pub fn positional(&self) -> Vec<NodeId> {
        self.posonlyargs.iter().chain(self.args).copied().collect()
    }
}

/// The `arguments` of a definition or a lambda.
pub fn args_of(ast: &Ast, node: NodeId) -> Option<Args<'_>> {
    let args = match ast.kind(node) {
        Kind::FunctionDef { args, .. }
        | Kind::AsyncFunctionDef { args, .. }
        | Kind::Lambda { args, .. } => *args,
        _ => return None,
    };
    match ast.kind(args) {
        Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults } => {
            Some(Args {
                posonlyargs,
                args,
                vararg: *vararg,
                kwonlyargs,
                kw_defaults,
                kwarg: *kwarg,
                defaults,
            })
        }
        _ => None,
    }
}

/// The name and the annotation of an `arg`.
pub fn arg_parts(ast: &Ast, arg: NodeId) -> (&str, Option<NodeId>) {
    match ast.kind(arg) {
        Kind::Arg { arg, annotation, .. } => (arg, *annotation),
        _ => ("", None),
    }
}

pub fn arg_name(ast: &Ast, arg: NodeId) -> String {
    arg_parts(ast, arg).0.to_string()
}

/// Every named parameter, positional first then keyword-only (`_every_param`).
pub fn every_param(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let Some(args) = args_of(ast, node) else { return Vec::new() };
    args.posonlyargs.iter().chain(args.args).chain(args.kwonlyargs).copied().collect()
}

/// The body of a definition.
pub fn def_body(ast: &Ast, node: NodeId) -> &[NodeId] {
    match ast.kind(node) {
        Kind::FunctionDef { body, .. }
        | Kind::AsyncFunctionDef { body, .. }
        | Kind::ClassDef { body, .. } => body,
        _ => &[],
    }
}

pub fn decorators(ast: &Ast, node: NodeId) -> &[NodeId] {
    match ast.kind(node) {
        Kind::FunctionDef { decorator_list, .. }
        | Kind::AsyncFunctionDef { decorator_list, .. }
        | Kind::ClassDef { decorator_list, .. } => decorator_list,
        _ => &[],
    }
}

pub fn returns_annotation(ast: &Ast, node: NodeId) -> Option<NodeId> {
    match ast.kind(node) {
        Kind::FunctionDef { returns, .. } | Kind::AsyncFunctionDef { returns, .. } => *returns,
        _ => None,
    }
}

/// The parts of a call.
pub fn call_parts(ast: &Ast, node: NodeId) -> Option<(NodeId, &[NodeId], &[NodeId])> {
    match ast.kind(node) {
        Kind::Call { func, args, keywords } => Some((*func, args, keywords)),
        _ => None,
    }
}

/// A keyword's name (`None` for a `**` unpacking) and value.
pub fn keyword_parts(ast: &Ast, keyword: NodeId) -> (Option<&str>, NodeId) {
    match ast.kind(keyword) {
        Kind::Keyword { arg, value } => (arg.as_deref(), *value),
        _ => (None, keyword),
    }
}

pub fn is_starred(ast: &Ast, node: NodeId) -> bool {
    matches!(ast.kind(node), Kind::Starred { .. })
}

pub fn is_name(ast: &Ast, node: NodeId, name: &str) -> bool {
    matches!(ast.kind(node), Kind::Name { id, .. } if id == name)
}

pub fn name_id(ast: &Ast, node: NodeId) -> Option<&str> {
    match ast.kind(node) {
        Kind::Name { id, .. } => Some(id),
        _ => None,
    }
}

pub fn str_constant(ast: &Ast, node: NodeId) -> Option<&str> {
    match ast.kind(node) {
        Kind::Constant { value: Constant::Str(text), .. } => Some(text),
        _ => None,
    }
}

/// How many arguments one argument list passes, `None` when an unpacking hides it
/// (`_arity`).
pub fn arity(ast: &Ast, args: &[NodeId], keywords: &[NodeId]) -> Option<i64> {
    if args.iter().any(|&arg| is_starred(ast, arg)) {
        return None;
    }
    if keywords.iter().any(|&keyword| keyword_parts(ast, keyword).0.is_none()) {
        return None;
    }
    Some((args.len() + keywords.len()) as i64)
}

/// How many arguments a call passes (`_call_arity`).
pub fn call_arity(ast: &Ast, node: NodeId) -> Option<i64> {
    let (_, args, keywords) = call_parts(ast, node)?;
    arity(ast, args, keywords)
}

/// The expression bound to one declared parameter position (`_bound_arg`).
pub fn bound_arg<S: AsRef<str>>(
    ast: &Ast,
    args: &[NodeId],
    keywords: &[NodeId],
    index: usize,
    param_names: Option<&[S]>,
) -> Option<NodeId> {
    if args.iter().any(|&arg| is_starred(ast, arg)) {
        return None;
    }
    if index < args.len() {
        return Some(args[index]);
    }
    let names = param_names?;
    let wanted = names.get(index)?.as_ref();
    keywords.iter().find_map(|&keyword| {
        let (arg, value) = keyword_parts(ast, keyword);
        (arg == Some(wanted)).then_some(value)
    })
}

/// Whether an annotation wraps its type in a persistent head (`_spells_persistent`).
pub fn spells_persistent(ast: &Ast, annotation: Option<NodeId>) -> bool {
    spells(ast, annotation, PERSISTENT_HEADS)
}

/// Whether an annotation wraps its type in a series head (`_spells_series`).
pub fn spells_series(ast: &Ast, annotation: Option<NodeId>) -> bool {
    spells(ast, annotation, SERIES_HEADS)
}

fn spells(ast: &Ast, annotation: Option<NodeId>, heads: &[&str]) -> bool {
    let Some(annotation) = annotation else { return false };
    walk(ast, annotation).into_iter().any(|sub| match ast.kind(sub) {
        Kind::Name { id, .. } => heads.contains(&id.as_str()),
        Kind::Attribute { attr, .. } => heads.contains(&attr.as_str()),
        _ => false,
    })
}

fn decorated(ast: &Ast, node: NodeId, name: &str) -> bool {
    decorators(ast, node).iter().any(|&decorator| {
        let target = match ast.kind(decorator) {
            Kind::Call { func, .. } => *func,
            _ => decorator,
        };
        match ast.kind(target) {
            Kind::Attribute { attr, .. } => attr == name,
            Kind::Name { id, .. } => id == name,
            _ => false,
        }
    })
}

/// Whether a definition is one implementation of an `@overload` group (`_is_overload`).
pub fn is_overload(ast: &Ast, node: NodeId) -> bool {
    decorated(ast, node, "overload")
}

/// Whether a definition is a Pine method (`_is_method`).
pub fn is_method(ast: &Ast, node: NodeId) -> bool {
    decorated(ast, node, "method")
}

/// What an `enumerate(x)` call iterates (`_enumerated`).
pub fn enumerated(ast: &Ast, node: NodeId) -> Option<NodeId> {
    let (func, args, _) = call_parts(ast, node)?;
    if args.len() != 1 {
        return None;
    }
    (dotted(ast, func).as_deref() == Some("enumerate")).then_some(args[0])
}

/// Whether a typeless value is Pine's typeless `na` (`_holds_na`).
pub fn holds_na(ast: &Ast, node: NodeId) -> bool {
    !is_name(ast, node, crate::pine_type_rules::DYN_DEFAULT)
}

/// The typeless `na` values a stored value holds, with their positions (`_typeless_nas`).
pub fn typeless_nas(ast: &Ast, node: NodeId, ty: &str) -> Vec<(NodeId, Vec<usize>)> {
    let mut out = Vec::new();
    typeless_nas_into(ast, node, ty, Vec::new(), &mut out);
    out
}

fn typeless_nas_into(
    ast: &Ast,
    node: NodeId,
    ty: &str,
    path: Vec<usize>,
    out: &mut Vec<(NodeId, Vec<usize>)>,
) {
    match ast.kind(node) {
        Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
            let elements = elements_of(ty);
            if elements.len() == elts.len() {
                for (index, (&element, item)) in elts.iter().zip(elements).enumerate() {
                    let mut inner = path.clone();
                    inner.push(index);
                    typeless_nas_into(ast, element, item, inner, out);
                }
            }
        }
        _ => {
            if ty == TYPELESS && holds_na(ast, node) {
                out.push((node, path));
            }
        }
    }
}

/// The type at a position of a tuple type (`_element_at`).
pub fn element_at(ty: &str, path: &[usize]) -> String {
    let mut ty = ty.to_string();
    for &index in path {
        let elements = elements_of(&ty);
        if index >= elements.len() {
            return UNKNOWN.to_string();
        }
        ty = elements[index].to_string();
    }
    ty
}

/// Whether the libraries are where a method name is looked up for a receiver
/// (`_reaches_library_methods`).
pub fn reaches_library_methods(receiver: &str, name: &str) -> bool {
    if SCALAR_RECEIVERS.contains(&receiver) {
        return true;
    }
    if name.starts_with('_') {
        return false;
    }
    if is_array(receiver) {
        return !LIST_ATTRS.contains(&name);
    }
    if is_map(receiver) {
        return !DICT_ATTRS.contains(&name);
    }
    is_matrix(receiver)
}

/// The conversion a typeless `na` needs where a value of a type goes (`_na_widen`).
pub fn na_widen(ty: &str) -> Option<String> {
    if ty == STR {
        return Some(WIDEN_STRING_NA.to_string());
    }
    if NUMERIC.contains(&ty) {
        return Some(WIDEN_NUMBER_NA.to_string());
    }
    if class_of(ty).is_some() {
        return Some(ty.to_string());
    }
    if is_array(ty) {
        return Some(WIDEN_ARRAY_NA.to_string());
    }
    if is_map(ty) {
        return Some(WIDEN_MAP_NA.to_string());
    }
    if is_matrix(ty) {
        return Some(WIDEN_MATRIX_NA.to_string());
    }
    None
}

/// What an instance vector carries for one context's conversion (`_vector_widen`).
pub fn vector_widen(kind: Option<&str>) -> Option<String> {
    match kind {
        None => None,
        Some(kind) if kind == WIDEN_HARMLESS || class_of(kind).is_some() => None,
        Some(kind) => Some(kind.to_string()),
    }
}

/// The conversion a value needs to go into a container of `element` (`_widen_for`).
pub fn widen_for(kind: &str, element: &str, other: &str) -> Option<String> {
    if other == TYPELESS {
        return if ["array", "map", "matrix"].contains(&kind) { na_widen(element) } else { None };
    }
    if element != FLOAT {
        return None;
    }
    if ["array", "map", "matrix"].contains(&kind) {
        return if other == INT {
            Some(WIDEN_FLOAT.to_string())
        } else if other == FLOAT {
            Some(WIDEN_HARMLESS.to_string())
        } else {
            None
        };
    }
    let conversion = if kind == "matrix_array" || kind == "merge_array" {
        if !is_array(other) {
            return None;
        }
        WIDEN_ITEMS
    } else if is_matrix(other) {
        WIDEN_MATRIX
    } else {
        return None;
    };
    let held = element_of(other);
    if held == INT {
        Some(conversion.to_string())
    } else if held == FLOAT {
        Some(WIDEN_HARMLESS.to_string())
    } else {
        None
    }
}

/// The lib classes that are a scalar, by object type (`lib_scalar_classes`).
fn scalar_class(ty: &str) -> Option<&'static str> {
    use std::sync::OnceLock;
    static SCALARS_BY_TY: OnceLock<HashMap<String, String>> = OnceLock::new();
    SCALARS_BY_TY
        .get_or_init(|| {
            data::scalar_classes()
                .iter()
                .map(|(name, scalar)| (object_ty(&builtin_class_id(name)), scalar.clone()))
                .collect()
        })
        .get(ty)
        .map(String::as_str)
}

/// Whether a value of one type may stand where another is declared (`_fits`).
pub fn fits(declared: &str, passed: &str) -> bool {
    let mut passed = passed;
    if SCALARS.contains(&declared) && passed.starts_with("o:") {
        passed = scalar_class(passed).unwrap_or(passed);
    }
    shape_conflict(declared, passed).is_none() && join(declared, passed) == declared
}

/// Whether a call site's answer differs between the contexts reaching it (`_per_context`).
pub fn per_context(ast: &Ast, call: NodeId) -> bool {
    get_pins(ast, call).is_some()
        || crate::pine_qualifier::get_series_lens(ast, call).is_some()
        || get_widens(ast, call).is_some()
}

/// Every call site of a function's own body, in pre-order (`_own_calls`).
pub fn own_calls(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut calls = Vec::new();
    let mut stack: Vec<NodeId> = def_body(ast, node).to_vec();
    while let Some(current) = stack.pop() {
        if let Some(site) = get_widen_site(ast, current) {
            calls.push(site);
        }
        match ast.kind(current) {
            Kind::Call { .. } => calls.push(current),
            Kind::Name { .. } | Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                continue;
            }
            _ => {}
        }
        stack.extend(children(ast, current));
    }
    calls.sort_by_key(|&call| site_order(ast, call));
    calls
}

/// The pre-order position of a call site (`_site_order`).
fn site_order(ast: &Ast, call: NodeId) -> (i64, i64) {
    if get_widens(ast, call).is_some() {
        let first = call_parts(ast, call).and_then(|(_, args, _)| args.first().copied());
        return (first.and_then(|arg| nid(ast, arg)).unwrap_or(0), 0);
    }
    (nid(ast, call).unwrap_or(0), 1)
}

/// What one lexical scope's own statements bind, declare and read (`_ScopeFacts`).
#[derive(Debug, Default)]
pub struct ScopeFacts {
    pub bound: HashSet<String>,
    pub declared_global: HashSet<String>,
    pub declared_nonlocal: HashSet<String>,
    pub loaded: HashSet<String>,
    pub nested: Vec<NodeId>,
}

/// The parts of a nested scope evaluated where it stands (`_scope_header`).
pub fn scope_header(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut out: Vec<NodeId> = decorators(ast, node).to_vec();
    if let Kind::ClassDef { bases, keywords, .. } = ast.kind(node) {
        out.extend(bases);
        out.extend(keywords.iter().map(|&keyword| keyword_parts(ast, keyword).1));
        return out;
    }
    let Some(args) = args_of(ast, node) else { return out };
    out.extend(args.defaults);
    out.extend(args.kw_defaults.iter().flatten());
    for &arg in args.posonlyargs.iter().chain(args.args).chain(args.kwonlyargs) {
        if let Some(annotation) = arg_parts(ast, arg).1 {
            out.push(annotation);
        }
    }
    if let Some(returns) = returns_annotation(ast, node) {
        out.push(returns);
    }
    out
}

/// What one scope's own statements bind, declare and read (`_scope_facts`).
pub fn scope_facts(ast: &Ast, node: NodeId) -> ScopeFacts {
    let mut stack: Vec<NodeId> = match ast.kind(node) {
        Kind::Lambda { body, .. } => vec![*body],
        _ => def_body(ast, node).to_vec(),
    };
    let mut facts = ScopeFacts::default();
    let mut headers: Vec<NodeId> = Vec::new();
    while let Some(current) = stack.pop() {
        match ast.kind(current) {
            Kind::Name { id, ctx } => {
                if *ctx == ExprContext::Load {
                    facts.loaded.insert(id.clone());
                } else {
                    facts.bound.insert(id.clone());
                }
                continue;
            }
            Kind::FunctionDef { .. }
            | Kind::AsyncFunctionDef { .. }
            | Kind::Lambda { .. }
            | Kind::ClassDef { .. } => {
                facts.nested.push(current);
                if let Kind::ClassDef { name, .. } = ast.kind(current) {
                    facts.bound.insert(name.clone());
                }
                headers.extend(scope_header(ast, current));
                continue;
            }
            Kind::Global { names } => facts.declared_global.extend(names.iter().cloned()),
            Kind::Nonlocal { names } => facts.declared_nonlocal.extend(names.iter().cloned()),
            Kind::Alias { name, asname } => {
                let bound = asname.as_deref().unwrap_or(name);
                facts.bound.insert(bound.split('.').next().unwrap_or("").to_string());
            }
            Kind::MatchMapping { rest: Some(rest), .. } => {
                facts.bound.insert(rest.clone());
            }
            Kind::ExceptHandler { name: Some(name), .. }
            | Kind::MatchAs { name: Some(name), .. }
            | Kind::MatchStar { name: Some(name) } => {
                facts.bound.insert(name.clone());
            }
            _ => {}
        }
        stack.extend(children(ast, current));
    }
    while let Some(current) = headers.pop() {
        match ast.kind(current) {
            Kind::Name { id, ctx } => {
                if *ctx == ExprContext::Load {
                    facts.loaded.insert(id.clone());
                }
                continue;
            }
            Kind::FunctionDef { .. }
            | Kind::AsyncFunctionDef { .. }
            | Kind::Lambda { .. }
            | Kind::ClassDef { .. } => {
                facts.nested.push(current);
                headers.extend(scope_header(ast, current));
                continue;
            }
            _ => {}
        }
        headers.extend(children(ast, current));
    }
    facts
}

/// A scope's free names: the lexically resolved ones and the ones `global` forces
/// to the module.
pub type FreeNames = (Rc<HashSet<String>>, Rc<HashSet<String>>);

/// The names one lexical scope reads from outside itself (`_scope_free`): the
/// lexically resolved ones and the ones `global` forces to the module.
pub fn scope_free(
    ast: &Ast,
    node: NodeId,
    memo: &mut HashMap<NodeId, FreeNames>,
    facts_of: &mut dyn FnMut(NodeId) -> Rc<ScopeFacts>,
) -> FreeNames {
    if let Some(cached) = memo.get(&node) {
        return cached.clone();
    }
    let facts = facts_of(node);
    let declared_global = &facts.declared_global;
    let mut bound: HashSet<String> = facts
        .bound
        .iter()
        .filter(|name| !declared_global.contains(*name) && !facts.declared_nonlocal.contains(*name))
        .cloned()
        .collect();
    if let Some(args) = args_of(ast, node) {
        for &arg in args.posonlyargs.iter().chain(args.args).chain(args.kwonlyargs) {
            bound.insert(arg_name(ast, arg));
        }
        for arg in [args.vararg, args.kwarg].into_iter().flatten() {
            bound.insert(arg_name(ast, arg));
        }
    }
    for &scope in &facts.nested {
        if !matches!(ast.kind(scope), Kind::Lambda { .. })
            && !declared_global.contains(def_name(ast, scope))
        {
            bound.insert(def_name(ast, scope).to_string());
        }
    }
    let mut free: HashSet<String> = facts
        .loaded
        .iter()
        .filter(|name| !bound.contains(*name) && !declared_global.contains(*name))
        .cloned()
        .collect();
    let mut module_free: HashSet<String> =
        facts.loaded.iter().filter(|name| declared_global.contains(*name)).cloned().collect();
    for &scope in &facts.nested {
        let (nested_free, nested_module) = scope_free(ast, scope, memo, facts_of);
        free.extend(nested_free.iter().filter(|name| !bound.contains(*name)).cloned());
        module_free.extend(nested_module.iter().cloned());
    }
    let answer = (Rc::new(free), Rc::new(module_free));
    memo.insert(node, answer.clone());
    answer
}

/// Every bare name a subtree calls (`_called_names`).
pub fn called_names(ast: &Ast, node: NodeId) -> HashSet<String> {
    walk(ast, node)
        .into_iter()
        .filter_map(|child| match ast.kind(child) {
            Kind::Call { func, .. } => name_id(ast, *func).map(str::to_string),
            _ => None,
        })
        .collect()
}

/// Every name one scope's statements bind by something other than a `def`, with the
/// position of every binding (`_bound_positions`), in first-seen order.
pub fn bound_positions(ast: &Ast, body: &[NodeId]) -> IndexMap<String, Vec<Pos>> {
    let mut out: IndexMap<String, Vec<Pos>> = IndexMap::new();
    let mut record = |name: &str, at: NodeId| {
        out.entry(name.to_string()).or_default().push(pos(ast, at));
    };
    let mut stack: Vec<NodeId> = body.to_vec();
    while let Some(current) = stack.pop() {
        match ast.kind(current) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. } => {
                continue;
            }
            Kind::ClassDef { name, .. } => {
                record(name, current);
                continue;
            }
            Kind::Name { id, ctx } => {
                if *ctx != ExprContext::Load {
                    record(id, current);
                }
            }
            Kind::Alias { name, asname } => {
                let bound = asname.as_deref().unwrap_or(name);
                record(bound.split('.').next().unwrap_or(""), current);
            }
            Kind::ExceptHandler { name: Some(name), .. }
            | Kind::MatchAs { name: Some(name), .. }
            | Kind::MatchStar { name: Some(name) } => record(name, current),
            Kind::MatchMapping { rest: Some(rest), .. } => record(rest, current),
            _ => {}
        }
        stack.extend(children(ast, current));
    }
    for positions in out.values_mut() {
        positions.sort();
    }
    out
}

/// Every statement list nested directly in one statement, with its label
/// (`_labelled_lists`).
pub fn labelled_lists(ast: &Ast, stmt: NodeId) -> Vec<(String, Vec<NodeId>)> {
    let mut out: Vec<(String, Vec<NodeId>)> = Vec::new();
    let kind = ast.kind(stmt);
    let class = kind.class();
    for label in ["body", "orelse", "finalbody"] {
        if let Some(index) = class.fields.iter().position(|field| field.name == label)
            && let pyne_ast::FieldRef::Nodes(nodes) = kind.field(index)
        {
            out.push((label.to_string(), nodes.clone()));
        }
    }
    match kind {
        Kind::Try { handlers, .. } | Kind::TryStar { handlers, .. } => {
            for (index, &handler) in handlers.iter().enumerate() {
                if let Kind::ExceptHandler { body, .. } = ast.kind(handler) {
                    out.push((format!("handler:{index}"), body.clone()));
                }
            }
        }
        Kind::Match { cases, .. } => {
            for (index, &case) in cases.iter().enumerate() {
                if let Kind::MatchCase { body, .. } = ast.kind(case) {
                    out.push((format!("case:{index}"), body.clone()));
                }
            }
        }
        _ => {}
    }
    out
}

/// Every statement list nested directly in one statement (`_statement_lists`).
pub fn statement_lists(ast: &Ast, stmt: NodeId) -> Vec<Vec<NodeId>> {
    labelled_lists(ast, stmt).into_iter().map(|(_, nodes)| nodes).collect()
}

/// Every statement of the module scope, nested ones included, in source order
/// (`_module_statements`).
pub fn module_statements(ast: &Ast, body: &[NodeId]) -> Vec<NodeId> {
    let mut out = Vec::new();
    module_statements_into(ast, body, &mut out);
    out
}

fn module_statements_into(ast: &Ast, body: &[NodeId], out: &mut Vec<NodeId>) {
    for &stmt in body {
        if matches!(
            ast.kind(stmt),
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::ClassDef { .. }
        ) {
            continue;
        }
        out.push(stmt);
        for nested in statement_lists(ast, stmt) {
            module_statements_into(ast, &nested, out);
        }
    }
}

/// Every `def` the module scope declares, with where it stands and whether nothing
/// guards it (`_module_defs`).
pub fn module_defs(ast: &Ast, body: &[NodeId]) -> IndexMap<String, Vec<(Pos, bool)>> {
    let mut out: IndexMap<String, Vec<(Pos, bool)>> = IndexMap::new();
    fn collect(
        ast: &Ast,
        stmts: &[NodeId],
        unguarded: bool,
        out: &mut IndexMap<String, Vec<(Pos, bool)>>,
    ) {
        for &stmt in stmts {
            match ast.kind(stmt) {
                Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => {
                    out.entry(name.clone()).or_default().push((pos(ast, stmt), unguarded));
                }
                Kind::ClassDef { .. } => {}
                _ => {
                    for nested in statement_lists(ast, stmt) {
                        collect(ast, &nested, false, out);
                    }
                }
            }
        }
    }
    collect(ast, body, true, &mut out);
    out
}

/// One position's chain of (compound statement, branch label) pairs (`_BranchPath`).
pub type BranchPath = Rc<Vec<(NodeId, String)>>;

/// Whether one pass of the module body can never reach both positions
/// (`_mutually_exclusive`).
pub fn mutually_exclusive(ast: &Ast, first: &BranchPath, second: &BranchPath) -> bool {
    for ((stmt, label), (other, other_label)) in first.iter().zip(second.iter()) {
        if stmt != other {
            return false;
        }
        if label != other_label {
            return matches!(ast.kind(*stmt), Kind::If { .. } | Kind::Match { .. });
        }
    }
    false
}

/// Where every statement of one module body stands in its branch structure
/// (`_BranchIndex`).
#[derive(Default)]
pub struct BranchIndex {
    entries: Vec<(Pos, Pos, BranchPath)>,
    starts: Vec<Pos>,
}

impl BranchIndex {
    pub fn new(ast: &Ast, body: &[NodeId]) -> BranchIndex {
        let mut entries: Vec<(Pos, Pos, BranchPath)> = Vec::new();
        fn collect(
            ast: &Ast,
            stmts: &[NodeId],
            path: &BranchPath,
            entries: &mut Vec<(Pos, Pos, BranchPath)>,
        ) {
            for &stmt in stmts {
                if is_def(ast, stmt) {
                    continue;
                }
                let (start, end) = span(ast, stmt);
                entries.push((start, end, path.clone()));
                if let Kind::Match { cases, .. } = ast.kind(stmt) {
                    for (index, &case) in cases.iter().enumerate() {
                        let Kind::MatchCase { pattern, guard, .. } = ast.kind(case) else {
                            continue;
                        };
                        if guard.is_some() {
                            continue;
                        }
                        let (pattern_start, pattern_end) = span(ast, *pattern);
                        let mut inner = (**path).clone();
                        inner.push((stmt, format!("case:{index}")));
                        entries.push((pattern_start, pattern_end, Rc::new(inner)));
                    }
                }
                for (label, nested) in labelled_lists(ast, stmt) {
                    let mut inner = (**path).clone();
                    inner.push((stmt, label));
                    collect(ast, &nested, &Rc::new(inner), entries);
                }
            }
        }
        collect(ast, body, &Rc::new(Vec::new()), &mut entries);
        entries.sort_by_key(|entry| entry.0);
        let starts = entries.iter().map(|entry| entry.0).collect();
        BranchIndex { entries, starts }
    }

    /// The branch path of the innermost statement covering one position.
    pub fn of(&self, position: Pos) -> BranchPath {
        let upto = self.starts.partition_point(|&start| start <= position);
        for index in (0..upto).rev() {
            let (_, end, path) = &self.entries[index];
            if position <= *end {
                return path.clone();
            }
        }
        Rc::new(Vec::new())
    }
}

/// The module-level names an importer is guaranteed to find a definition under
/// (`_exportable_names`).
pub fn exportable_names(
    ast: &Ast,
    body: &[NodeId],
    rebinds: &IndexMap<String, Vec<(Pos, BranchPath)>>,
    groups: &dyn Fn(&str) -> bool,
) -> std::collections::BTreeSet<String> {
    let mut exportable = std::collections::BTreeSet::new();
    for (name, positions) in module_defs(ast, body) {
        let &(last, unguarded) = positions.iter().max().expect("a definition");
        if !unguarded {
            continue;
        }
        if rebinds.get(&name).is_some_and(|binds| binds.iter().any(|(at, _)| *at > last)) {
            continue;
        }
        if groups(&name) && !positions.iter().all(|&(_, flag)| flag) {
            continue;
        }
        exportable.insert(name);
    }
    exportable
}

/// What the walk needs from the whole tree before it starts (`_scan_tree`): every
/// dotted name the annotations spell with the annotation it stands in, and what
/// every security id publishes.
pub struct Scan {
    pub annotations: IndexMap<String, NodeId>,
    pub writes: HashMap<String, Vec<NodeId>>,
}

pub fn scan_tree(ast: &Ast, tree: NodeId) -> Result<Scan, Unsupported> {
    let mut scan = Scan { annotations: IndexMap::new(), writes: HashMap::new() };
    let mut todo: VecDeque<NodeId> = VecDeque::from([tree]);
    while let Some(node) = todo.pop_front() {
        match ast.kind(node) {
            Kind::Call { func, args, .. } => {
                if args.len() >= 2
                    && dotted(ast, *func).as_deref() == Some("__sec_write__")
                    && let Some(sec_id) = str_constant(ast, args[0])
                {
                    scan.writes.entry(sec_id.to_string()).or_default().push(args[1]);
                }
            }
            Kind::Arg { annotation: Some(annotation), .. } | Kind::AnnAssign { annotation, .. } => {
                take(ast, *annotation, *annotation, &mut scan.annotations)?;
            }
            Kind::FunctionDef { returns: Some(returns), .. }
            | Kind::AsyncFunctionDef { returns: Some(returns), .. } => {
                take(ast, *returns, *returns, &mut scan.annotations)?;
            }
            _ => {}
        }
        for child in children(ast, node) {
            if !matches!(ast.kind(child), Kind::Name { .. } | Kind::Constant { .. }) {
                todo.push_back(child);
            }
        }
    }
    Ok(scan)
}

/// Record every dotted name an annotation spells, stringized ones parsed.
fn take(
    ast: &Ast,
    expr: NodeId,
    at: NodeId,
    annotations: &mut IndexMap<String, NodeId>,
) -> Result<(), Unsupported> {
    for child in walk(ast, expr) {
        match ast.kind(child) {
            Kind::Name { .. } | Kind::Attribute { .. } => {
                if let Some(spelled) = dotted(ast, child) {
                    annotations.entry(spelled).or_insert(at);
                }
            }
            Kind::Constant { value: Constant::Str(text), .. } => {
                let (parsed, root) = forward_reference(text)?;
                take(&parsed, root, at, annotations)?;
            }
            _ => {}
        }
    }
    Ok(())
}
