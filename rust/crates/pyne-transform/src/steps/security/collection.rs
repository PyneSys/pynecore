//! What the security step reads off the source shapes: the arguments of a call,
//! the walks that skip nested functions, the Pine "simple" chain and hoistable
//! binding rules, the plain-OHLCV fast path, and the history-reference rule behind
//! `closed_shift`.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::walk::{children, walk};
use pyne_ast::{Ast, Category, Constant, ExprContext, Int, Kind, NodeId};

use super::predicates::{is_security_call, is_security_lower_tf_call};
use super::{BindingCounts, Security, SeriesScope};
use crate::Error;

/// Strategy state accessors are meaningful only in the chart context -- the
/// security child process has no strategy state of its own, so referencing these
/// in a request.security() / request.security_lower_tf() expression is always a
/// programmer error. Only a direct attribute access is rejected; at runtime the
/// security child returns inert defaults for an escaped alias.
const FORBIDDEN_STRATEGY_STATE_ATTRS: [&str; 13] = [
    "equity",
    "eventrades",
    "grossloss",
    "grossprofit",
    "initial_capital",
    "losstrades",
    "max_drawdown",
    "max_runup",
    "netprofit",
    "openprofit",
    "position_avg_price",
    "position_size",
    "wintrades",
];

/// Raw price series the security child can serve WITHOUT running the script's
/// `main()`: each is set straight from the bar, so a `request.security[_lower_tf]`
/// whose expression is only these can take the fast path.
const OHLCV_PASSTHROUGH_FIELDS: [&str; 9] =
    ["open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4", "hlcc4"];

/// Annotations of an `AnnAssign` that DECLARE a history-bearing variable.
const SERIES_ANNOTATIONS: [&str; 3] = ["Series", "PersistentSeries", "IBPersistentSeries"];

/// Lib attributes that are no series (`lib_series_attrs.NON_SERIES_LIB_ATTRS`).
const NON_SERIES_LIB_ATTRS: [&str; 1] = ["extra_fields"];

pub(crate) fn is_function(kind: &Kind) -> bool {
    matches!(kind, Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

pub(crate) fn function_name(ast: &Ast, function: NodeId) -> &str {
    match ast.kind(function) {
        Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => name,
        _ => unreachable!("no function"),
    }
}

pub(crate) fn function_body(ast: &Ast, function: NodeId) -> &Vec<NodeId> {
    match ast.kind(function) {
        Kind::FunctionDef { body, .. } | Kind::AsyncFunctionDef { body, .. } => body,
        _ => unreachable!("no function"),
    }
}

pub(crate) fn function_body_mut(ast: &mut Ast, function: NodeId) -> &mut Vec<NodeId> {
    match ast.kind_mut(function) {
        Kind::FunctionDef { body, .. } | Kind::AsyncFunctionDef { body, .. } => body,
        _ => unreachable!("no function"),
    }
}

/// The `arguments` node of a function.
pub(crate) fn function_args(ast: &Ast, function: NodeId) -> NodeId {
    match ast.kind(function) {
        Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. } => *args,
        _ => unreachable!("no function"),
    }
}

/// The name of an `arg` node.
pub(crate) fn arg_name(ast: &Ast, arg: NodeId) -> &str {
    match ast.kind(arg) {
        Kind::Arg { arg, .. } => arg,
        _ => unreachable!("no arg"),
    }
}

/// The parameters of a function as `arg` nodes: positional-only, positional,
/// keyword-only, then `*args` and `**kwargs` (`_DependencyAnalyzer._params`).
pub(crate) fn params(ast: &Ast, function: NodeId) -> Vec<NodeId> {
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
        ast.kind(function_args(ast, function))
    else {
        unreachable!("no arguments")
    };
    let mut out: Vec<NodeId> = posonlyargs.iter().chain(args).chain(kwonlyargs).copied().collect();
    out.extend(vararg.iter().chain(kwarg.iter()).copied());
    out
}

/// The str value of a constant, `None` for anything else.
pub(crate) fn str_value(ast: &Ast, node: NodeId) -> Option<&str> {
    match ast.kind(node) {
        Kind::Constant { value: Constant::Str(value), .. } => Some(value),
        _ => None,
    }
}

fn is_name(ast: &Ast, node: NodeId, name: &str) -> bool {
    matches!(ast.kind(node), Kind::Name { id, .. } if id == name)
}

/// The keyword arguments of a call by name, a later one winning
/// (`{kw.arg: kw.value for kw in call.keywords if kw.arg is not None}`).
fn keyword_values(ast: &Ast, call: NodeId) -> HashMap<&str, NodeId> {
    let Kind::Call { keywords, .. } = ast.kind(call) else { unreachable!("no call") };
    let mut out = HashMap::new();
    for &keyword in keywords {
        if let Kind::Keyword { arg: Some(arg), value } = ast.kind(keyword) {
            out.insert(arg.as_str(), *value);
        }
    }
    out
}

/// The arguments of a call by Pine's positional order, each by keyword or position.
fn call_args<const N: usize>(ast: &Ast, call: NodeId, names: [&str; N]) -> [Option<NodeId>; N] {
    let Kind::Call { args, .. } = ast.kind(call) else { unreachable!("no call") };
    let keywords = keyword_values(ast, call);
    let mut out = [None; N];
    for (index, name) in names.iter().enumerate() {
        out[index] = keywords.get(name).copied().or_else(|| args.get(index).copied());
    }
    out
}

/// `(symbol, timeframe, expression, ignore_invalid_symbol)` of a
/// `request.security_lower_tf()` call (`_extract_ltf_args`).
pub(crate) fn extract_ltf_args(ast: &Ast, call: NodeId) -> [Option<NodeId>; 4] {
    call_args(ast, call, ["symbol", "timeframe", "expression", "ignore_invalid_symbol"])
}

/// `(symbol, timeframe, expression, gaps, lookahead, ignore_invalid_symbol,
/// currency)` of a `request.security()` call (`_extract_args`), in Pine v6's
/// positional order.
pub(crate) fn extract_args(ast: &Ast, call: NodeId) -> [Option<NodeId>; 7] {
    call_args(
        ast,
        call,
        [
            "symbol",
            "timeframe",
            "expression",
            "gaps",
            "lookahead",
            "ignore_invalid_symbol",
            "currency",
        ],
    )
}

/// The node and every descendant in pre-order, nested function definitions skipped
/// (`_walk_skip_funcs`).
pub(crate) fn walk_skip_funcs(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut out = vec![node];
    out.extend(pyne_ast::walk::descendants(ast, node, &is_function));
    out
}

/// The node and every descendant in post-order (children before their parent),
/// nested function definitions skipped (`_walk_skip_funcs_post`).
pub(crate) fn walk_skip_funcs_post(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut stack: Vec<(NodeId, std::vec::IntoIter<NodeId>)> =
        vec![(node, children(ast, node).into_iter())];
    while let Some((current, rest)) = stack.last_mut() {
        let current = *current;
        let mut next = None;
        for child in rest.by_ref() {
            if !is_function(ast.kind(child)) {
                next = Some(child);
                break;
            }
        }
        match next {
            Some(child) => stack.push((child, children(ast, child).into_iter())),
            None => {
                stack.pop();
                out.push(current);
            }
        }
    }
    out
}

/// The nodes of a function's OWN body, nested definitions excluded
/// (`_walk_skip_funcs_body`).
pub(crate) fn walk_skip_funcs_body(ast: &Ast, function: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    for &stmt in function_body(ast, function) {
        if is_function(ast.kind(stmt)) {
            continue;
        }
        out.extend(walk_skip_funcs(ast, stmt));
    }
    out
}

/// The first `lib.strategy.<state attr>` of an expression, breadth first
/// (`_find_forbidden_strategy_state`).
pub(crate) fn find_forbidden_strategy_state(ast: &Ast, expr: NodeId) -> Option<NodeId> {
    walk(ast, expr).into_iter().find(|&sub| {
        let Kind::Attribute { value, attr, .. } = ast.kind(sub) else { return false };
        if !FORBIDDEN_STRATEGY_STATE_ATTRS.contains(&attr.as_str()) {
            return false;
        }
        matches!(ast.kind(*value),
            Kind::Attribute { value: root, attr, .. } if attr == "strategy" && is_name(ast, *root, "lib"))
    })
}

/// Whether an expression can be evaluated at module level: constants, `lib.*`
/// attribute chains, and calls whose callee and every argument are such
/// (`_is_module_level_expr`).
pub(crate) fn is_module_level_expr(ast: &Ast, node: NodeId) -> bool {
    match ast.kind(node) {
        Kind::Constant { .. } => true,
        Kind::Attribute { value, .. } => is_module_level_expr(ast, *value),
        Kind::Name { id, .. } => id == "lib",
        // A call is only safe if its callee AND every argument are --
        // ``ticker.heikinashi(inputSymbol)`` has a lib.* callee but a local
        // argument, so it must not be emitted at module level.
        Kind::Call { func, args, keywords } => {
            is_module_level_expr(ast, *func)
                && args.iter().all(|&arg| is_module_level_expr(ast, arg))
                && keywords
                    .iter()
                    .all(|&keyword| is_module_level_expr(ast, keyword_value(ast, keyword)))
        }
        _ => false,
    }
}

fn keyword_value(ast: &Ast, keyword: NodeId) -> NodeId {
    match ast.kind(keyword) {
        Kind::Keyword { value, .. } => *value,
        _ => unreachable!("no keyword"),
    }
}

/// Whether a context's feed is only known once the script runs: its symbol or
/// timeframe is not evaluable at module level (`_resolved_at_runtime`).
pub(crate) fn resolved_at_runtime(
    ast: &Ast,
    symbol: Option<NodeId>,
    timeframe: Option<NodeId>,
) -> bool {
    [symbol, timeframe].into_iter().flatten().any(|expr| !is_module_level_expr(ast, expr))
}

/// Whether an expression is a Pine "simple" chain over the names that already hold
/// their final value at function entry (`_is_simple_chain`): constants, `lib.*`
/// chains, calls over such chains, plain operators over them, and those names.
pub(crate) fn is_simple_chain(ast: &Ast, node: NodeId, available: &HashSet<String>) -> bool {
    match ast.kind(node) {
        Kind::Constant { .. } => true,
        Kind::Name { id, .. } => id == "lib" || available.contains(id),
        Kind::Attribute { value, .. } => is_simple_chain(ast, *value, available),
        Kind::Call { func, args, keywords } => {
            is_simple_chain(ast, *func, available)
                && args.iter().all(|&arg| is_simple_chain(ast, arg, available))
                && keywords
                    .iter()
                    .all(|&keyword| is_simple_chain(ast, keyword_value(ast, keyword), available))
        }
        Kind::BinOp { .. }
        | Kind::UnaryOp { .. }
        | Kind::BoolOp { .. }
        | Kind::Compare { .. }
        | Kind::IfExp { .. }
        | Kind::Tuple { .. }
        | Kind::List { .. } => children(ast, node)
            .into_iter()
            .filter(|&child| ast.kind(child).class().category == Category::Expr)
            .all(|child| is_simple_chain(ast, child, available)),
        _ => false,
    }
}

/// How often a function binds or deletes each name, and what it declares `global`
/// / `nonlocal`, nested functions included (`_binding_counts`).
pub(crate) fn binding_counts(ast: &Ast, function: NodeId) -> BindingCounts {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut declared = HashSet::new();
    for sub in walk(ast, function) {
        match ast.kind(sub) {
            Kind::Name { id, ctx: ExprContext::Store | ExprContext::Del } => {
                *counts.entry(id.clone()).or_default() += 1;
            }
            Kind::Global { names } | Kind::Nonlocal { names } => {
                declared.extend(names.iter().cloned());
            }
            _ => {}
        }
    }
    (counts, declared)
}

/// The top-level bindings of a function that may move to its first statements,
/// with its stable parameters (`_hoistable_bindings`).
///
/// A parameter is stable when the body never assigns, deletes or declares it. A
/// binding qualifies when it is a plain `name = <simple chain>` standing directly
/// in the body, the name is bound exactly once and not declared; a parameter
/// counts too, but only while every preceding statement is itself a hoistable
/// binding (the form the instantiation pass pins a call site's constant in).
pub(crate) fn hoistable_bindings(
    ast: &Ast,
    function: NodeId,
    counts: &BindingCounts,
) -> (IndexMap<String, NodeId>, HashSet<String>) {
    let (counts, declared) = counts;
    let params: HashSet<String> =
        params(ast, function).iter().map(|&arg| arg_name(ast, arg).to_string()).collect();
    let stable_params: HashSet<String> = params
        .iter()
        .filter(|name| !counts.contains_key(*name) && !declared.contains(*name))
        .cloned()
        .collect();

    let mut hoistable = IndexMap::new();
    // Names already holding their final value at function entry
    let mut available = stable_params.clone();
    // A parameter may only be rebound while the whole prefix of the body consists
    // of hoistable bindings
    let mut prefix = true;
    for &stmt in function_body(ast, function) {
        let mut ok = false;
        if let Kind::Assign { targets, value, .. } = ast.kind(stmt)
            && let [target] = targets[..]
            && let Kind::Name { id, .. } = ast.kind(target)
            && !declared.contains(id)
            && counts.get(id) == Some(&1)
            && (prefix || !params.contains(id))
            && is_simple_chain(ast, *value, &available)
        {
            hoistable.insert(id.clone(), stmt);
            available.insert(id.clone());
            ok = true;
        }
        prefix = prefix && ok;
    }
    (hoistable, stable_params)
}

/// Names read by an expression, `lib` excluded (`_referenced_names`).
pub(crate) fn referenced_names(ast: &Ast, node: NodeId) -> HashSet<String> {
    walk(ast, node)
        .into_iter()
        .filter_map(|sub| match ast.kind(sub) {
            Kind::Name { id, .. } if id != "lib" => Some(id.clone()),
            _ => None,
        })
        .collect()
}

/// The hoistable bindings `needed` requires, transitively, in source order
/// (`_collect_hoisted`).
pub(crate) fn collect_hoisted(
    ast: &Ast,
    hoistable: &IndexMap<String, NodeId>,
    needed: &HashSet<String>,
) -> Vec<NodeId> {
    let mut selected: HashSet<&str> = HashSet::new();
    let mut pending: Vec<&str> = needed
        .iter()
        .filter_map(|name| hoistable.get_key_value(name).map(|(key, _)| key.as_str()))
        .collect();
    while let Some(name) = pending.pop() {
        if !selected.insert(name) {
            continue;
        }
        let Kind::Assign { value, .. } = ast.kind(hoistable[name]) else { unreachable!() };
        for reference in referenced_names(ast, *value) {
            if let Some((key, _)) = hoistable.get_key_value(&reference)
                && !selected.contains(key.as_str())
            {
                pending.push(key.as_str());
            }
        }
    }
    hoistable
        .iter()
        .filter(|(name, _)| selected.contains(name.as_str()))
        .map(|(_, &stmt)| stmt)
        .collect()
}

/// The raw-OHLCV field a bare `close` / `lib.close` names (`_ohlcv_field`).
fn ohlcv_field(ast: &Ast, node: NodeId) -> Option<&str> {
    match ast.kind(node) {
        Kind::Name { id, .. } if OHLCV_PASSTHROUGH_FIELDS.contains(&id.as_str()) => Some(id),
        Kind::Attribute { value, attr, .. }
            if OHLCV_PASSTHROUGH_FIELDS.contains(&attr.as_str()) && is_name(ast, *value, "lib") =>
        {
            Some(attr)
        }
        _ => None,
    }
}

/// `(field names, is tuple)` of a plain-OHLCV expression, `None` for any other
/// (`_ohlcv_passthrough`).
pub(crate) fn ohlcv_passthrough(
    ast: &Ast,
    expression: Option<NodeId>,
) -> Option<(Vec<String>, bool)> {
    let expression = expression?;
    match ast.kind(expression) {
        Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
            if elts.is_empty() {
                return None;
            }
            let fields: Option<Vec<String>> =
                elts.iter().map(|&elt| ohlcv_field(ast, elt).map(str::to_string)).collect();
            fields.map(|fields| (fields, true))
        }
        _ => ohlcv_field(ast, expression).map(|field| (vec![field.to_string()], false)),
    }
}

/// Whether a node literally names the lookahead-ON singleton (`_is_lookahead_on`).
pub(crate) fn is_lookahead_on(ast: &Ast, node: Option<NodeId>) -> bool {
    let Some(node) = node else { return false };
    match ast.kind(node) {
        Kind::Name { id, .. } => id == "lookahead_on",
        Kind::Attribute { value, attr, .. } => {
            attr == "lookahead_on" || (attr == "ON" && is_name(ast, *value, "Lookahead"))
        }
        _ => false,
    }
}

/// Whether an annotation declares a series variable (`_is_series_annotation`).
fn is_series_annotation(ast: &Ast, node: NodeId) -> bool {
    let node = match ast.kind(node) {
        Kind::Subscript { value, .. } => *value,
        _ => node,
    };
    matches!(ast.kind(node), Kind::Name { id, .. } if SERIES_ANNOTATIONS.contains(&id.as_str()))
}

/// Whether a parameter annotation gives the parameter a history buffer: a plain
/// `Series` / `Series[T]` (`_is_param_series_annotation`).
fn is_param_series_annotation(ast: &Ast, node: Option<NodeId>) -> bool {
    let Some(node) = node else { return false };
    let node = match ast.kind(node) {
        Kind::Subscript { value, .. } => *value,
        _ => node,
    };
    is_name(ast, node, "Series")
}

/// Whether a node is an int constant >= 1, a history offset (`_is_history_index`).
fn is_history_index(ast: &Ast, node: NodeId) -> bool {
    match ast.kind(node) {
        Kind::Constant { value: Constant::Int(Int::Small(value)), .. } => *value >= 1,
        Kind::Constant { value: Constant::Int(Int::Big(digits)), .. } => !digits.starts_with('-'),
        _ => false,
    }
}

/// Whether a node is the `inline_series(expr, k)` call PyneComp emits in place of
/// an expression history reference, with `k` an offset >= 1 (`_is_inline_series_call`).
fn is_inline_series_call(ast: &Ast, node: NodeId) -> bool {
    let Kind::Call { func, args, keywords } = ast.kind(node) else { return false };
    if args.len() != 2 || !keywords.is_empty() {
        return false;
    }
    let name = match ast.kind(*func) {
        Kind::Name { id, .. } => id.as_str(),
        Kind::Attribute { attr, .. } => attr.as_str(),
        _ => return false,
    };
    name == "inline_series" && is_history_index(ast, args[1])
}

/// Whether a subscript receiver is what the library series pass turns into a
/// series: an `Attribute` chain rooted at `lib` whose first segment is a series
/// (`_is_lib_series_chain`). A `Call` never is, which keeps `lib.ta.sma(...)[1]` out.
fn is_lib_series_chain(ast: &Ast, node: NodeId) -> bool {
    if !matches!(ast.kind(node), Kind::Attribute { .. }) {
        return false;
    }
    let mut first = None;
    let mut current = node;
    while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
        first = Some(attr.as_str());
        current = *value;
    }
    is_name(ast, current, "lib") && !first.is_some_and(|attr| NON_SERIES_LIB_ATTRS.contains(&attr))
}

/// Whether a name has the `__hist_N__` shape of the temps the inline-series hoist
/// lifts a call into (`_HIST_TEMP_RE`, `__hist_\d+__`). A non-ASCII digit is a
/// `\d` the native side does not classify: `Unsupported`.
fn is_hist_temp(name: &str) -> Result<bool, Error> {
    let Some(digits) = name.strip_prefix("__hist_").and_then(|rest| rest.strip_suffix("__")) else {
        return Ok(false);
    };
    if digits.is_empty() {
        return Ok(false);
    }
    if digits.chars().all(|c| c.is_ascii_digit()) {
        return Ok(true);
    }
    if digits.chars().any(|c| !c.is_ascii() && c.is_numeric()) {
        return Err(Error::Unsupported(format!("security: a temp name {name:?}")));
    }
    Ok(false)
}

/// What one function scope binds (`_scope_bindings`): the series a declaration or
/// a `Series`-annotated parameter gives a history buffer to, every name it binds,
/// and the value each `__hist_N__` temp holds (a temp bound twice is dropped).
pub(crate) fn scope_bindings(
    ast: &Ast,
    function: NodeId,
    own_nodes: &[NodeId],
) -> Result<SeriesScope, Error> {
    let mut scope = SeriesScope::default();
    let mut seen: HashSet<String> = HashSet::new();
    for param in params(ast, function) {
        scope.locals.insert(arg_name(ast, param).to_string());
    }
    let Kind::Arguments { args, .. } = ast.kind(function_args(ast, function)) else {
        unreachable!()
    };
    for &arg in args {
        let Kind::Arg { arg: name, annotation, .. } = ast.kind(arg) else { unreachable!() };
        if is_param_series_annotation(ast, *annotation) {
            scope.series.insert(name.clone());
        }
    }

    for &node in own_nodes {
        match ast.kind(node) {
            Kind::Name { id, ctx: ExprContext::Store | ExprContext::Del } => {
                scope.locals.insert(id.clone());
            }
            Kind::AnnAssign { target, annotation, .. } => {
                if let Kind::Name { id, .. } = ast.kind(*target)
                    && is_series_annotation(ast, *annotation)
                {
                    scope.series.insert(id.clone());
                }
            }
            Kind::Assign { targets, value, .. } if targets.len() == 1 => {
                let Kind::Name { id, .. } = ast.kind(targets[0]) else { continue };
                if !is_hist_temp(id)? {
                    continue;
                }
                if !seen.insert(id.clone()) {
                    scope.hist.remove(id);
                    continue;
                }
                scope.hist.insert(id.clone(), *value);
            }
            _ => {}
        }
    }
    Ok(scope)
}

impl Security {
    /// Whether a name resolves to a series along the enclosing scope chain
    /// (`_is_declared_series`): a name bound in the current scope without a series
    /// declaration there shadows any parent series of that name.
    fn is_declared_series(&self, name: &str) -> bool {
        let Some((current, parents)) = self.scopes.split_last() else { return false };
        if current.series.contains(name) {
            return true;
        }
        if current.locals.contains(name) {
            return false;
        }
        parents.iter().rev().any(|scope| scope.series.contains(name))
    }

    /// Whether a `[k]` subscript of a node addresses a history buffer
    /// (`_is_series_receiver`).
    fn is_series_receiver(&self, ast: &Ast, node: NodeId) -> bool {
        if let Kind::Name { id, .. } = ast.kind(node) {
            let bound = self.scopes.last().and_then(|scope| scope.hist.get(id));
            if let Some(&bound) = bound
                && is_inline_series_call(ast, bound)
            {
                return true;
            }
            return self.is_declared_series(id);
        }
        is_lib_series_chain(ast, node)
    }

    /// Whether a node reads a CLOSED bar, i.e. lowers to a history read
    /// (`_is_history_reference`): `inline_series(expr, k>=1)`, directly or through
    /// a `__hist_N__` temp of this scope, or `<series receiver>[k>=1]`.
    fn is_history_reference(&self, ast: &Ast, mut node: NodeId) -> bool {
        if let Kind::Name { id, .. } = ast.kind(node)
            && let Some(scope) = self.scopes.last()
            && let Some(&bound) = scope.hist.get(id)
        {
            node = bound;
        }
        if is_inline_series_call(ast, node) {
            return true;
        }
        match ast.kind(node) {
            Kind::Subscript { value, slice, .. } => {
                is_history_index(ast, *slice) && self.is_series_receiver(ast, *value)
            }
            _ => false,
        }
    }

    /// Whether a context's value can only change when its HTF period closes
    /// (`_closed_shift`): `lookahead_on` with an expression that is entirely a
    /// history reference, or a tuple / list of them.
    pub(crate) fn closed_shift(
        &self,
        ast: &Ast,
        expression: Option<NodeId>,
        lookahead: Option<NodeId>,
    ) -> bool {
        let Some(expression) = expression else { return false };
        if !is_lookahead_on(ast, lookahead) {
            return false;
        }
        match ast.kind(expression) {
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                !elts.is_empty() && elts.iter().all(|&elt| self.is_history_reference(ast, elt))
            }
            _ => self.is_history_reference(ast, expression),
        }
    }

    /// The security calls of a function's own body, each marked with a fresh sec id
    /// in its `_sec_id` attribute (`_collect_calls`): `(call, sid, is_ltf)`.
    pub(crate) fn collect_calls(
        &mut self,
        ast: &mut Ast,
        own_nodes: &[NodeId],
    ) -> Vec<(NodeId, String, bool)> {
        let mut calls = Vec::new();
        for &node in own_nodes {
            let ltf = if is_security_call(ast, node) {
                false
            } else if is_security_lower_tf_call(ast, node) {
                true
            } else {
                continue;
            };
            let sec_id = self.gen_id();
            ast.set_extra(node, "_sec_id", pyne_ast::Value::Str(sec_id.clone()));
            if ltf {
                self.ltf_sec_ids.insert(sec_id.clone());
            }
            calls.push((node, sec_id, ltf));
        }
        calls
    }

    /// [`hoistable_bindings`] of a function of the lowered module, its binding
    /// counts kept for the rest of the pass (`_lowered_hoistable`): a lift rewrites
    /// statement lists but binds no name.
    pub(crate) fn lowered_hoistable(
        &mut self,
        ast: &Ast,
        function: NodeId,
    ) -> (IndexMap<String, NodeId>, HashSet<String>) {
        let counts = self
            .binding_counts_memo
            .entry(function)
            .or_insert_with(|| binding_counts(ast, function));
        hoistable_bindings(ast, function, counts)
    }
}
