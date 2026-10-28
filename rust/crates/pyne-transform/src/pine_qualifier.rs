//! Port of `transformers/pine_qualifier.py`: the Pine qualifier lattice and the rules
//! that move values along it.
//!
//! Pine gives every value a qualifier besides its type: `const` (known at compile
//! time), `simple` (known before the first bar) and `series` (may change from bar to
//! bar). The Python module documents the measured law the rules follow; this port
//! keeps the same vocabulary, the same rules and the same stamp accessors, and is
//! checked against it one function at a time.

use pyne_ast::{Ast, Int, Kind, NodeId, Value};

/// Known at compile time
pub const CONST: u8 = 0;
/// Known before the first bar: inputs, the symbol, the chart
pub const SIMPLE: u8 = 1;
/// May change from bar to bar
pub const SERIES: u8 = 2;
/// A binding nothing has read yet -- above every qualifier, so `min` keeps the
/// lowest one a read has seen
pub const NOT_READ: u8 = 3;

/// Namespaces whose values are known before the first bar
pub const SIMPLE_NAMESPACES: &[&str] = &["syminfo", "timeframe", "chart"];

/// Namespaces whose values change from bar to bar
pub const SERIES_NAMESPACES: &[&str] = &[
    "barstate",
    "ta",
    "strategy",
    "strategy.opentrades",
    "strategy.closedtrades",
    "dividends",
    "earnings",
];

/// Built-ins whose result is `const` when every argument is
pub const CONST_PRESERVING: &[&str] = &[
    "color.new",
    "color.rgb",
    "math.abs",
    "math.acos",
    "math.asin",
    "math.atan",
    "math.ceil",
    "math.cos",
    "math.exp",
    "math.floor",
    "math.log",
    "math.log10",
    "math.max",
    "math.min",
    "math.pow",
    "math.round",
    "math.sign",
    "math.sin",
    "math.sqrt",
    "math.tan",
    "str.contains",
    "str.endswith",
    "str.length",
    "str.lower",
    "str.pos",
    "str.repeat",
    "str.replace",
    "str.startswith",
    "str.substring",
    "str.tonumber",
    "str.trim",
    "str.upper",
];

/// Built-ins whose result is `simple` when no argument is series
pub const SIMPLE_PRESERVING: &[&str] = &[
    "na",
    "nz",
    "int",
    "float",
    "bool",
    "string",
    "color",
    "math.avg",
    "math.round_to_mintick",
    "str.tostring",
    "str.format",
    "str.split",
    "timestamp",
    "timeframe.in_seconds",
    "timeframe.from_seconds",
];

/// Namespaces whose functions are all `simple` when no argument is series
pub const SIMPLE_CALL_NAMESPACES: &[&str] = &["ticker", "syminfo"];

/// Python builtins a Pyne script calls that only convert or combine their arguments
pub const PLAIN_BUILTINS: &[&str] =
    &["int", "float", "bool", "str", "abs", "min", "max", "round", "range"];

/// The built-ins whose machine depends on the qualifier of `length`
pub const WINDOW_CALLS: &[&str] =
    &["ta.highest", "ta.lowest", "ta.highestbars", "ta.lowestbars", "ta.linreg"];

/// The keyword the window built-ins take their machine from
pub const SERIES_LENGTH_KEYWORD: &str = "_series_length";

/// Attribute a window call carries its verdict under, where every context agrees
pub const SERIES_LEN_ATTR: &str = "_pine_series_len";

/// Attribute a window call carries its per-context verdicts under
pub const SERIES_LENS_ATTR: &str = "_pine_series_lens";

/// The qualifier of a lib value read (`lib_value_qualifier`).
pub fn lib_value_qualifier(name: &str, ty: &str) -> u8 {
    let Some((namespace, member)) = name.rsplit_once('.') else {
        // The bare built-in variables: prices, time and the bar clock
        return SERIES;
    };
    if namespace.is_empty() {
        return SERIES;
    }
    let head = namespace.split_once('.').map_or(namespace, |(head, _)| head);
    if SIMPLE_NAMESPACES.contains(&head) {
        return SIMPLE;
    }
    if namespace == "session" {
        return if member.starts_with("is") { SERIES } else { CONST };
    }
    if SERIES_NAMESPACES.contains(&namespace) {
        // Their enum members (`strategy.long`, `dividends.gross`) are constants
        if ty.starts_with("o:lib#") && namespace != "ta" {
            return CONST;
        }
        if name == "strategy.account_currency" {
            return SIMPLE;
        }
        return SERIES;
    }
    if namespace == "dayofweek" && member == "dayofweek" {
        return SERIES;
    }
    CONST
}

/// The qualifier of a lib call's result (`lib_call_qualifier`); `argc` is `None`
/// when an unpacking hides the argument count.
pub fn lib_call_qualifier(
    name: &str,
    arguments: impl IntoIterator<Item = u8>,
    argc: Option<usize>,
) -> u8 {
    let strongest = arguments.into_iter().max().unwrap_or(CONST);
    if name.starts_with("input.") || name == "input" {
        return strongest.max(SIMPLE);
    }
    if name == "timestamp" && argc == Some(1) {
        // Only the single date-string form is const
        return strongest;
    }
    if CONST_PRESERVING.contains(&name) {
        return strongest;
    }
    let head = name.split_once('.').map_or(name, |(head, _)| head);
    if SIMPLE_PRESERVING.contains(&name) || SIMPLE_CALL_NAMESPACES.contains(&head) {
        return strongest.max(SIMPLE);
    }
    SERIES
}

/// The qualifier of a Python builtin call, `None` when the callee is no such builtin
/// (`builtin_call_qualifier`).
pub fn builtin_call_qualifier(name: &str, arguments: impl IntoIterator<Item = u8>) -> Option<u8> {
    if !PLAIN_BUILTINS.contains(&name) {
        return None;
    }
    Some(arguments.into_iter().max().unwrap_or(CONST))
}

/// The `length` argument of a window call, `None` when the call does not spell one
/// (`window_length`).
pub fn window_length(ast: &Ast, name: &str, node: NodeId) -> Option<NodeId> {
    let Kind::Call { args, keywords, .. } = ast.kind(node) else { return None };
    let keyword_is = |keyword: NodeId, name: &str| matches!(ast.kind(keyword), Kind::Keyword { arg: Some(arg), .. } if arg == name);
    for &keyword in keywords {
        if let Kind::Keyword { arg: Some(arg), value } = ast.kind(keyword)
            && arg == "length"
        {
            return Some(*value);
        }
    }
    let positional: Vec<NodeId> = args
        .iter()
        .copied()
        .filter(|&arg| !matches!(ast.kind(arg), Kind::Starred { .. }))
        .collect();
    if positional.len() != args.len() {
        return None;
    }
    if name != "ta.linreg"
        && positional.len() == 1
        && !keywords.iter().any(|&keyword| keyword_is(keyword, "source"))
    {
        // The `ta.highest(length)` form
        return Some(positional[0]);
    }
    positional.get(1).copied()
}

/// A window call's verdict: true for a series length, false for a ring one, `None`
/// when unstamped (`get_series_len`).
pub fn get_series_len(ast: &Ast, node: NodeId) -> Option<bool> {
    match ast.extra(node, SERIES_LEN_ATTR) {
        Some(Value::Bool(flag)) => Some(*flag),
        _ => None,
    }
}

/// Stamp a window call's verdict, `None` included (`set_series_len`).
pub fn set_series_len(ast: &mut Ast, node: NodeId, value: Option<bool>) -> NodeId {
    ast.set_extra(node, SERIES_LEN_ATTR, value.map_or(Value::None, Value::Bool));
    node
}

/// The per-context verdicts of a window call the contexts disagree on, `None` when
/// one verdict holds for all (`get_series_lens`).
pub fn get_series_lens(ast: &Ast, node: NodeId) -> Option<Vec<(i64, bool)>> {
    let Some(Value::Dict(items)) = ast.extra(node, SERIES_LENS_ATTR) else { return None };
    items
        .iter()
        .map(|(context, verdict)| match (context, verdict) {
            (Value::Int(Int::Small(context)), Value::Bool(verdict)) => Some((*context, *verdict)),
            _ => None,
        })
        .collect()
}

/// Stamp the per-context verdicts of a window call, `None` included (`set_series_lens`).
pub fn set_series_lens(ast: &mut Ast, node: NodeId, value: Option<&[(i64, bool)]>) -> NodeId {
    let value = value.map_or(Value::None, |items| {
        Value::Dict(
            items
                .iter()
                .map(|&(context, verdict)| (Value::Int(Int::Small(context)), Value::Bool(verdict)))
                .collect(),
        )
    });
    ast.set_extra(node, SERIES_LENS_ATTR, value);
    node
}
