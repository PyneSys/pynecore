//! Port of `transformers/const_fold.py`: TradingView's parse-time constant folding.
//!
//! Every maximal constant subtree is evaluated once with exact doubles, the
//! transcendentals with fdlibm (StrictMath), and replaced by its literal rounded
//! through [`quantize_embed`]. Constants propagate through single names in
//! straight-line order; control flow kills the names it assigns. The Pine type
//! of a folded subtree is stamped on the emitted literal.
//!
//! A Pine int is a double: an int literal and int arithmetic are `f64`, and an
//! int is embedded as the Python int of its double's exact value (beyond `i64`,
//! its exact decimal digits). A `lib.math.round` with a positive precision scales
//! by the parser's power of ten, `fdlibm::pow_no_tail`, like the `pow` fold.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use pyne_ast::dump;
use pyne_ast::walk::{Transformer, Visited, copy_location, generic_visit, walk};
use pyne_ast::{
    Ast, Category, Constant, ExprContext, FieldRef, Int, Kind, NodeId, Operator, UnaryOperator,
};

use crate::fdlibm;
use crate::pine_type_rules::{
    BOOL, CallRule, Classes, FLOAT, INT, STR, UNKNOWN, annotation_type, binop_type, is_numeric,
    join, lib_type_override, set_ty, unaryop_type,
};
use crate::{Context, Error};

/// A Python number the fold computes with: an `int`, held as the integral double a
/// Pine int is (never `-0.0`), or a `float`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(f64),
    Float(f64),
}

impl Num {
    /// `float(v)`: the double itself.
    fn float(self) -> f64 {
        match self {
            Num::Int(value) | Num::Float(value) => value,
        }
    }

    /// An int from an integral double: an int has no signed zero.
    fn int(value: f64) -> Num {
        Num::Int(if value == 0.0 { 0.0 } else { value })
    }
}

/// The module a compiled script imports its Pine casts from.
const PINE_CAST: &str = "pynecore.core.pine_cast";

/// What `_try_eval` answers: `None` is `_BAIL`, not a compile-time constant.
type Eval = Result<Option<Num>, Error>;

fn unsupported(message: impl Into<String>) -> Error {
    Error::Unsupported(message.into())
}

// --- the embedding --------------------------------------------------------

/// `10**k` for the up to three digits the shortest spelling of a double in
/// `[1e-3, 1)` can carry past the 16th decimal place.
const POW10: [u64; 4] = [1, 10, 100, 1000];

/// The shortest decimal digits of a positive finite double (Python's `repr`), as
/// `(digits, exponent)` with `value == digits * 10**exponent`.
fn shortest_digits(magnitude: f64) -> (u64, i32) {
    let (digits, point) = dump::shortest_digits(magnitude);
    let exponent = point - digits.len() as i32;
    (digits.parse().expect("at most 17 digits"), exponent)
}

/// How the exact binary value of a positive normal double compares with
/// `digits * 10**exponent`, for a negative exponent of at most 19 places and a
/// double below one (Python's `Decimal(magnitude)` against the shortest digits).
fn compare_exact(magnitude: f64, digits: u64, exponent: i32) -> Ordering {
    let bits = magnitude.to_bits();
    let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
    let shift = 1075 - ((bits >> 52) & 0x7ff) as u32;
    // mantissa * 2**-shift against digits * 10**exponent, both sides scaled up to
    // integers: mantissa * 10**-exponent against digits * 2**shift
    let left = mantissa as u128 * 10u128.pow((-exponent) as u32);
    let right = (digits as u128) << shift;
    left.cmp(&right)
}

fn quantize_float(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let magnitude = value.abs();
    if magnitude < 1e-3 {
        return value;
    }
    let (digits, exponent) = shortest_digits(magnitude);
    if exponent >= -16 {
        // No digit below the 16th place: the digits round to themselves, and the
        // shortest spelling parses back to the very double
        return value;
    }
    let places = (-16 - exponent) as usize;
    let scale = POW10[places];
    let (down, tail) = (digits / scale, digits % scale);
    let rounded = match (2 * tail).cmp(&scale) {
        Ordering::Greater => down + 1,
        Ordering::Less => down,
        Ordering::Equal => match compare_exact(magnitude, digits, exponent) {
            Ordering::Greater => down + 1,
            Ordering::Less => down,
            Ordering::Equal => down + (down & 1),
        },
    };
    let parsed: f64 = format!("{rounded}e-16").parse().expect("a decimal");
    parsed.copysign(value)
}

/// The value a folded constant or a source literal is embedded with
/// (`quantize_embed`): a float rounded to 16 decimal places while `|v| >= 1e-3`,
/// a tie at the 17th decimal decided on the double's exact binary value, a zero
/// unsigned; an int unchanged.
pub fn quantize_embed(value: Num) -> Num {
    match value {
        Num::Int(_) => value,
        Num::Float(value) => Num::Float(quantize_float(value)),
    }
}

/// The literal an embedded value is spelled as.
fn constant(value: Num) -> Constant {
    match value {
        Num::Int(value) => Constant::Int(Int::from_integral(value)),
        Num::Float(value) => Constant::Float(value),
    }
}

/// Whether `node` is already the literal `value`: same type, same value, same zero sign.
fn is_literal(ast: &Ast, node: NodeId, value: Num) -> bool {
    match (ast.kind(node), value) {
        (Kind::Constant { value: Constant::Int(have), .. }, Num::Int(want)) => {
            *have == Int::from_integral(want)
        }
        (Kind::Constant { value: Constant::Float(have), .. }, Num::Float(want)) => {
            *have == want && have.is_sign_negative() == want.is_sign_negative()
        }
        _ => false,
    }
}

// --- the evaluation -------------------------------------------------------

/// The `lib.math` module constants, as the runtime module computes them.
fn math_constant(name: &str) -> Option<f64> {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        "phi" => Some(phi),
        "rphi" => Some(1.0 / phi),
        _ => None,
    }
}

/// The attribute name of a `lib.math.<name>` chain (`_is_lib_math_attr`).
fn lib_math_attr(ast: &Ast, node: NodeId) -> Option<&str> {
    let Kind::Attribute { value, attr, .. } = ast.kind(node) else {
        return None;
    };
    let Kind::Attribute { value: root, attr: module, .. } = ast.kind(*value) else {
        return None;
    };
    match ast.kind(*root) {
        Kind::Name { id, .. } if module == "math" && id == "lib" => Some(attr),
        _ => None,
    }
}

/// `StrictMath.log(10)`, the divisor of the parser's `log10`.
fn fold_log10(x: f64) -> f64 {
    // MEASURED (8/8 arguments where the candidates differ): the parser's log10 is
    // ``StrictMath.log(x) / StrictMath.log(10)``, not ``StrictMath.log10``
    fdlibm::log(x) / fdlibm::log(10.0)
}

/// The fdlibm folds of `_FOLD_FDLIBM`, by name, with their argument counts.
fn fold_fdlibm(name: &str, args: &[f64]) -> Option<Option<f64>> {
    let unary: fn(f64) -> f64 = match name {
        "sin" => fdlibm::sin,
        "cos" => fdlibm::cos,
        "tan" => fdlibm::tan,
        "exp" => fdlibm::exp,
        "log" => fdlibm::log,
        "log10" => fold_log10,
        "asin" => fdlibm::asin,
        "acos" => fdlibm::acos,
        "atan" => fdlibm::atan,
        "pow" => {
            // MEASURED (probes pw1-pw7, 452 constants): the parser's pow is StrictMath's
            // without the tail of the reduced exponent
            return Some(match args {
                &[x, y] => Some(fdlibm::pow_no_tail(x, y)),
                _ => None,
            });
        }
        _ => return None,
    };
    Some(match args {
        &[x] => Some(unary(x)),
        _ => None,
    })
}

/// Python's comparison of two numbers: an int is an exact double, so it is the
/// comparison of the doubles.
fn compare(left: Num, right: Num) -> Ordering {
    left.float().partial_cmp(&right.float()).expect("finite operands")
}

/// `lib.math.round` as the parser folds it, the integer rounding: the number
/// rounded to 15 decimal places, half up, then half away from zero
/// (`_fold_round`).
fn fold_round(number: Num) -> Eval {
    // From here up every double is an integer
    const INT_LIMIT: f64 = 4503599627370496.0;
    let negative = compare(number, Num::Float(0.0)) == Ordering::Less;
    let magnitude = match number {
        Num::Int(value) if negative => Num::Int(-value),
        Num::Float(value) if negative => Num::Float(-value),
        _ => number,
    };
    if compare(magnitude, Num::Float(INT_LIMIT)) != Ordering::Less {
        return Ok(Some(number));
    }
    let mut units = magnitude.float().floor();
    if reaches_tie(magnitude.float() - units, 15) {
        units += 1.0;
    }
    // ``float(-units if negative else units)``: an int negated, so never -0.0
    Ok(Some(Num::Float(if negative && units != 0.0 { -units } else { units })))
}

/// `lib.math.round(number, precision)` with a positive precision, as the parser
/// folds it (`_fold_round_to`): the magnitude rounded to 15 decimal places, half
/// up, times the parser's `10**precision` (a precision above 16 acts as 16), that
/// product rounded half up to `int(precision)` places and then to an integer, and
/// divided back by the power.
fn fold_round_to(number: Num, precision: Num) -> Eval {
    const MAX_PRECISION: f64 = 16.0;
    let negative = compare(number, Num::Float(0.0)) == Ordering::Less;
    let p = precision.float().min(MAX_PRECISION);
    let scale = fdlibm::pow_no_tail(10.0, p);
    let scaled = round_to_15_places(number.float().abs()) * scale;
    if !scaled.is_finite() {
        return Ok(None);
    }
    let mut units = scaled.floor();
    if reaches_tie(scaled - units, p as u32) {
        units += 1.0;
    }
    let value = units / scale;
    // ``0.0 - value``: a negative number rounded away to nothing is 0, not -0
    Ok(Some(Num::Float(if negative { 0.0 - value } else { value })))
}

/// The double nearest to a finite non-negative double rounded to 15 decimal places,
/// half up.
fn round_to_15_places(magnitude: f64) -> f64 {
    // From 8 up an ulp is wider than twice the 5e-16 the rounding can move the
    // number, so the nearest double is the number itself
    if magnitude >= 8.0 {
        return magnitude;
    }
    // Below 2**-52 the number is less than half of 1e-15
    if magnitude < f64::EPSILON {
        return 0.0;
    }
    // magnitude == mantissa * 2**-shift exactly, with 50 <= shift <= 104
    let (mantissa, exponent) = decompose(magnitude);
    let shift = (-exponent) as u32;
    let places = (u128::from(mantissa) * 1_000_000_000_000_000 + (1u128 << (shift - 1))) >> shift;
    // places < 8 * 10**15 < 2**53, so both operands are exact and the quotient is
    // correctly rounded
    places as f64 / 1e15
}

/// Whether a fraction in `[0, 1)` reaches 0.5 once rounded to `places` decimal
/// places, half up: `fraction >= 0.5 - 0.5 / 10**places` exactly.
fn reaches_tie(fraction: f64, places: u32) -> bool {
    if fraction >= 0.5 {
        return true;
    }
    if places == 0 || fraction < 0.25 {
        return false;
    }
    // In [0.25, 0.5) the fraction is mantissa * 2**-54 exactly:
    // 2 * mantissa * 2**-54 * tie >= tie - 1
    let (mantissa, _) = decompose(fraction);
    let tie = 10u128.pow(places);
    u128::from(mantissa) * tie >= (tie - 1) << 53
}

/// A finite non-negative double as `mantissa * 2**exponent`.
fn decompose(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    if biased == 0 { (fraction, -1074) } else { (fraction | (1 << 52), biased - 1075) }
}

/// One of `_FOLD_EXACT`: the `lib.math` runtime function itself, `None` where
/// it raises one of the errors the fold catches.
fn fold_exact(name: &str, args: &[Num]) -> Eval {
    let single = match args {
        &[only] => Some(only),
        _ => None,
    };
    let value = match name {
        "sqrt" => single.map(|x| {
            if compare(x, Num::Float(0.0)) == Ordering::Less {
                Num::Float(f64::NAN)
            } else {
                Num::Float(x.float().sqrt())
            }
        }),
        "abs" => match single {
            Some(Num::Int(value)) => Some(Num::Int(value.abs())),
            Some(Num::Float(value)) => Some(Num::Float(value.abs())),
            None => None,
        },
        // ``pine_int(math.floor(x))``: the int floor back as a float, so never -0.0
        "floor" | "ceil" => single.map(|x| {
            let value = match x {
                Num::Int(value) => value,
                Num::Float(value) if name == "floor" => value.floor(),
                Num::Float(value) => value.ceil(),
            };
            Num::Float(if value == 0.0 { 0.0 } else { value })
        }),
        "sign" => single.map(|x| {
            Num::Float(match compare(x, Num::Float(0.0)) {
                Ordering::Equal => 0.0,
                Ordering::Greater => 1.0,
                Ordering::Less => -1.0,
            })
        }),
        // ``math.degrees`` / ``math.radians``: one multiplication by the folded factor
        "todegrees" => single.map(|x| Num::Float(x.float() * (180.0 / std::f64::consts::PI))),
        "toradians" => single.map(|x| Num::Float(x.float() * (std::f64::consts::PI / 180.0))),
        // ``builtins.max`` / ``builtins.min``: the first of the extreme operands
        "max" | "min" => {
            let wanted = if name == "max" { Ordering::Greater } else { Ordering::Less };
            let mut best = args[0];
            for &item in &args[1..] {
                if compare(item, best) == wanted {
                    best = item;
                }
            }
            Some(best)
        }
        "avg" => Some(Num::Float(fold_avg(args))),
        "round" => match *args {
            [number] => return fold_round(number),
            [number, precision] => {
                if compare(precision, Num::Float(0.0)) == Ordering::Greater {
                    return fold_round_to(number, precision);
                }
                return fold_round(number);
            }
            _ => None,
        },
        _ => None,
    };
    Ok(value)
}

/// `lib.math.avg`: a plain sum up to two operands, a Kahan sum from three on.
fn fold_avg(args: &[Num]) -> f64 {
    let count = args.len() as f64;
    if args.len() <= 2 {
        let mut sum = 0.0;
        for n in args {
            sum += n.float();
        }
        return sum / count;
    }
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for n in args {
        let y = n.float() - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    (sum - compensation) / count
}

/// `+ - * / %` on two numbers: the double result, `%` the truncated remainder of
/// `safe_mod`. Two ints give an int, except through `/`.
fn fold_binop(op: Operator, left: Num, right: Num) -> Option<Num> {
    let (a, b) = (left.float(), right.float());
    let value = match op {
        Operator::Add => a + b,
        Operator::Sub => a - b,
        Operator::Mult => a * b,
        Operator::Div => a / b,
        Operator::Mod => a % b,
        _ => return None,
    };
    if !value.is_finite() {
        return None;
    }
    Some(match (left, right) {
        (Num::Int(_), Num::Int(_)) if op != Operator::Div => Num::int(value),
        _ => Num::Float(value),
    })
}

/// Evaluate a subtree as a parse-time constant with exact doubles (`_try_eval`).
fn try_eval(ast: &Ast, node: NodeId, env: &HashMap<String, Num>) -> Eval {
    match ast.kind(node) {
        Kind::Constant { value, .. } => Ok(match value {
            // ``float(v)``: the nearest double, ties to even
            Constant::Int(Int::Small(value)) => Some(Num::Int(*value as f64)),
            // An int beyond the double range is an infinity: a domain edge
            Constant::Int(Int::Big(digits)) => {
                digits.parse::<f64>().ok().filter(|value| value.is_finite()).map(Num::Int)
            }
            // An overflowing float literal (``1e309``) is an infinity: a domain edge
            Constant::Float(value) if value.is_finite() => Some(Num::Float(*value)),
            _ => None,
        }),
        Kind::Name { id, ctx } => {
            Ok(if *ctx == ExprContext::Load { env.get(id).copied() } else { None })
        }
        Kind::UnaryOp { op: op @ (UnaryOperator::USub | UnaryOperator::UAdd), operand } => {
            let Some(value) = try_eval(ast, *operand, env)? else {
                return Ok(None);
            };
            Ok(Some(match (op, value) {
                (UnaryOperator::USub, Num::Int(v)) => Num::int(-v),
                (UnaryOperator::USub, Num::Float(v)) => Num::Float(-v),
                _ => value,
            }))
        }
        Kind::BinOp { left, op, right } => {
            if !matches!(
                op,
                Operator::Add | Operator::Sub | Operator::Mult | Operator::Div | Operator::Mod
            ) {
                return Ok(None);
            }
            // Both sides are evaluated, as in Python: one the port cannot model
            // answers Unsupported even when the other one is no constant
            let left_value = try_eval(ast, *left, env)?;
            let right_value = try_eval(ast, *right, env)?;
            let (Some(left_value), Some(right_value)) = (left_value, right_value) else {
                return Ok(None);
            };
            if matches!(op, Operator::Div | Operator::Mod) && right_value.float() == 0.0 {
                return Ok(None);
            }
            Ok(fold_binop(*op, left_value, right_value))
        }
        Kind::Attribute { .. } => {
            Ok(lib_math_attr(ast, node).and_then(math_constant).map(Num::Float))
        }
        Kind::Call { func, args, keywords } => {
            if !keywords.is_empty() {
                return Ok(None);
            }
            let Some(name) = lib_math_attr(ast, *func) else {
                return Ok(None);
            };
            let mut values = Vec::with_capacity(args.len());
            for &arg in args {
                if matches!(ast.kind(arg), Kind::Starred { .. }) {
                    return Ok(None);
                }
                let Some(value) = try_eval(ast, arg, env)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            let floats: Vec<f64> = values.iter().map(|value| value.float()).collect();
            let value = if let Some(result) = fold_fdlibm(name, &floats) {
                match result {
                    Some(value) => Num::Float(value),
                    None => return Ok(None),
                }
            } else if EXACT_NAMES.contains(&name) {
                if values.is_empty() {
                    return Ok(None);
                }
                match fold_exact(name, &values)? {
                    Some(value) => value,
                    None => return Ok(None),
                }
            } else {
                return Ok(None);
            };
            // Also rejects NA results (domain edges) -- those stay runtime
            Ok(match value {
                Num::Float(value) if !value.is_finite() => None,
                value => Some(value),
            })
        }
        _ => Ok(None),
    }
}

/// The `lib.math` functions folded with their runtime implementations (`_FOLD_EXACT_NAMES`).
const EXACT_NAMES: [&str; 11] = [
    "sqrt",
    "abs",
    "floor",
    "ceil",
    "min",
    "max",
    "avg",
    "sign",
    "round",
    "todegrees",
    "toradians",
];

// --- the types ------------------------------------------------------------

/// The constant names in scope with their exact values, and the Pine types of
/// the folded names beside them (the Python `env` / `types` pair).
#[derive(Clone, Debug, Default)]
struct Env {
    values: HashMap<String, Num>,
    types: HashMap<String, String>,
}

/// Pine type of a subtree that folded to a constant (`const_type`).
fn const_type(ast: &Ast, node: NodeId, env: &Env) -> String {
    match ast.kind(node) {
        Kind::Constant { value, .. } => match value {
            Constant::Bool(_) => BOOL,
            Constant::Int(_) => INT,
            Constant::Float(_) => FLOAT,
            Constant::Str(_) => STR,
            _ => UNKNOWN,
        }
        .to_string(),
        Kind::Name { id, .. } => {
            if env.values.contains_key(id) {
                env.types.get(id).cloned().unwrap_or_else(|| UNKNOWN.to_string())
            } else {
                UNKNOWN.to_string()
            }
        }
        Kind::UnaryOp { op, operand } => unaryop_type(*op, &const_type(ast, *operand, env)).into(),
        Kind::BinOp { left, op, right } => {
            binop_type(*op, &const_type(ast, *left, env), &const_type(ast, *right, env)).into()
        }
        Kind::IfExp { body, orelse, .. } => {
            join(&const_type(ast, *body, env), &const_type(ast, *orelse, env))
        }
        Kind::Call { func, args, .. } => const_call_type(ast, *func, args, env),
        _ => UNKNOWN.to_string(),
    }
}

/// Pine type of a folded `lib.math.*` call (`_const_call_type`).
fn const_call_type(ast: &Ast, func: NodeId, args: &[NodeId], env: &Env) -> String {
    let mut parts = Vec::new();
    let mut current = func;
    while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
        parts.push(attr.as_str());
        current = *value;
    }
    let Kind::Name { id, .. } = ast.kind(current) else {
        return UNKNOWN.to_string();
    };
    parts.push(id);
    parts.reverse();
    let dotted = parts.join(".");
    // ``lib.math.floor`` -> ``math.floor``
    let key = dotted.split_once('.').map_or("", |(_, key)| key);
    let mut rule = lib_type_override(if key.is_empty() { &dotted } else { key });
    if let Some(CallRule::ByArity(split)) = rule {
        rule = split.iter().find(|&&(count, _)| count == args.len()).map(|&(_, inner)| inner);
    }
    let Some(CallRule::Is(rule)) = rule else {
        return UNKNOWN.to_string();
    };
    if rule == "all_int" {
        let types: Vec<String> = args.iter().map(|&arg| const_type(ast, arg, env)).collect();
        if types.is_empty() || types.iter().any(|ty| !is_numeric(ty)) {
            return UNKNOWN.to_string();
        }
        return if types.iter().all(|ty| ty == INT) { INT } else { FLOAT }.to_string();
    }
    if let Some(index) = rule.strip_prefix("arg")
        && !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
    {
        return match index.parse::<usize>() {
            Ok(index) if index < args.len() => const_type(ast, args[index], env),
            _ => UNKNOWN.to_string(),
        };
    }
    rule.to_string()
}

/// Give an emitted literal its Pine type, when there is one to give (`_stamp`).
fn stamp(ast: &mut Ast, node: NodeId, ty: &str) -> NodeId {
    if ty != UNKNOWN {
        set_ty(ast, node, ty);
    }
    node
}

/// The literal that replaces a folded subtree, or the subtree itself when it
/// already is that literal; stamped with the subtree's type.
fn embed(ast: &mut Ast, node: NodeId, value: Num, env: &Env) -> NodeId {
    let quantized = quantize_embed(value);
    let ty = const_type(ast, node, env);
    let target = if is_literal(ast, node, quantized) {
        node
    } else {
        let literal = ast.add(Kind::Constant { value: constant(quantized), kind: None });
        copy_location(ast, literal, node);
        literal
    };
    stamp(ast, target, &ty)
}

// --- the expression folder ------------------------------------------------

/// Every name a statement (sub)tree can (re)bind or delete (`_assigned_names`).
fn assigned_names(ast: &Ast, node: NodeId, names: &mut HashSet<String>) {
    for n in walk(ast, node) {
        match ast.kind(n) {
            Kind::Name { id, ctx: ExprContext::Store | ExprContext::Del } => {
                names.insert(id.clone());
            }
            Kind::MatchAs { name: Some(name), .. } | Kind::MatchStar { name: Some(name) } => {
                names.insert(name.clone());
            }
            Kind::MatchMapping { rest: Some(rest), .. } => {
                names.insert(rest.clone());
            }
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                names.insert(name.clone());
            }
            Kind::Global { names: declared } | Kind::Nonlocal { names: declared } => {
                names.extend(declared.iter().cloned());
            }
            &Kind::NamedExpr { target, .. } => {
                if let Kind::Name { id, .. } = ast.kind(target) {
                    names.insert(id.clone());
                }
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                names.insert(name.clone());
            }
            _ => {}
        }
    }
}

/// Replaces every maximal constant subtree of one expression with its quantized
/// literal (`_ExprFolder`); the first error stops the walk.
struct Folder<'e> {
    env: &'e Env,
    /// Local names of `pine_cast.cast_bool`, the call a Pine `bool()` compiles to
    bool_casts: &'e HashSet<String>,
    /// The casts of the innermost module or function scope: a lambda or
    /// comprehension body resolves its free names there, never in a class body
    lexical_casts: &'e HashSet<String>,
    /// The constants of that scope, for the same reason
    lexical_env: &'e Env,
    /// The names of that scope that are never rebound: a lambda or generator
    /// body runs later, when only these still hold the value seen here
    stable: &'e HashSet<String>,
    error: Option<Error>,
}

impl Folder<'_> {
    /// The lexical environment without the names a lambda or comprehension binds;
    /// `deferred` when its body runs after the enclosing expression, which keeps
    /// only the stable names.
    fn shadowed(&self, bound: &HashSet<String>, deferred: bool) -> Env {
        let mut inner = Env::default();
        for (name, value) in &self.lexical_env.values {
            if !bound.contains(name) && (!deferred || self.stable.contains(name)) {
                inner.values.insert(name.clone(), *value);
            }
        }
        inner.types = self.lexical_env.types.clone();
        inner
    }

    /// The lexical cast names a lambda or comprehension scope does not rebind.
    fn shadowed_casts(&self, bound: &HashSet<String>) -> HashSet<String> {
        self.lexical_casts.iter().filter(|name| !bound.contains(*name)).cloned().collect()
    }

    /// `bool()` of a constant number folded to its exact truth value
    /// (`_ExprFolder._fold_bool_cast`); `None` where the call is no such cast.
    fn fold_bool_cast(&self, ast: &mut Ast, id: NodeId) -> Result<Option<NodeId>, Error> {
        let Kind::Call { func, args, keywords } = ast.kind(id) else {
            return Ok(None);
        };
        let is_cast =
            matches!(ast.kind(*func), Kind::Name { id, .. } if self.bool_casts.contains(id));
        if !is_cast || args.len() != 1 || !keywords.is_empty() {
            return Ok(None);
        }
        let arg = args[0];
        let Some(value) = try_eval(ast, arg, &self.env.values)? else {
            return Ok(None);
        };
        let literal =
            ast.add(Kind::Constant { value: Constant::Bool(value.float() != 0.0), kind: None });
        copy_location(ast, literal, id);
        Ok(Some(stamp(ast, literal, BOOL)))
    }

    fn fail(&mut self, error: Option<Error>) {
        if self.error.is_none() {
            self.error = error;
        }
    }
}

impl Transformer for Folder<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        if ast.kind(id).class().category == Category::Expr {
            match try_eval(ast, id, &self.env.values) {
                Err(error) => {
                    self.error = Some(error);
                    return Visited::Node(id);
                }
                Ok(Some(value)) => return Visited::Node(embed(ast, id, value, self.env)),
                Ok(None) => {}
            }
            if !self.bool_casts.is_empty() {
                match self.fold_bool_cast(ast, id) {
                    Err(error) => {
                        self.error = Some(error);
                        return Visited::Node(id);
                    }
                    Ok(Some(literal)) => return Visited::Node(literal),
                    Ok(None) => {}
                }
            }
        }
        match ast.kind(id) {
            &Kind::Subscript { value, slice, .. } => {
                // ``x[1]`` on a constant-valued series variable is NOT the constant:
                // bar 0 reads na from the history. Keep a bare subscripted name; only
                // the index and any composite value expression are folded.
                if !matches!(ast.kind(value), Kind::Name { .. }) {
                    let Visited::Node(new) = self.visit(ast, value) else { unreachable!() };
                    if let Kind::Subscript { value, .. } = ast.kind_mut(id) {
                        *value = new;
                    }
                }
                let Visited::Node(new) = self.visit(ast, slice) else { unreachable!() };
                if let Kind::Subscript { slice, .. } = ast.kind_mut(id) {
                    *slice = new;
                }
            }
            &Kind::Lambda { args, body } => {
                let mut bound = HashSet::new();
                if let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
                    ast.kind(args)
                {
                    let all = args.iter().chain(posonlyargs).chain(kwonlyargs);
                    for &arg in all.chain(vararg.iter()).chain(kwarg.iter()) {
                        if let Kind::Arg { arg, .. } = ast.kind(arg) {
                            bound.insert(arg.clone());
                        }
                    }
                }
                let inner = self.shadowed(&bound, true);
                let casts = self.shadowed_casts(&bound);
                let mut folder = Folder {
                    env: &inner,
                    bool_casts: &casts,
                    lexical_casts: &casts,
                    lexical_env: &inner,
                    stable: self.stable,
                    error: None,
                };
                let Visited::Node(new) = folder.visit(ast, body) else { unreachable!() };
                self.fail(folder.error);
                if let Kind::Lambda { body, .. } = ast.kind_mut(id) {
                    *body = new;
                }
            }
            Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. } => {
                // The outermost iterable is evaluated in the enclosing scope, the
                // rest of the comprehension in its own
                let (elts, generators) = match ast.kind(id) {
                    Kind::ListComp { elt, generators }
                    | Kind::SetComp { elt, generators }
                    | Kind::GeneratorExp { elt, generators } => (vec![*elt], generators.clone()),
                    Kind::DictComp { key, value, generators } => {
                        (vec![*key, *value], generators.clone())
                    }
                    _ => unreachable!(),
                };
                let Kind::Comprehension { iter: first_iter, .. } = *ast.kind(generators[0]) else {
                    unreachable!()
                };
                let Visited::Node(new) = self.visit(ast, first_iter) else { unreachable!() };
                if let Kind::Comprehension { iter, .. } = ast.kind_mut(generators[0]) {
                    *iter = new;
                }
                let mut bound = HashSet::new();
                assigned_names(ast, id, &mut bound);
                let deferred = matches!(ast.kind(id), Kind::GeneratorExp { .. });
                let inner = self.shadowed(&bound, deferred);
                let casts = self.shadowed_casts(&bound);
                let mut folder = Folder {
                    env: &inner,
                    bool_casts: &casts,
                    lexical_casts: &casts,
                    lexical_env: &inner,
                    stable: self.stable,
                    error: None,
                };
                let mut folded = Vec::with_capacity(elts.len());
                for elt in elts {
                    let Visited::Node(new) = folder.visit(ast, elt) else { unreachable!() };
                    folded.push(new);
                }
                match ast.kind_mut(id) {
                    Kind::ListComp { elt, .. }
                    | Kind::SetComp { elt, .. }
                    | Kind::GeneratorExp { elt, .. } => *elt = folded[0],
                    Kind::DictComp { key, value, .. } => {
                        *key = folded[0];
                        *value = folded[1];
                    }
                    _ => unreachable!(),
                }
                for (index, &generator) in generators.iter().enumerate() {
                    let Kind::Comprehension { target, iter, ref ifs, .. } = *ast.kind(generator)
                    else {
                        unreachable!()
                    };
                    let ifs = ifs.clone();
                    let Visited::Node(target) = folder.visit(ast, target) else { unreachable!() };
                    let iter = if index == 0 {
                        iter
                    } else {
                        let Visited::Node(new) = folder.visit(ast, iter) else { unreachable!() };
                        new
                    };
                    let mut conditions = Vec::with_capacity(ifs.len());
                    for condition in ifs {
                        let Visited::Node(new) = folder.visit(ast, condition) else {
                            unreachable!()
                        };
                        conditions.push(new);
                    }
                    if let Kind::Comprehension { target: t, iter: i, ifs, .. } =
                        ast.kind_mut(generator)
                    {
                        *t = target;
                        *i = iter;
                        *ifs = conditions;
                    }
                }
                self.fail(folder.error);
            }
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

/// Fold one expression in an environment (`ConstFoldTransformer._fold`).
fn fold(
    ast: &mut Ast,
    node: NodeId,
    env: &Env,
    bool_casts: &HashSet<String>,
    lexical_casts: &HashSet<String>,
    lexical_env: &Env,
    stable: &HashSet<String>,
) -> Result<NodeId, Error> {
    let mut folder = Folder { env, bool_casts, lexical_casts, lexical_env, stable, error: None };
    let Visited::Node(new) = folder.visit(ast, node) else { unreachable!() };
    match folder.error {
        Some(error) => Err(error),
        None => Ok(new),
    }
}

// --- what the fold knows before it starts -----------------------------------

/// Whether an annotation names a state-carrying type (`_is_stateful_annotation`).
fn is_stateful_annotation(ast: &Ast, node: NodeId) -> bool {
    let node = match ast.kind(node) {
        Kind::Subscript { value, .. } => *value,
        _ => node,
    };
    let name = match ast.kind(node) {
        Kind::Attribute { attr, .. } => attr.as_str(),
        Kind::Name { id, .. } => id.as_str(),
        _ => "",
    };
    name.starts_with("Persistent") || name.starts_with("IBPersistent")
}

fn is_function(kind: &Kind) -> bool {
    matches!(kind, Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

/// What the fold needs to know about a module before it starts (`_ModuleFacts`):
/// per scope, the state-carrying names stored more than once; the names any
/// `global` / `nonlocal` declares; whether the module has a `:=` at all.
struct ModuleFacts {
    outer_writes: HashSet<String>,
    has_walrus: bool,
    blocked: HashMap<NodeId, HashSet<String>>,
}

impl ModuleFacts {
    fn new(ast: &Ast, tree: NodeId) -> ModuleFacts {
        let mut outer_writes = HashSet::new();
        let mut has_walrus = false;
        // Per scope, what its OWN region holds (nested function bodies are regions
        // of their own, added in below)
        let mut stateful: HashMap<NodeId, HashSet<String>> = HashMap::new();
        let mut stores: HashMap<NodeId, HashMap<String, usize>> = HashMap::new();
        let mut enclosing: HashMap<NodeId, NodeId> = HashMap::new();
        let mut scopes: Vec<NodeId> = Vec::new();
        let mut regions: Vec<(NodeId, Vec<NodeId>)> = vec![(tree, vec![tree])];
        while let Some((scope, mut pending)) = regions.pop() {
            let mut own_stateful = HashSet::new();
            let mut own_stores: HashMap<String, usize> = HashMap::new();
            while let Some(node) = pending.pop() {
                let kind = ast.kind(node);
                if let Kind::Name { id, ctx } = kind {
                    if *ctx == ExprContext::Store {
                        *own_stores.entry(id.clone()).or_default() += 1;
                    }
                    continue;
                }
                let function = is_function(kind);
                match kind {
                    Kind::FunctionDef { body, .. } | Kind::AsyncFunctionDef { body, .. } => {
                        // The body is the function's own region; the name, decorators,
                        // parameters and annotations stand in the enclosing one
                        enclosing.insert(node, scope);
                        scopes.push(node);
                        regions.push((node, body.clone()));
                    }
                    &Kind::AnnAssign { target, annotation, .. } => {
                        if let Kind::Name { id, .. } = ast.kind(target)
                            && is_stateful_annotation(ast, annotation)
                        {
                            own_stateful.insert(id.clone());
                        }
                    }
                    Kind::Global { names } | Kind::Nonlocal { names } => {
                        outer_writes.extend(names.iter().cloned());
                    }
                    Kind::NamedExpr { .. } => has_walrus = true,
                    _ => {}
                }
                let class = kind.class();
                for (index, field) in class.fields.iter().enumerate() {
                    match kind.field(index) {
                        FieldRef::Node(child) => pending.push(*child),
                        FieldRef::OptNode(child) => pending.extend(child.iter().copied()),
                        FieldRef::Nodes(items) if !(function && field.name == "body") => {
                            pending.extend(items.iter().copied());
                        }
                        FieldRef::OptNodes(items) => pending.extend(items.iter().flatten()),
                        _ => {}
                    }
                }
            }
            stateful.insert(scope, own_stateful);
            stores.insert(scope, own_stores);
        }
        // A nested function is found after the one enclosing it: in reverse, every
        // region is complete before it is added to its enclosing scope
        for &scope in scopes.iter().rev() {
            let outer = enclosing[&scope];
            let inner_stateful = stateful[&scope].clone();
            stateful.get_mut(&outer).expect("a scope").extend(inner_stateful);
            let inner_stores = stores[&scope].clone();
            let outer_stores = stores.get_mut(&outer).expect("a scope");
            for (name, count) in inner_stores {
                *outer_stores.entry(name).or_default() += count;
            }
        }
        // The AnnAssign target itself is one store: only additional ones mutate
        let blocked = stateful
            .into_iter()
            .map(|(scope, names)| {
                let counts = &stores[&scope];
                let names = names
                    .into_iter()
                    .filter(|name| counts.get(name).copied().unwrap_or(0) > 1)
                    .collect();
                (scope, names)
            })
            .collect();
        ModuleFacts { outer_writes, has_walrus, blocked }
    }
}

/// The name a wildcard import binds in [`statement_bindings`]: any name.
const ANY_NAME: &str = "*";

/// The bindings an import alias makes (`alias.asname or alias.name`, the first
/// dotted part for a plain `import`).
fn alias_binding(ast: &Ast, alias: NodeId, plain_import: bool) -> String {
    let Kind::Alias { name, asname } = ast.kind(alias) else {
        return String::new();
    };
    match asname {
        Some(asname) => asname.clone(),
        None if plain_import => name.split('.').next().unwrap_or(name).to_string(),
        None => name.clone(),
    }
}

/// The fields that hold statement blocks (and the handlers / cases holding blocks).
const BLOCK_FIELDS: [&str; 5] = ["body", "orelse", "finalbody", "handlers", "cases"];

/// Bindings in one Python scope, counted, including unreachable assignments
/// (`_ScopeBindings`).
#[derive(Default)]
struct ScopeBindings {
    names: HashMap<String, usize>,
    wildcard_import: bool,
}

impl ScopeBindings {
    fn new(ast: &Ast, body: &[NodeId]) -> ScopeBindings {
        let mut bindings = ScopeBindings::default();
        for &statement in body {
            bindings.visit(ast, statement);
        }
        bindings
    }

    fn count(&mut self, name: &str) {
        *self.names.entry(name.to_string()).or_default() += 1;
    }

    fn visit_all(&mut self, ast: &Ast, nodes: &[NodeId]) {
        for &node in nodes {
            self.visit(ast, node);
        }
    }

    fn generic(&mut self, ast: &Ast, node: NodeId) {
        for child in pyne_ast::walk::children(ast, node) {
            self.visit(ast, child);
        }
    }

    fn visit(&mut self, ast: &Ast, node: NodeId) {
        match ast.kind(node) {
            Kind::Name { id, ctx } => {
                if matches!(ctx, ExprContext::Store | ExprContext::Del) {
                    self.count(id);
                }
            }
            Kind::FunctionDef { name, args, decorator_list, returns, .. }
            | Kind::AsyncFunctionDef { name, args, decorator_list, returns, .. } => {
                self.count(name);
                self.visit_all(ast, decorator_list);
                self.visit(ast, *args);
                if let Some(returns) = returns {
                    self.visit(ast, *returns);
                }
            }
            Kind::Lambda { args, .. } => self.visit(ast, *args),
            Kind::ClassDef { name, bases, keywords, decorator_list, .. } => {
                self.count(name);
                self.visit_all(ast, bases);
                self.visit_all(ast, keywords);
                self.visit_all(ast, decorator_list);
            }
            Kind::Import { names } => {
                for &alias in names {
                    let bound = alias_binding(ast, alias, true);
                    self.count(&bound);
                }
            }
            Kind::ImportFrom { names, .. } => {
                for &alias in names {
                    if matches!(ast.kind(alias), Kind::Alias { name, .. } if name == "*") {
                        self.wildcard_import = true;
                    } else {
                        let bound = alias_binding(ast, alias, false);
                        self.count(&bound);
                    }
                }
            }
            Kind::ExceptHandler { name, .. } => {
                if let Some(name) = name {
                    self.count(name);
                }
                self.generic(ast, node);
            }
            Kind::MatchAs { name, .. } | Kind::MatchStar { name } => {
                if let Some(name) = name {
                    self.count(name);
                }
                self.generic(ast, node);
            }
            Kind::MatchMapping { rest, .. } => {
                if let Some(rest) = rest {
                    self.count(rest);
                }
                self.generic(ast, node);
            }
            _ => self.generic(ast, node),
        }
    }
}

// --- the statement walk ---------------------------------------------------

/// The names of a scope whose value can never change once bound
/// (`_stable_names`): bound exactly once, never declared `global` / `nonlocal`
/// anywhere, and not a parameterized annotation (those can carry
/// Series/Persistent state instead of a native scalar value).
fn stable_names(
    ast: &Ast,
    body: &[NodeId],
    bindings: &ScopeBindings,
    outer_writes: &HashSet<String>,
) -> HashSet<String> {
    if bindings.wildcard_import {
        return HashSet::new();
    }
    let mut names: HashSet<String> = bindings
        .names
        .iter()
        .filter(|&(name, &count)| count == 1 && !outer_writes.contains(name))
        .map(|(name, _)| name.clone())
        .collect();
    for &stmt in body {
        if let &Kind::AnnAssign { target, annotation, .. } = ast.kind(stmt)
            && let Kind::Name { id, .. } = ast.kind(target)
            && matches!(ast.kind(annotation), Kind::Subscript { .. })
        {
            names.remove(id);
        }
    }
    names
}

/// The transform's state over one module (`ConstFoldTransformer`). The Python
/// environments are dicts several places hold at once (the module's `env` is
/// also its `_global_env`), so they live in an arena and are passed by index.
struct ConstFold {
    envs: Vec<Env>,
    /// The module-level environment a function captures from, as an arena index
    global_env: usize,
    global_names: HashSet<String>,
    /// The Persistent names a later store mutates in the current scope
    blocked: HashSet<String>,
    scope_blocked: HashMap<NodeId, HashSet<String>>,
    has_walrus: bool,
    bindings_memo: HashMap<NodeId, HashSet<String>>,
    /// Local names of `pine_cast.cast_bool` in the current scope
    bool_casts: HashSet<String>,
    /// The casts of the innermost enclosing module or function scope: a class
    /// body or a method resolves its free names there, never in a class body
    lexical_casts: HashSet<String>,
    /// The environment a lambda or comprehension body sees inside a class body,
    /// as an arena index; `None` outside one, where it sees the scope's own
    lexical_env: Option<usize>,
    /// The names of the innermost module or function scope that are never
    /// rebound, the only ones a lambda or generator body may fold
    stable: HashSet<String>,
    /// The names any `global` / `nonlocal` statement of the module declares
    outer_writes: HashSet<String>,
}

impl ConstFold {
    fn new_env(&mut self, env: Env) -> usize {
        self.envs.push(env);
        self.envs.len() - 1
    }

    /// A copy of an environment (`dict(env), dict(types)`).
    fn copy_env(&mut self, env: usize) -> usize {
        let copy = self.envs[env].clone();
        self.new_env(copy)
    }

    fn fold(&self, ast: &mut Ast, node: NodeId, env: usize) -> Result<NodeId, Error> {
        let lexical = &self.envs[self.lexical_env.unwrap_or(env)];
        fold(
            ast,
            node,
            &self.envs[env],
            &self.bool_casts,
            &self.lexical_casts,
            lexical,
            &self.stable,
        )
    }

    fn fold_opt(
        &self,
        ast: &mut Ast,
        node: Option<NodeId>,
        env: usize,
    ) -> Result<Option<NodeId>, Error> {
        node.map(|node| self.fold(ast, node, env)).transpose()
    }

    /// `_statement_bindings`: a statement's bindings, remembered per statement.
    fn statement_bindings(&mut self, ast: &Ast, stmt: NodeId) -> HashSet<String> {
        if let Some(names) = self.bindings_memo.get(&stmt) {
            return names.clone();
        }
        let mut found = HashSet::new();
        let kind = ast.kind(stmt);
        match kind {
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                found.insert(name.clone());
            }
            Kind::Global { names } | Kind::Nonlocal { names } => {
                found.extend(names.iter().cloned());
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                found.insert(name.clone());
            }
            Kind::Import { names } => {
                found.extend(names.iter().map(|&alias| alias_binding(ast, alias, true)));
            }
            Kind::ImportFrom { names, .. } => {
                found.extend(names.iter().map(|&alias| alias_binding(ast, alias, false)));
            }
            _ => {}
        }
        let class = kind.class();
        for (index, field) in class.fields.iter().enumerate() {
            match kind.field(index) {
                FieldRef::Nodes(items) if BLOCK_FIELDS.contains(&field.name) => {
                    for &item in items {
                        found.extend(self.statement_bindings(ast, item));
                    }
                }
                FieldRef::Node(child) => assigned_names(ast, *child, &mut found),
                FieldRef::OptNode(Some(child)) => assigned_names(ast, *child, &mut found),
                FieldRef::Nodes(items) => {
                    for &item in items {
                        assigned_names(ast, item, &mut found);
                    }
                }
                FieldRef::OptNodes(items) => {
                    for &item in items.iter().flatten() {
                        assigned_names(ast, item, &mut found);
                    }
                }
                _ => {}
            }
        }
        self.bindings_memo.insert(stmt, found.clone());
        found
    }

    /// Drop every name a statement can (re)bind from the environment (`_kill`).
    fn kill(&mut self, ast: &Ast, stmt: NodeId, env: usize) {
        if self.envs[env].values.is_empty() {
            return;
        }
        let names = self.statement_bindings(ast, stmt);
        let values = &mut self.envs[env].values;
        if names.contains(ANY_NAME) {
            values.clear();
            return;
        }
        for name in &names {
            values.remove(name);
        }
    }

    fn pop(&mut self, env: usize, name: &str) {
        self.envs[env].values.remove(name);
    }

    /// Record a folded name: its type always, its value unless a later store mutates it.
    fn bind(&mut self, env: usize, name: &str, value: Num, ty: &str) {
        let blocked = self.blocked.contains(name);
        let env = &mut self.envs[env];
        env.types.insert(name.to_string(), ty.to_string());
        if blocked {
            env.values.remove(name);
        } else {
            env.values.insert(name.to_string(), value);
        }
    }

    fn process_body(&mut self, ast: &mut Ast, body: &[NodeId], env: usize) -> Result<(), Error> {
        for &stmt in body {
            self.process_stmt(ast, stmt, env)?;
        }
        Ok(())
    }

    /// Process a block in a copy of the environment.
    fn process_branch(&mut self, ast: &mut Ast, body: &[NodeId], env: usize) -> Result<(), Error> {
        let copy = self.copy_env(env);
        self.process_body(ast, body, copy)
    }

    fn process_function(&mut self, ast: &mut Ast, stmt: NodeId, env: usize) -> Result<(), Error> {
        let (name, args, body) = match ast.kind(stmt) {
            Kind::FunctionDef { name, args, body, .. }
            | Kind::AsyncFunctionDef { name, args, body, .. } => {
                (name.clone(), *args, body.clone())
            }
            _ => unreachable!(),
        };
        let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(args) else {
            return Err(unsupported("a function without arguments"));
        };
        let (defaults, kw_defaults) = (defaults.clone(), kw_defaults.clone());
        for (index, default) in defaults.into_iter().enumerate() {
            let new = self.fold(ast, default, env)?;
            if let Kind::Arguments { defaults, .. } = ast.kind_mut(args) {
                defaults[index] = new;
            }
        }
        for (index, default) in kw_defaults.into_iter().enumerate() {
            let Some(default) = default else { continue };
            let new = self.fold(ast, default, env)?;
            if let Kind::Arguments { kw_defaults, .. } = ast.kind_mut(args) {
                kw_defaults[index] = Some(new);
            }
        }
        self.pop(env, &name);
        let local_bindings = ScopeBindings::new(ast, &body);
        let mut params = HashSet::new();
        if let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } = ast.kind(args)
        {
            let all = posonlyargs.iter().chain(args).chain(kwonlyargs);
            for &arg in all.chain(vararg.iter()).chain(kwarg.iter()) {
                if let Kind::Arg { arg, .. } = ast.kind(arg) {
                    params.insert(arg.clone());
                }
            }
        }
        let mut bound: HashSet<String> = local_bindings.names.keys().cloned().collect();
        bound.extend(params.iter().cloned());
        let globals = &self.envs[self.global_env];
        let mut captured = Env::default();
        for (name, value) in &globals.values {
            if self.global_names.contains(name) && !bound.contains(name) {
                let ty = globals.types.get(name).ok_or_else(|| {
                    unsupported(format!("a module constant {name} without a type"))
                })?;
                captured.values.insert(name.clone(), *value);
                captured.types.insert(name.clone(), ty.clone());
            }
        }
        let outer_globals = self.global_env;
        self.global_env = self.new_env(captured.clone());
        let blocked = self
            .scope_blocked
            .get(&stmt)
            .cloned()
            .ok_or_else(|| unsupported("a function the module facts did not see"))?;
        let outer_blocked = std::mem::replace(&mut self.blocked, blocked);
        let casts: HashSet<String> =
            self.lexical_casts.iter().filter(|name| !bound.contains(*name)).cloned().collect();
        let outer_casts = std::mem::replace(&mut self.bool_casts, casts.clone());
        let outer_lexical = std::mem::replace(&mut self.lexical_casts, casts);
        let outer_env = self.lexical_env.take();
        let mut stable: HashSet<String> = captured.values.keys().cloned().collect();
        stable.extend(
            stable_names(ast, &body, &local_bindings, &self.outer_writes)
                .into_iter()
                .filter(|name| !params.contains(name)),
        );
        let outer_stable = std::mem::replace(&mut self.stable, stable);
        let local = self.new_env(captured);
        let result = self.process_body(ast, &body, local);
        self.stable = outer_stable;
        self.lexical_env = outer_env;
        self.bool_casts = outer_casts;
        self.lexical_casts = outer_lexical;
        self.blocked = outer_blocked;
        self.global_env = outer_globals;
        result
    }

    fn process_stmt(&mut self, ast: &mut Ast, stmt: NodeId, env: usize) -> Result<(), Error> {
        // A walrus rebinding anywhere in the statement makes that name untrackable
        // from here on (evaluation order inside one statement is not modeled). A
        // module without any walrus is not searched
        if self.has_walrus {
            for n in walk(ast, stmt) {
                if let &Kind::NamedExpr { target, .. } = ast.kind(n)
                    && let Kind::Name { id, .. } = ast.kind(target)
                {
                    let id = id.clone();
                    self.pop(env, &id);
                }
            }
        }

        match ast.kind(stmt).clone() {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                self.process_function(ast, stmt, env)?;
            }
            Kind::ClassDef { name, body, .. } => {
                self.pop(env, &name);
                let bound = ScopeBindings::new(ast, &body).names;
                let casts =
                    self.lexical_casts.iter().filter(|name| !bound.contains_key(*name)).cloned();
                let outer_casts = std::mem::replace(&mut self.bool_casts, casts.collect());
                // A class binding is invisible to a lambda or comprehension body:
                // it sees the module constants a function would capture
                let globals = &self.envs[self.global_env];
                let mut captured = Env::default();
                for (name, value) in &globals.values {
                    if self.global_names.contains(name) {
                        let ty = globals.types.get(name).ok_or_else(|| {
                            unsupported(format!("a module constant {name} without a type"))
                        })?;
                        captured.values.insert(name.clone(), *value);
                        captured.types.insert(name.clone(), ty.clone());
                    }
                }
                let captured = self.new_env(captured);
                let outer_env = self.lexical_env.replace(captured);
                let fresh = self.new_env(Env::default());
                let result = self.process_body(ast, &body, fresh);
                self.lexical_env = outer_env;
                self.bool_casts = outer_casts;
                result?;
            }
            Kind::Assign { targets, value, .. } => {
                let folded = try_eval(ast, value, &self.envs[env].values)?;
                let names: Vec<String> = targets
                    .iter()
                    .filter_map(|&target| match ast.kind(target) {
                        Kind::Name { id, .. } => Some(id.clone()),
                        _ => None,
                    })
                    .collect();
                match folded {
                    Some(folded) if names.len() == targets.len() => {
                        // The emitted right side is the embedding-quantized literal (a
                        // residual runtime read -- series history, nested scope -- must
                        // see the embedded value); the environment keeps the exact
                        // double for const chains
                        let ty = const_type(ast, value, &self.envs[env]);
                        let new = embed(ast, value, folded, &self.envs[env]);
                        set_value(ast, stmt, new);
                        for name in &names {
                            self.bind(env, name, folded, &ty);
                        }
                    }
                    _ => {
                        let new = self.fold(ast, value, env)?;
                        set_value(ast, stmt, new);
                        self.kill(ast, stmt, env);
                    }
                }
            }
            Kind::AnnAssign { target, annotation, value, .. } => {
                let Some(value) = value else {
                    if let Kind::Name { id, .. } = ast.kind(target) {
                        let id = id.clone();
                        self.pop(env, &id);
                    }
                    return Ok(());
                };
                let folded = try_eval(ast, value, &self.envs[env].values)?;
                let target_name = match ast.kind(target) {
                    Kind::Name { id, .. } => Some(id.clone()),
                    _ => None,
                };
                match (folded, target_name) {
                    (Some(folded), Some(name)) => {
                        // An explicit annotation is a declaration, so it outranks what
                        // the folded initializer happens to be
                        let declared = annotation_type(ast, Some(annotation), &Classes::new())?;
                        let ty = if declared != UNKNOWN {
                            declared
                        } else {
                            const_type(ast, value, &self.envs[env])
                        };
                        let quantized = quantize_embed(folded);
                        let new = if is_literal(ast, value, quantized) {
                            value
                        } else {
                            let literal =
                                ast.add(Kind::Constant { value: constant(quantized), kind: None });
                            copy_location(ast, literal, value);
                            literal
                        };
                        stamp(ast, new, &ty);
                        set_value(ast, stmt, new);
                        self.bind(env, &name, folded, &ty);
                    }
                    (_, target_name) => {
                        let new = self.fold(ast, value, env)?;
                        set_value(ast, stmt, new);
                        if let Some(name) = target_name {
                            self.pop(env, &name);
                        }
                    }
                }
            }
            Kind::AugAssign { target, value, .. } => {
                let new = self.fold(ast, value, env)?;
                set_value(ast, stmt, new);
                if let Kind::Name { id, .. } = ast.kind(target) {
                    let id = id.clone();
                    self.pop(env, &id);
                }
            }
            Kind::If { test, body, orelse } => {
                let new = self.fold(ast, test, env)?;
                set_test(ast, stmt, new);
                self.process_branch(ast, &body, env)?;
                self.process_branch(ast, &orelse, env)?;
                self.kill(ast, stmt, env);
            }
            Kind::For { iter, body, orelse, .. } | Kind::AsyncFor { iter, body, orelse, .. } => {
                let new = self.fold(ast, iter, env)?;
                match ast.kind_mut(stmt) {
                    Kind::For { iter, .. } | Kind::AsyncFor { iter, .. } => *iter = new,
                    _ => unreachable!(),
                }
                // Names the loop assigns are unknown both inside (previous
                // iteration) and after it; a constant assigned inside the body
                // re-enters the environment past its own line
                self.kill(ast, stmt, env);
                self.process_branch(ast, &body, env)?;
                self.process_branch(ast, &orelse, env)?;
                self.kill(ast, stmt, env);
            }
            Kind::While { test, body, orelse } => {
                self.kill(ast, stmt, env);
                let new = self.fold(ast, test, env)?;
                set_test(ast, stmt, new);
                self.process_branch(ast, &body, env)?;
                self.process_branch(ast, &orelse, env)?;
                self.kill(ast, stmt, env);
            }
            Kind::With { items, body, .. } | Kind::AsyncWith { items, body, .. } => {
                for item in items {
                    let Kind::WithItem { context_expr, .. } = *ast.kind(item) else {
                        continue;
                    };
                    let new = self.fold(ast, context_expr, env)?;
                    if let Kind::WithItem { context_expr, .. } = ast.kind_mut(item) {
                        *context_expr = new;
                    }
                }
                self.kill(ast, stmt, env);
                self.process_branch(ast, &body, env)?;
                self.kill(ast, stmt, env);
            }
            Kind::Try { body, handlers, orelse, finalbody } => {
                self.kill(ast, stmt, env);
                self.process_branch(ast, &body, env)?;
                for handler in handlers {
                    if let Kind::ExceptHandler { body, .. } = ast.kind(handler) {
                        let body = body.clone();
                        self.process_branch(ast, &body, env)?;
                    }
                }
                self.process_branch(ast, &orelse, env)?;
                self.process_branch(ast, &finalbody, env)?;
                self.kill(ast, stmt, env);
            }
            Kind::Return { value } => {
                let new = self.fold_opt(ast, value, env)?;
                if let Kind::Return { value } = ast.kind_mut(stmt) {
                    *value = new;
                }
            }
            Kind::Expr { value } => {
                let new = self.fold(ast, value, env)?;
                set_value(ast, stmt, new);
            }
            Kind::Assert { test, msg } => {
                let test = self.fold(ast, test, env)?;
                set_test(ast, stmt, test);
                let msg = self.fold_opt(ast, msg, env)?;
                if let Kind::Assert { msg: slot, .. } = ast.kind_mut(stmt) {
                    *slot = msg;
                }
            }
            Kind::Raise { exc, cause } => {
                let exc = self.fold_opt(ast, exc, env)?;
                let cause = self.fold_opt(ast, cause, env)?;
                if let Kind::Raise { exc: exc_slot, cause: cause_slot } = ast.kind_mut(stmt) {
                    *exc_slot = exc;
                    *cause_slot = cause;
                }
            }
            Kind::Global { names } | Kind::Nonlocal { names } => {
                for name in names {
                    self.pop(env, &name);
                }
            }
            Kind::Pass | Kind::Break | Kind::Continue => {}
            // Unmodeled statement kind: fold nothing inside, kill what it can rebind
            _ => self.kill(ast, stmt, env),
        }
        Ok(())
    }
}

/// Set the `value` field of an `Assign`, `AnnAssign`, `AugAssign` or `Expr`.
fn set_value(ast: &mut Ast, stmt: NodeId, new: NodeId) {
    match ast.kind_mut(stmt) {
        Kind::Assign { value, .. } | Kind::AugAssign { value, .. } | Kind::Expr { value } => {
            *value = new;
        }
        Kind::AnnAssign { value, .. } => *value = Some(new),
        _ => unreachable!(),
    }
}

/// Set the `test` field of an `If`, `While` or `Assert`.
fn set_test(ast: &mut Ast, stmt: NodeId, new: NodeId) {
    match ast.kind_mut(stmt) {
        Kind::If { test, .. } | Kind::While { test, .. } | Kind::Assert { test, .. } => {
            *test = new;
        }
        _ => unreachable!(),
    }
}

/// Fold the constants of a module (`ConstFoldTransformer.visit`).
pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(unsupported("not a module"));
    };
    let body = body.clone();
    let facts = ModuleFacts::new(ast, module);
    let bindings = ScopeBindings::new(ast, &body);
    let global_names = stable_names(ast, &body, &bindings, &facts.outer_writes);
    // A name bound once, by the `pine_cast` import, is the cast itself
    let mut bool_casts = HashSet::new();
    for &stmt in &body {
        if let Kind::ImportFrom { module: Some(module), names, .. } = ast.kind(stmt)
            && module == PINE_CAST
        {
            for &alias in names {
                if let Kind::Alias { name, asname } = ast.kind(alias)
                    && name == "cast_bool"
                {
                    let bound = asname.clone().unwrap_or_else(|| name.clone());
                    if bindings.names.get(&bound) == Some(&1) {
                        bool_casts.insert(bound);
                    }
                }
            }
        }
    }
    let blocked = facts.blocked.get(&module).cloned().unwrap_or_default();
    let mut fold = ConstFold {
        envs: vec![Env::default()],
        global_env: 0,
        global_names: global_names.clone(),
        blocked,
        scope_blocked: facts.blocked,
        has_walrus: facts.has_walrus,
        bindings_memo: HashMap::new(),
        lexical_casts: bool_casts.clone(),
        bool_casts,
        lexical_env: None,
        stable: global_names.clone(),
        outer_writes: facts.outer_writes,
    };
    fold.process_body(ast, &body, 0)?;
    Ok(module)
}

/// `_try_eval` and `quantize_embed` of one expression, for the parity tests: the
/// embedded literal, `None` where the expression is no constant.
pub fn eval_embedded(ast: &Ast, node: NodeId) -> Result<Option<Constant>, Error> {
    Ok(try_eval(ast, node, &HashMap::new())?.map(|value| constant(quantize_embed(value))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize() {
        assert_eq!(quantize_float(0.21402930929524275), 0.2140293092952427);
        assert_eq!(quantize_float(0.27161597376132945), 0.2716159737613295);
        assert_eq!(quantize_float(-0.0), 0.0);
        assert_eq!(quantize_float(1e-4), 1e-4);
        assert_eq!(quantize_float(0.1), 0.1);
        assert_eq!(quantize_float(1.0 / 3.0), 0.3333333333333333);
        assert_eq!(quantize_float(-2.0 / 3.0), -0.6666666666666666);
        assert_eq!(quantize_float(0.12345678901234567), 0.1234567890123457);
    }

    #[test]
    fn mixed_compare() {
        assert_eq!(compare(Num::Int(1.0), Num::Float(1.0)), Ordering::Equal);
        assert_eq!(compare(Num::Int(1.0), Num::Float(1.5)), Ordering::Less);
        assert_eq!(compare(Num::Int(-2.0), Num::Float(-1.5)), Ordering::Less);
        assert_eq!(compare(Num::Float(-1.5), Num::Int(-2.0)), Ordering::Greater);
    }
}
