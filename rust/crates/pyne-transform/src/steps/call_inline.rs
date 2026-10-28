//! Port of `transformers/call_inline.py`: a call to a trivial stateless lib
//! wrapper (`math.abs`, `array.get`, ...) inside a function body is replaced by
//! the wrapper's own body, written out as one expression, from the generated
//! template registry (`call_inline_templates.json`, see [`data::inline_templates`]).
//! A guard a literal argument decides is resolved here, a non-trivial argument is
//! bound once by a walrus, and the argument evaluation order of the call is kept.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

use pyne_ast::walk::{
    Transformer, Visited, child_statements, children, copy_location, generic_visit, walk,
    walk_statements,
};
use pyne_ast::{
    Ast, BoolOperator, Category, CmpOperator, Constant, ExprContext, FieldKind, FieldMut, FieldRef,
    Int, Kind, Loc, NodeId, Operator, UnaryOperator, Value,
};
use serde_json::Value as Json;

use crate::data::{self, InlineFunction, InlineTemplates};
use crate::pine_type_rules::{TY_ATTR, stamp_lowering, ty_stamp};
use crate::{Context, Error};

/// The module every free name of an inlined body resolves through
const SUPPORT_MODULE: &str = "pynecore.core.inline_support";
/// Alias prefix of an anchored support name
const SUPPORT_ALIAS_PREFIX: &str = "__inl·";
/// Attribute tagging the root of a subtree that came from the call site's own
/// argument list
const ARG_MARK: &str = "_inl_argument";
/// The state vector parameter of the slot lowering (`state_param.DEFAULT_STATE_PARAM`)
const DEFAULT_STATE_PARAM: &str = "__state__";

fn support_alias(anchor: &str) -> String {
    format!("{SUPPORT_ALIAS_PREFIX}{anchor}__")
}

fn temp_name(counter: usize) -> String {
    format!("__inl{counter}·__")
}

fn placeholder_name(index: usize) -> String {
    format!("__inlp{index}·__")
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::Unsupported(message.into())
}

fn load(ast: &mut Ast, name: &str) -> NodeId {
    ast.add(Kind::Name { id: name.to_string(), ctx: ExprContext::Load })
}

fn is_expr(ast: &Ast, id: NodeId) -> bool {
    ast.kind(id).class().category == Category::Expr
}

/// Python's truth value of a marker the pass reads (`getattr(node, name, False)`).
fn marked(ast: &Ast, id: NodeId, name: &str) -> Result<bool, Error> {
    match ast.extra(id, name) {
        None | Some(Value::None) => Ok(false),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(other) => Err(unsupported(format!("a {name} marker {other:?}"))),
    }
}

/// `_copy_node`: the fields deeply (a node reached twice is copied twice), the
/// location and every stamped attribute as they are.
fn copy_node(ast: &mut Ast, id: NodeId) -> NodeId {
    let mut node = ast[id].clone();
    let class = node.kind.class();
    for (index, field) in class.fields.iter().enumerate() {
        match field.kind {
            FieldKind::Node | FieldKind::OptNode | FieldKind::Nodes | FieldKind::OptNodes => {}
            _ => continue,
        }
        match node.kind.field_mut(index) {
            FieldMut::Node(child) => *child = copy_node(ast, *child),
            FieldMut::OptNode(Some(child)) => *child = copy_node(ast, *child),
            FieldMut::Nodes(items) => {
                for child in items.iter_mut() {
                    *child = copy_node(ast, *child);
                }
            }
            FieldMut::OptNodes(items) => {
                for child in items.iter_mut().flatten() {
                    *child = copy_node(ast, *child);
                }
            }
            _ => {}
        }
    }
    let copy = ast.add_at(node.kind, node.loc);
    ast[copy].extra = node.extra;
    copy
}

/// `_arg_copy`: a detached copy of an actual argument, tagged as the user's own.
fn arg_copy(ast: &mut Ast, id: NodeId) -> NodeId {
    let copy = copy_node(ast, id);
    ast.set_extra(copy, ARG_MARK, Value::Bool(true));
    copy
}

/// `_stamp_emission`: anchor every node of the pass's own emission at the call
/// site's point. The argument subtrees are the user's code and keep their
/// locations; a call without a location leaves the emission unlocated, for
/// `fix_locations` to anchor with the enclosing statement.
fn stamp_emission(ast: &mut Ast, root: NodeId, call: NodeId) -> Result<(), Error> {
    let loc = ast[call].loc;
    let point = loc.lineno.map(|line| {
        let col = loc.col_offset.unwrap_or(0);
        Loc::new(line, col, line, col)
    });
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if marked(ast, id, ARG_MARK)? {
            continue;
        }
        if ast.kind(id).class().located {
            ast[id].loc = point.unwrap_or_default();
        }
        stack.extend(children(ast, id));
    }
    Ok(())
}

/// `_mark_exact`: mark the comparisons the pass emitted as exact, never entering an
/// argument subtree.
fn mark_exact(ast: &mut Ast, id: NodeId) -> Result<(), Error> {
    if marked(ast, id, ARG_MARK)? {
        return Ok(());
    }
    if matches!(ast.kind(id), Kind::Compare { .. }) {
        ast.set_extra(id, "pine_exact", Value::Bool(true));
    }
    for child in children(ast, id) {
        if is_expr(ast, child) {
            mark_exact(ast, child)?;
        }
    }
    Ok(())
}

/// `_func_path`: the dotted path of a plain name / attribute chain.
fn func_path(ast: &Ast, func: NodeId) -> Option<String> {
    if let Kind::Name { id, .. } = ast.kind(func) {
        return Some(id.clone());
    }
    let mut parts: Vec<&str> = Vec::new();
    let mut current = func;
    while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
        parts.push(attr);
        current = *value;
    }
    match ast.kind(current) {
        Kind::Name { id, .. } if !parts.is_empty() => {
            parts.push(id);
            parts.reverse();
            Some(parts.join("."))
        }
        _ => None,
    }
}

/// `_anchor_of`: the support attribute an emitted alias stands for.
fn anchor_of(name: &str) -> Option<&str> {
    name.strip_prefix(SUPPORT_ALIAS_PREFIX)?.strip_suffix("__")
}

// --- templates ---------------------------------------------------------------

/// A template constant (`[type, value]`, see `scripts/call_inline_collector.py`).
fn template_constant(spec: &Json) -> Result<Constant, Error> {
    let bad = || unsupported(format!("a template constant {spec}"));
    let Some([tag, value]) = spec.as_array().map(Vec::as_slice) else { return Err(bad()) };
    Ok(match (tag.as_str(), value) {
        (Some("none"), Json::Null) => Constant::None,
        (Some("bool"), Json::Bool(flag)) => Constant::Bool(*flag),
        (Some("str"), Json::String(text)) => Constant::Str(text.clone()),
        (Some("int"), Json::String(digits)) => match digits.parse::<i64>() {
            Ok(number) => Constant::Int(Int::Small(number)),
            Err(_) => Constant::Int(Int::Big(digits.clone())),
        },
        (Some("float"), Json::String(spelling)) => {
            Constant::Float(spelling.parse::<f64>().map_err(|_| bad())?)
        }
        _ => return Err(bad()),
    })
}

/// `_build`: one node of a derived template for a call site. A parameter read is a
/// tagged copy of its argument; every other node is built afresh and unlocated, for
/// [`stamp_emission`] to anchor at the call site.
fn build(ast: &mut Ast, spec: &Json, args: &[NodeId]) -> Result<NodeId, Error> {
    let bad = |what: &str| unsupported(format!("a template node with {what}: {spec}"));
    if let Some(index) = spec.get("param") {
        let index = index.as_u64().ok_or_else(|| bad("a parameter index"))? as usize;
        let arg = *args.get(index).ok_or_else(|| bad("a parameter beyond the arguments"))?;
        return Ok(arg_copy(ast, arg));
    }
    let class_name = spec.get("_type").and_then(Json::as_str).ok_or_else(|| bad("no class"))?;
    let class = pyne_ast::class_index(class_name).ok_or_else(|| bad("an unknown class"))?;
    let mut kind = Kind::empty(class);
    let fields = kind.class().fields;
    for (index, field) in fields.iter().enumerate() {
        let Some(value) = spec.get(field.name) else {
            if matches!(field.kind, FieldKind::Node) {
                return Err(bad("a required field missing"));
            }
            continue;
        };
        let text = |value: &Json| value.as_str().map(str::to_string).ok_or_else(|| bad("text"));
        match kind.field_mut(index) {
            FieldMut::Node(slot) => *slot = build(ast, value, args)?,
            FieldMut::OptNode(slot) => {
                *slot = if value.is_null() { None } else { Some(build(ast, value, args)?) }
            }
            FieldMut::Nodes(slot) => {
                for item in value.as_array().ok_or_else(|| bad("a node list"))? {
                    slot.push(build(ast, item, args)?);
                }
            }
            FieldMut::OptNodes(slot) => {
                for item in value.as_array().ok_or_else(|| bad("a node list"))? {
                    slot.push(if item.is_null() { None } else { Some(build(ast, item, args)?) });
                }
            }
            FieldMut::Str(slot) => *slot = text(value)?,
            FieldMut::OptStr(slot) => {
                *slot = if value.is_null() { None } else { Some(text(value)?) }
            }
            FieldMut::Strs(slot) => {
                for item in value.as_array().ok_or_else(|| bad("a text list"))? {
                    slot.push(text(item)?);
                }
            }
            FieldMut::Int(slot) => *slot = value.as_i64().ok_or_else(|| bad("an int"))?,
            FieldMut::OptInt(slot) => *slot = value.as_i64(),
            FieldMut::Constant(slot) => *slot = template_constant(value)?,
            FieldMut::ExprContext(slot) => {
                *slot = value
                    .as_str()
                    .and_then(ExprContext::from_name)
                    .ok_or_else(|| bad("a context"))?
            }
            FieldMut::Operator(slot) => {
                *slot = value
                    .as_str()
                    .and_then(Operator::from_name)
                    .ok_or_else(|| bad("an operator"))?
            }
            FieldMut::UnaryOperator(slot) => {
                *slot = value
                    .as_str()
                    .and_then(UnaryOperator::from_name)
                    .ok_or_else(|| bad("an operator"))?
            }
            FieldMut::BoolOperator(slot) => {
                *slot = value
                    .as_str()
                    .and_then(BoolOperator::from_name)
                    .ok_or_else(|| bad("an operator"))?
            }
            FieldMut::CmpOperators(slot) => {
                for item in value.as_array().ok_or_else(|| bad("an operator list"))? {
                    slot.push(
                        item.as_str()
                            .and_then(CmpOperator::from_name)
                            .ok_or_else(|| bad("an operator"))?,
                    );
                }
            }
        }
    }
    let node = ast.add(kind);
    if spec.get("arg").and_then(Json::as_bool).unwrap_or(false) {
        ast.set_extra(node, ARG_MARK, Value::Bool(true));
    }
    Ok(node)
}

/// `_varargs_template(...).render`: the fixed-arity expansion of `math.max` /
/// `math.min` over already bound operands.
fn render_varargs(
    ast: &mut Ast,
    builtin: &str,
    na_anchor: &str,
    builtins_anchor: &str,
    args: &[NodeId],
) -> NodeId {
    let operands = |ast: &mut Ast| {
        let elts = args.iter().map(|&arg| arg_copy(ast, arg)).collect();
        ast.add(Kind::Tuple { elts, ctx: ExprContext::Load })
    };
    let mut probes = Vec::with_capacity(args.len());
    for &arg in args {
        let left = arg_copy(ast, arg);
        let right = arg_copy(ast, arg);
        let compare =
            ast.add(Kind::Compare { left, ops: vec![CmpOperator::Eq], comparators: vec![right] });
        probes.push(ast.add(Kind::UnaryOp { op: UnaryOperator::Not, operand: compare }));
    }
    let test = ast.add(Kind::BoolOp { op: BoolOperator::Or, values: probes });
    let na_func = load(ast, &support_alias(na_anchor));
    let na_args = vec![operands(ast)];
    let body = ast.add(Kind::Call { func: na_func, args: na_args, keywords: Vec::new() });
    let module = load(ast, &support_alias(builtins_anchor));
    let func = ast.add(Kind::Attribute {
        value: module,
        attr: builtin.to_string(),
        ctx: ExprContext::Load,
    });
    let builtin_args = vec![operands(ast)];
    let orelse = ast.add(Kind::Call { func, args: builtin_args, keywords: Vec::new() });
    ast.add(Kind::IfExp { test, body, orelse })
}

// --- constant folding --------------------------------------------------------

/// A numeric literal (`_literal`): a plain int or a non-nan float, optionally negated.
#[derive(Clone, Debug)]
enum Number {
    Int(i64),
    /// An int beyond 64 bits, as its decimal digits with an optional `-`
    Big(String),
    Float(f64),
}

impl Number {
    fn negate(self) -> Number {
        match self {
            Number::Int(value) => match value.checked_neg() {
                Some(negated) => Number::Int(negated),
                None => Number::Big(value.unsigned_abs().to_string()),
            },
            Number::Big(digits) => match digits.strip_prefix('-') {
                Some(positive) => Number::Big(positive.to_string()),
                None => Number::Big(format!("-{digits}")),
            },
            Number::Float(value) => Number::Float(-value),
        }
    }

    /// An int's decimal digits with an optional `-`.
    fn digits(&self) -> String {
        match self {
            Number::Int(value) => value.to_string(),
            Number::Big(digits) => digits.clone(),
            Number::Float(_) => unreachable!("a float has no int digits"),
        }
    }

    fn constant(self) -> Constant {
        match self {
            Number::Int(value) => Constant::Int(Int::Small(value)),
            Number::Big(digits) => Constant::Int(Int::Big(digits)),
            Number::Float(value) => Constant::Float(value),
        }
    }
}

fn literal(ast: &Ast, id: NodeId) -> Option<Number> {
    match ast.kind(id) {
        Kind::Constant { value: Constant::Int(Int::Small(value)), .. } => Some(Number::Int(*value)),
        Kind::Constant { value: Constant::Int(Int::Big(digits)), .. } => {
            Some(Number::Big(digits.clone()))
        }
        Kind::Constant { value: Constant::Float(value), .. } if !value.is_nan() => {
            Some(Number::Float(*value))
        }
        Kind::UnaryOp { op: UnaryOperator::USub, operand } => {
            literal(ast, *operand).map(Number::negate)
        }
        _ => None,
    }
}

/// `int(value)` of a literal; `None` where Python raises (`int(inf)`).
fn py_int(value: Number) -> Option<Number> {
    match value {
        Number::Float(number) if number.is_infinite() => None,
        Number::Float(number) => {
            let truncated = number.trunc();
            if truncated.abs() < 9_223_372_036_854_775_808.0 {
                Some(Number::Int(truncated as i64))
            } else {
                // The exact digits of an integral double
                Some(Number::Big(format!("{truncated:.0}")))
            }
        }
        other => Some(other),
    }
}

/// `float(value)` of a literal, correctly rounded; `None` where Python raises (an
/// int beyond the double range).
fn py_float(value: Number) -> Option<Number> {
    match value {
        Number::Int(number) => Some(Number::Float(number as f64)),
        Number::Big(digits) => {
            let number = digits.parse::<f64>().ok()?;
            number.is_finite().then_some(Number::Float(number))
        }
        float => Some(float),
    }
}

/// The order of two ints spelled as their canonical decimal digits.
fn compare_digits(left: &str, right: &str) -> Ordering {
    fn sign(digits: &str) -> (bool, &str) {
        digits.strip_prefix('-').map_or((false, digits), |magnitude| (true, magnitude))
    }
    let ((left_negative, left), (right_negative, right)) = (sign(left), sign(right));
    match (left_negative, right_negative) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        _ => {
            let magnitude = left.len().cmp(&right.len()).then_with(|| left.cmp(right));
            if left_negative { magnitude.reverse() } else { magnitude }
        }
    }
}

/// Python's exact order of an int, spelled as its digits, against a float.
fn compare_int_float(int: &str, float: f64) -> Ordering {
    if float.is_infinite() {
        return if float > 0.0 { Ordering::Less } else { Ordering::Greater };
    }
    let floor = float.floor();
    match compare_digits(int, &Int::from_integral(floor).to_string()) {
        Ordering::Equal if float > floor => Ordering::Less,
        other => other,
    }
}

/// Python's order of two numeric literals, exact across int and float.
fn compare_numbers(left: &Number, right: &Number) -> Result<Ordering, Error> {
    Ok(match (left, right) {
        (Number::Int(a), Number::Int(b)) => a.cmp(b),
        (Number::Float(a), Number::Float(b)) => {
            a.partial_cmp(b).ok_or_else(|| unsupported("a nan literal"))?
        }
        (Number::Float(_), _) => compare_numbers(right, left)?.reverse(),
        (int, Number::Float(float)) => compare_int_float(&int.digits(), *float),
        (left, right) => compare_digits(&left.digits(), &right.digits()),
    })
}

/// `_LiteralFolder`: evaluate what a literal argument decides, post-order.
struct LiteralFolder<'a> {
    /// Placeholder names standing for a proven loop counter argument
    counters: &'a HashSet<String>,
    error: Option<Error>,
}

impl LiteralFolder<'_> {
    fn fold_call(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        let Kind::Call { func, args, keywords } = ast.kind(id) else { unreachable!() };
        if !keywords.is_empty() {
            return id;
        }
        let Kind::Name { id: name, .. } = ast.kind(*func) else { return id };
        let fold: fn(Number) -> Option<Number> = match anchor_of(name) {
            Some("py_int") => py_int,
            Some("py_float") => py_float,
            _ => return id,
        };
        if args.len() != 1 {
            return id;
        }
        let Some(value) = literal(ast, args[0]) else { return id };
        match fold(value) {
            Some(result) => ast.add(Kind::Constant { value: result.constant(), kind: None }),
            None => id,
        }
    }

    fn fold_compare(&mut self, ast: &mut Ast, id: NodeId) -> Result<NodeId, Error> {
        let Kind::Compare { left, ops, comparators } = ast.kind(id) else { unreachable!() };
        if let ([op @ (CmpOperator::Eq | CmpOperator::NotEq)], [right]) =
            (ops.as_slice(), comparators.as_slice())
            && let Kind::Name { id: left_name, .. } = ast.kind(*left)
            && self.counters.contains(left_name)
            && let Kind::Name { id: right_name, .. } = ast.kind(*right)
            && right_name == left_name
        {
            let value = Constant::Bool(*op == CmpOperator::Eq);
            return Ok(ast.add(Kind::Constant { value, kind: None }));
        }
        let mut values = Vec::with_capacity(comparators.len() + 1);
        for operand in std::iter::once(left).chain(comparators.iter()) {
            match literal(ast, *operand) {
                Some(value) => values.push(value),
                None => return Ok(id),
            }
        }
        // A chain is the conjunction of its adjacent pairs
        let mut result = true;
        for (index, op) in ops.iter().enumerate() {
            use std::cmp::Ordering::{Equal, Greater, Less};
            let test: fn(Ordering) -> bool = match op {
                CmpOperator::Eq => |order| order == Equal,
                CmpOperator::NotEq => |order| order != Equal,
                CmpOperator::Lt => |order| order == Less,
                CmpOperator::LtE => |order| order != Greater,
                CmpOperator::Gt => |order| order == Greater,
                CmpOperator::GtE => |order| order != Less,
                _ => return Ok(id),
            };
            if result {
                result = test(compare_numbers(&values[index], &values[index + 1])?);
            }
        }
        Ok(ast.add(Kind::Constant { value: Constant::Bool(result), kind: None }))
    }
}

impl Transformer for LiteralFolder<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        generic_visit(self, ast, id);
        let answer = match ast.kind(id) {
            Kind::Call { .. } => self.fold_call(ast, id),
            Kind::Compare { .. } => self.fold_compare(ast, id).unwrap_or_else(|error| {
                self.error = Some(error);
                id
            }),
            _ => id,
        };
        Visited::Node(answer)
    }
}

/// `_bool_constant`: the value of a `bool` constant node.
fn bool_constant(ast: &Ast, id: NodeId) -> Option<bool> {
    match ast.kind(id) {
        Kind::Constant { value: Constant::Bool(flag), .. } => Some(*flag),
        _ => None,
    }
}

/// `_simplify_test`: simplify a guard test whose operands the literal fold decided.
fn simplify_test(ast: &mut Ast, id: NodeId) -> NodeId {
    match ast.kind(id) {
        &Kind::UnaryOp { op: UnaryOperator::Not, operand } => {
            let operand = simplify_test(ast, operand);
            if let Some(decided) = bool_constant(ast, operand) {
                return ast.add(Kind::Constant { value: Constant::Bool(!decided), kind: None });
            }
            let Kind::UnaryOp { operand: slot, .. } = ast.kind_mut(id) else { unreachable!() };
            *slot = operand;
            id
        }
        Kind::BoolOp { op, values } => {
            let identity = *op == BoolOperator::And;
            let mut kept = Vec::new();
            for value in values.clone() {
                let simplified = simplify_test(ast, value);
                if bool_constant(ast, simplified) == Some(identity) {
                    continue;
                }
                kept.push(simplified);
            }
            if kept.is_empty() {
                return ast.add(Kind::Constant { value: Constant::Bool(identity), kind: None });
            }
            if bool_constant(ast, kept[0]).is_some() || kept.len() == 1 {
                return kept[0];
            }
            let Kind::BoolOp { values: slot, .. } = ast.kind_mut(id) else { unreachable!() };
            *slot = kept;
            id
        }
        _ => id,
    }
}

/// `_fold_guards`: drop the guards a literal argument decides.
fn fold_guards(ast: &mut Ast, mut id: NodeId) -> NodeId {
    while let &Kind::IfExp { test, .. } = ast.kind(id) {
        let test = simplify_test(ast, test);
        let Kind::IfExp { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
        *slot = test;
        let &Kind::IfExp { body, orelse, .. } = ast.kind(id) else { unreachable!() };
        match bool_constant(ast, test) {
            None => {
                let body = fold_guards(ast, body);
                let orelse = fold_guards(ast, orelse);
                let Kind::IfExp { body: body_slot, orelse: orelse_slot, .. } = ast.kind_mut(id)
                else {
                    unreachable!()
                };
                *body_slot = body;
                *orelse_slot = orelse;
                return id;
            }
            Some(true) => id = body,
            Some(false) => id = orelse,
        }
    }
    id
}

// --- argument purity and evaluation order ------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArgClass {
    /// A constant: substitutable anywhere, no side effect, no way to raise
    Const,
    /// A plain name read: can raise `NameError`, so it is ordered
    Name,
    /// A slot read of the Series/Persistent lowering (`__state__[7]`)
    Slot,
    /// Anything else: evaluated exactly once, in source order
    Impure,
}

/// `_is_state_name`: `__state__` or a per-scope `__state·<scope>__`.
fn is_state_name(name: &str) -> bool {
    name == DEFAULT_STATE_PARAM || (name.starts_with("__state·") && name.ends_with("__"))
}

/// `_arg_class`: classify an actual argument by what re-evaluating it would cost.
fn arg_class(ast: &Ast, id: NodeId) -> ArgClass {
    match ast.kind(id) {
        Kind::Constant { .. } => ArgClass::Const,
        Kind::Name { .. } => ArgClass::Name,
        &Kind::Subscript { value, slice, .. }
            if matches!(ast.kind(value), Kind::Name { id, .. } if is_state_name(id))
                && matches!(ast.kind(slice), Kind::Constant { value: Constant::Int(_), .. }) =>
        {
            ArgClass::Slot
        }
        _ => ArgClass::Impure,
    }
}

/// `_always_nodes`: the names an expression reads on every path, in order.
fn always_nodes(ast: &Ast, id: NodeId, out: &mut Vec<NodeId>) {
    match ast.kind(id) {
        Kind::Name { .. } => out.push(id),
        Kind::Constant { .. } => {}
        &Kind::NamedExpr { target, value } => {
            always_nodes(ast, value, out);
            if matches!(ast.kind(target), Kind::Name { .. }) {
                out.push(target);
            }
        }
        &Kind::UnaryOp { operand, .. } => always_nodes(ast, operand, out),
        &Kind::BinOp { left, right, .. } => {
            always_nodes(ast, left, out);
            always_nodes(ast, right, out);
        }
        Kind::BoolOp { values, .. } => always_nodes(ast, values[0], out),
        Kind::Compare { left, comparators, .. } => {
            always_nodes(ast, *left, out);
            always_nodes(ast, comparators[0], out);
        }
        &Kind::IfExp { test, .. } => always_nodes(ast, test, out),
        Kind::Call { func, args, .. } => {
            always_nodes(ast, *func, out);
            for &arg in args {
                if matches!(ast.kind(arg), Kind::Starred { .. }) {
                    return;
                }
                always_nodes(ast, arg, out);
            }
        }
        &Kind::Attribute { value, .. } => always_nodes(ast, value, out),
        &Kind::Subscript { value, slice, .. } => {
            always_nodes(ast, value, out);
            if is_expr(ast, slice) {
                always_nodes(ast, slice, out);
            }
        }
        Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
            for &elt in elts {
                if matches!(ast.kind(elt), Kind::Starred { .. }) {
                    return;
                }
                always_nodes(ast, elt, out);
            }
        }
        _ => {}
    }
}

/// `_replace_node`: swap one node for another, found by identity in field order.
fn replace_node(ast: &mut Ast, root: NodeId, old: NodeId, new: NodeId) -> bool {
    let fields = ast.kind(root).class().fields.len();
    for index in 0..fields {
        let items: Vec<NodeId> = match ast.kind(root).field(index) {
            FieldRef::Node(child) => vec![*child],
            FieldRef::OptNode(child) => child.iter().copied().collect(),
            FieldRef::Nodes(items) => items.clone(),
            FieldRef::OptNodes(items) => items.iter().flatten().copied().collect(),
            _ => continue,
        };
        for item in items {
            if item == old {
                match ast.kind_mut(root).field_mut(index) {
                    FieldMut::Node(slot) => *slot = new,
                    FieldMut::OptNode(slot) => *slot = Some(new),
                    FieldMut::Nodes(slot) => {
                        let position = slot.iter().position(|&child| child == old);
                        slot[position.expect("the node found above")] = new;
                    }
                    FieldMut::OptNodes(slot) => {
                        let position = slot.iter().position(|&child| child == Some(old));
                        slot[position.expect("the node found above")] = Some(new);
                    }
                    _ => unreachable!(),
                }
                return true;
            }
            if replace_node(ast, item, old, new) {
                return true;
            }
        }
    }
    false
}

/// What a placeholder is replaced with, built afresh at every read.
#[derive(Clone, Debug)]
enum Factory {
    /// A tagged copy of the argument itself
    Copy(NodeId),
    /// A read of the temporary the argument was bound to
    Temp(String),
}

/// `_PlaceholderSubstitutor`: replace every read of a placeholder.
struct PlaceholderSubstitutor<'a> {
    factories: &'a HashMap<String, Factory>,
}

impl Transformer for PlaceholderSubstitutor<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if let Kind::Name { id: name, .. } = ast.kind(id) {
            return Visited::Node(match self.factories.get(name) {
                Some(Factory::Copy(argument)) => arg_copy(ast, *argument),
                Some(Factory::Temp(temp)) => {
                    let temp = temp.clone();
                    load(ast, &temp)
                }
                None => id,
            });
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }
}

/// `_probe`: `<value> is <name>`, an always-true test pinning when the value is
/// evaluated, marked exact.
fn probe(ast: &mut Ast, value: NodeId, name: &str) -> NodeId {
    let comparator = load(ast, name);
    let probe = ast.add(Kind::Compare {
        left: value,
        ops: vec![CmpOperator::Is],
        comparators: vec![comparator],
    });
    ast.set_extra(probe, "pine_exact", Value::Bool(true));
    probe
}

/// A walrus binding `value` to `name`.
fn bind(ast: &mut Ast, name: &str, value: NodeId) -> NodeId {
    let target = ast.add(Kind::Name { id: name.to_string(), ctx: ExprContext::Store });
    ast.add(Kind::NamedExpr { target, value })
}

// --- the pass ----------------------------------------------------------------

/// `_NameIndex`: every name the module binds anywhere, and its import map from the
/// direct import statements of the module body.
#[derive(Default)]
struct NameIndex {
    bound: HashSet<String>,
    import_map: HashMap<String, (String, Option<String>)>,
}

impl NameIndex {
    fn build(ast: &Ast, module: NodeId) -> NameIndex {
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let toplevel: HashSet<NodeId> = body
            .iter()
            .copied()
            .filter(|&stmt| matches!(ast.kind(stmt), Kind::Import { .. } | Kind::ImportFrom { .. }))
            .collect();
        let mut index = NameIndex::default();
        index.visit(ast, module, &toplevel);
        index
    }

    fn visit(&mut self, ast: &Ast, id: NodeId, toplevel: &HashSet<NodeId>) {
        match ast.kind(id) {
            Kind::Name { id: name, ctx: ExprContext::Store | ExprContext::Del } => {
                self.bound.insert(name.clone());
            }
            Kind::FunctionDef { name, args, .. } | Kind::AsyncFunctionDef { name, args, .. } => {
                self.bound.insert(name.clone());
                if let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
                    ast.kind(*args)
                {
                    for &arg in
                        args.iter().chain(posonlyargs).chain(kwonlyargs).chain(vararg).chain(kwarg)
                    {
                        if let Kind::Arg { arg, .. } = ast.kind(arg) {
                            self.bound.insert(arg.clone());
                        }
                    }
                }
            }
            Kind::ClassDef { name, .. } => {
                self.bound.insert(name.clone());
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                self.bound.insert(name.clone());
            }
            Kind::Import { names } => {
                let trusted = toplevel.contains(&id);
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                    let first = name.split('.').next().unwrap_or_default();
                    let bound = asname.as_deref().unwrap_or(first);
                    if !trusted {
                        self.bound.insert(bound.to_string());
                        continue;
                    }
                    let module = if asname.is_some() { name.as_str() } else { first };
                    self.import_map.insert(bound.to_string(), (module.to_string(), None));
                }
                return;
            }
            Kind::ImportFrom { module, names, level } => {
                let module = module.as_deref().filter(|module| !module.is_empty());
                let trusted = toplevel.contains(&id) && level.unwrap_or(0) == 0;
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                    let bound = asname.as_deref().unwrap_or(name);
                    match module {
                        Some(module) if trusted => {
                            self.import_map.insert(
                                bound.to_string(),
                                (module.to_string(), Some(name.clone())),
                            );
                        }
                        _ => {
                            self.bound.insert(bound.to_string());
                        }
                    }
                }
                return;
            }
            _ => {}
        }
        for child in children(ast, id) {
            self.visit(ast, child, toplevel);
        }
    }
}

/// `_Rebinds`: whether a subtree binds a name.
fn rebinds(ast: &Ast, root: NodeId, target: &str) -> bool {
    walk(ast, root).into_iter().any(|id| match ast.kind(id) {
        Kind::Name { id: name, ctx } => name == target && *ctx != ExprContext::Load,
        Kind::FunctionDef { name, .. }
        | Kind::AsyncFunctionDef { name, .. }
        | Kind::ClassDef { name, .. } => name == target,
        Kind::ExceptHandler { name, .. }
        | Kind::MatchAs { name, .. }
        | Kind::MatchStar { name } => name.as_deref() == Some(target),
        Kind::Import { names } | Kind::ImportFrom { names, .. } => names.iter().any(|&alias| {
            matches!(ast.kind(alias), Kind::Alias { name, asname }
                if asname.as_deref().unwrap_or(name.split('.').next().unwrap_or_default()) == target)
        }),
        _ => false,
    })
}

struct CallInline<'t> {
    templates: &'t InlineTemplates,
    index: NameIndex,
    used: BTreeSet<String>,
    counter: usize,
    func_depth: usize,
    /// The enclosing function definitions, innermost last
    functions: Vec<NodeId>,
    /// The loop counters proven never na at the current position
    counters: HashSet<String>,
    error: Option<Error>,
}

impl<'t> CallInline<'t> {
    /// `_resolve_path`: the path a callee spells once the import map resolved its base.
    fn resolve_path(&self, ast: &Ast, func: NodeId) -> Option<String> {
        let path = func_path(ast, func)?;
        let mut parts = path.split('.');
        let base = parts.next()?;
        if self.index.bound.contains(base) {
            return None;
        }
        let (module, attr) = self.index.import_map.get(base)?;
        let mut resolved = vec![module.as_str()];
        resolved.extend(attr.as_deref());
        resolved.extend(parts);
        Some(resolved.join("."))
    }

    fn inline(&mut self, ast: &mut Ast, node: NodeId) -> Result<Option<NodeId>, Error> {
        let templates: &'t InlineTemplates = self.templates;
        let Kind::Call { func, args, keywords } = ast.kind(node) else { unreachable!() };
        let (func, args) = (*func, args.clone());
        if !keywords.is_empty()
            || args.iter().any(|&arg| matches!(ast.kind(arg), Kind::Starred { .. }))
        {
            return Ok(None);
        }
        let Some(key) = self.resolve_path(ast, func).and_then(|path| templates.callees.get(&path))
        else {
            return Ok(None);
        };
        let function = templates
            .functions
            .get(key)
            .ok_or_else(|| unsupported(format!("no inline template for {key}")))?;
        match function {
            InlineFunction::Derive { params, .. } if args.len() != params.len() => return Ok(None),
            InlineFunction::Varargs { .. } if !templates.varargs_arities.contains(&args.len()) => {
                return Ok(None);
            }
            _ => {}
        }

        let classes: Vec<ArgClass> = args.iter().map(|&arg| arg_class(ast, arg)).collect();
        // A constant goes in verbatim, so the literal fold can decide the guards it
        // controls; everything else renders as a placeholder
        let mut placeholders: Vec<Option<String>> = Vec::with_capacity(args.len());
        let mut render_args = Vec::with_capacity(args.len());
        for (index, &arg) in args.iter().enumerate() {
            if classes[index] == ArgClass::Const {
                placeholders.push(None);
                render_args.push(arg_copy(ast, arg));
            } else {
                let name = placeholder_name(index);
                render_args.push(load(ast, &name));
                placeholders.push(Some(name));
            }
        }
        let counters: HashSet<String> = args
            .iter()
            .enumerate()
            .filter(|&(index, &arg)| {
                classes[index] == ArgClass::Name
                    && matches!(ast.kind(arg), Kind::Name { id, .. } if self.counters.contains(id))
            })
            .filter_map(|(index, _)| placeholders[index].clone())
            .collect();

        let rendered = match function {
            InlineFunction::Derive { body, .. } => build(ast, body, &render_args)?,
            InlineFunction::Varargs { builtin, na_anchor, builtins_anchor } => {
                render_varargs(ast, builtin, na_anchor, builtins_anchor, &render_args)
            }
        };
        let mut folder = LiteralFolder { counters: &counters, error: None };
        let folded = folder.visit(ast, rendered);
        if let Some(error) = folder.error {
            return Err(error);
        }
        let Visited::Node(folded) = folded else { unreachable!() };
        let mut expression = fold_guards(ast, folded);
        // Marked before the arguments go in: the user's subtrees stay untouched
        mark_exact(ast, expression)?;

        // The anchors are collected from the folded body; the arguments going in
        // below bring none of their own
        let mut reads = vec![0usize; args.len()];
        for child in walk(ast, expression) {
            let Kind::Name { id: name, .. } = ast.kind(child) else { continue };
            if let Some(index) = placeholders.iter().position(|p| p.as_deref() == Some(name)) {
                reads[index] += 1;
            } else if let Some(anchor) = anchor_of(name) {
                self.used.insert(anchor.to_string());
            }
        }
        let mut always = Vec::new();
        always_nodes(ast, expression, &mut always);
        // The first always-read of each placeholder, in the order they are found,
        // which is the order of their first reads
        let mut first_always: Vec<(usize, NodeId)> = Vec::new();
        for &name_node in &always {
            let Kind::Name { id: name, .. } = ast.kind(name_node) else { continue };
            if let Some(index) = placeholders.iter().position(|p| p.as_deref() == Some(name))
                && !first_always.iter().any(|&(seen, _)| seen == index)
            {
                first_always.push((index, name_node));
            }
        }
        let anchor_of_index =
            |index: usize| first_always.iter().find(|&&(seen, _)| seen == index).map(|&(_, n)| n);

        // A name read can raise, so it is ordered against the impure arguments
        let ordered: Vec<usize> = (0..args.len())
            .filter(|&index| matches!(classes[index], ArgClass::Name | ArgClass::Impure))
            .collect();
        let impure = classes.contains(&ArgClass::Impure);
        let natural: Vec<usize> = first_always
            .iter()
            .map(|&(index, _)| index)
            .filter(|index| ordered.contains(index))
            .collect();
        let force = impure && natural != ordered;

        let mut factories: HashMap<String, Factory> = HashMap::new();
        let mut probes = Vec::new();
        for (index, &kind) in classes.iter().enumerate() {
            if kind == ArgClass::Const {
                continue;
            }
            let name = placeholders[index].clone().expect("a placeholder per non-constant");
            let argument = args[index];
            let anchor = anchor_of_index(index);
            if force && ordered.contains(&index) {
                if kind == ArgClass::Name {
                    // A name needs no temporary: the probe only pins where it is read
                    let Kind::Name { id: read, .. } = ast.kind(argument) else { unreachable!() };
                    let read = read.clone();
                    let copy = arg_copy(ast, argument);
                    probes.push(probe(ast, copy, &read));
                    factories.insert(name, Factory::Copy(argument));
                    continue;
                }
                self.counter += 1;
                let temp = temp_name(self.counter);
                let copy = arg_copy(ast, argument);
                let binding = bind(ast, &temp, copy);
                probes.push(probe(ast, binding, &temp));
                factories.insert(name, Factory::Temp(temp));
                continue;
            }
            let direct = kind == ArgClass::Name
                || (kind == ArgClass::Slot && (reads[index] <= 1 || anchor.is_none()))
                || (kind == ArgClass::Impure && reads[index] <= 1 && anchor.is_some());
            if direct {
                factories.insert(name, Factory::Copy(argument));
                continue;
            }
            // Read more than once: bound once at its first read, which is on every path
            let anchor =
                anchor.ok_or_else(|| unsupported("a binding without an always-read anchor"))?;
            self.counter += 1;
            let temp = temp_name(self.counter);
            let copy = arg_copy(ast, argument);
            let binding = bind(ast, &temp, copy);
            replace_node(ast, expression, anchor, binding);
            factories.insert(name, Factory::Temp(temp));
        }
        if !factories.is_empty() {
            // Python discards what the visit answers for the root itself
            PlaceholderSubstitutor { factories: &factories }.visit(ast, expression);
        }
        for probe in probes.into_iter().rev() {
            expression =
                ast.add(Kind::BoolOp { op: BoolOperator::And, values: vec![probe, expression] });
        }

        stamp_emission(ast, expression, node)?;

        // `stamp_lowering` keeps a string stamp it meets and types the rest; the Python
        // pass would carry any other stamp along as it is
        for child in walk(ast, expression) {
            match ast.extra(child, TY_ATTR) {
                None | Some(Value::None | Value::Str(_)) => {}
                Some(other) => return Err(unsupported(format!("a {TY_ATTR} stamp {other:?}"))),
            }
        }
        let ty = ty_stamp(ast, node)?.to_string();
        let expression = stamp_lowering(ast, expression, &ty);
        copy_location(ast, expression, node);
        Ok(Some(expression))
    }

    /// `_visit_definitions`: visit the function definitions among the statements.
    fn visit_definitions(&mut self, ast: &mut Ast, id: NodeId) {
        for child in child_statements(ast, id) {
            match ast.kind(child) {
                Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                    self.visit(ast, child);
                }
                Kind::ClassDef { .. } => {}
                _ => self.visit_definitions(ast, child),
            }
        }
    }

    fn visit_module(&mut self, ast: &mut Ast, module: NodeId) {
        self.index = NameIndex::build(ast, module);
        self.visit_definitions(ast, module);
        if self.used.is_empty() || self.error.is_some() {
            return;
        }
        let names = self
            .used
            .iter()
            .map(|name| {
                ast.add(Kind::Alias { name: name.clone(), asname: Some(support_alias(name)) })
            })
            .collect();
        let import = ast.add(Kind::ImportFrom {
            module: Some(SUPPORT_MODULE.to_string()),
            names,
            level: Some(0),
        });
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let mut position = 0;
        if let Some(&first) = body.first()
            && let Kind::Expr { value } = ast.kind(first)
            && matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
        {
            position = 1;
        }
        for (index, &stmt) in body.iter().enumerate().skip(position) {
            match ast.kind(stmt) {
                Kind::Import { .. } | Kind::ImportFrom { .. } => position = index + 1,
                Kind::Expr { .. } => {}
                _ => break,
            }
        }
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        body.insert(position, import);
    }

    /// `_loop_counter`: the counter of a `pine_range` loop that is never na in its body.
    fn loop_counter(&self, ast: &Ast, node: NodeId) -> Option<String> {
        let &innermost = self.functions.last()?;
        let Kind::For { target, iter, body, .. } = ast.kind(node) else { unreachable!() };
        let Kind::Name { id: name, .. } = ast.kind(*target) else { return None };
        let Kind::Call { func, .. } = ast.kind(*iter) else { return None };
        let path = self.resolve_path(ast, *func)?;
        if !self.templates.loop_ranges.contains(&path) {
            return None;
        }
        if body.iter().any(|&stmt| rebinds(ast, stmt, name)) {
            return None;
        }
        let declared = walk_statements(ast, innermost).into_iter().any(|stmt| {
            matches!(ast.kind(stmt), Kind::Global { names } | Kind::Nonlocal { names }
                if names.iter().any(|declared| declared == name))
        });
        (!declared).then(|| name.clone())
    }

    fn visit_for(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        let &Kind::For { target, iter, .. } = ast.kind(node) else { unreachable!() };
        let target = self.visit_node(ast, target)?;
        let iter = self.visit_node(ast, iter)?;
        let Kind::For { target: target_slot, iter: iter_slot, .. } = ast.kind_mut(node) else {
            unreachable!()
        };
        *target_slot = target;
        *iter_slot = iter;
        let outer = self.counters.clone();
        if let Some(counter) = self.loop_counter(ast, node) {
            self.counters.insert(counter);
        }
        let body = self.visit_statements(ast, node, |kind| match kind {
            Kind::For { body, .. } => body,
            _ => unreachable!(),
        })?;
        let Kind::For { body: slot, .. } = ast.kind_mut(node) else { unreachable!() };
        *slot = body;
        self.counters = outer;
        let orelse = self.visit_statements(ast, node, |kind| match kind {
            Kind::For { orelse, .. } => orelse,
            _ => unreachable!(),
        })?;
        let Kind::For { orelse: slot, .. } = ast.kind_mut(node) else { unreachable!() };
        *slot = orelse;
        Ok(())
    }

    fn visit_function_def(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        let (Kind::FunctionDef { args, decorator_list, .. }
        | Kind::AsyncFunctionDef { args, decorator_list, .. }) = ast.kind(node)
        else {
            unreachable!()
        };
        // Decorators and defaults are evaluated in the enclosing scope; Python discards
        // what their visits answer
        let mut outer_nodes = decorator_list.clone();
        if let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(*args) {
            outer_nodes.extend(defaults);
            outer_nodes.extend(kw_defaults.iter().flatten());
        }
        for id in outer_nodes {
            self.visit(ast, id);
        }
        let outer = std::mem::take(&mut self.counters);
        self.functions.push(node);
        self.func_depth += 1;
        let body = self.visit_statements(ast, node, |kind| match kind {
            Kind::FunctionDef { body, .. } | Kind::AsyncFunctionDef { body, .. } => body,
            _ => unreachable!(),
        });
        self.func_depth -= 1;
        self.functions.pop();
        self.counters = outer;
        let body = body?;
        let (Kind::FunctionDef { body: slot, .. } | Kind::AsyncFunctionDef { body: slot, .. }) =
            ast.kind_mut(node)
        else {
            unreachable!()
        };
        *slot = body;
        Ok(())
    }

    /// Visit a statement list (`[self.visit(stmt) for stmt in ...]`).
    fn visit_statements(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        field: fn(&Kind) -> &Vec<NodeId>,
    ) -> Result<Vec<NodeId>, Error> {
        let statements = field(ast.kind(node)).clone();
        statements.into_iter().map(|stmt| self.visit_node(ast, stmt)).collect()
    }

    /// A visit whose answer must be a single node.
    fn visit_node(&mut self, ast: &mut Ast, id: NodeId) -> Result<NodeId, Error> {
        match self.visit(ast, id) {
            Visited::Node(new) => Ok(new),
            other => Err(unsupported(format!("{other:?} for a {}", ast.name(id)))),
        }
    }
}

impl Transformer for CallInline<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        let answer = match ast.kind(id) {
            Kind::Module { .. } => {
                self.visit_module(ast, id);
                Ok(id)
            }
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                self.visit_function_def(ast, id).map(|()| id)
            }
            Kind::For { .. } => self.visit_for(ast, id).map(|()| id),
            // A temporary would bind in the wrong scope: not entered
            Kind::ClassDef { .. }
            | Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. } => Ok(id),
            Kind::Call { .. } => {
                generic_visit(self, ast, id);
                if self.func_depth == 0 || self.error.is_some() {
                    Ok(id)
                } else {
                    self.inline(ast, id).map(|inlined| inlined.unwrap_or(id))
                }
            }
            _ => {
                generic_visit(self, ast, id);
                Ok(id)
            }
        };
        match answer {
            Ok(node) => Visited::Node(node),
            Err(error) => {
                self.error.get_or_insert(error);
                Visited::Node(id)
            }
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(unsupported("CallInline on a non-module"));
    }
    let mut pass = CallInline {
        templates: data::inline_templates(),
        index: NameIndex::default(),
        used: BTreeSet::new(),
        counter: 0,
        func_depth: 0,
        functions: Vec::new(),
        counters: HashSet::new(),
        error: None,
    };
    pass.visit(ast, module);
    match pass.error {
        Some(error) => Err(error),
        None => Ok(module),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(digits: &str) -> Number {
        Number::Big(digits.into())
    }

    #[test]
    fn literals_compare_exactly_beyond_64_bits() {
        let order = |left: Number, right: Number| compare_numbers(&left, &right).unwrap();
        assert_eq!(order(big("99999999999999999999"), Number::Float(1e20)), Ordering::Less);
        assert_eq!(order(big("100000000000000000000"), Number::Float(1e20)), Ordering::Equal);
        assert_eq!(order(big("100000000000000000001"), Number::Float(1e20)), Ordering::Greater);
        assert_eq!(order(Number::Float(1e20), big("100000000000000000001")), Ordering::Less);
        assert_eq!(order(big("-99999999999999999999"), Number::Int(i64::MIN)), Ordering::Less);
        assert_eq!(order(big("9223372036854775808"), Number::Int(i64::MAX)), Ordering::Greater);
        assert_eq!(
            order(big("-1000000000000000000000"), big("-100000000000000000000")),
            Ordering::Less
        );
        assert_eq!(
            order(big("1000000000000000000000"), Number::Float(f64::INFINITY)),
            Ordering::Less
        );
        assert_eq!(order(Number::Int(3), Number::Float(2.5)), Ordering::Greater);
        assert_eq!(order(Number::Int(-3), Number::Float(-2.5)), Ordering::Less);
        assert_eq!(order(Number::Int(0), Number::Float(-0.0)), Ordering::Equal);
    }
}
