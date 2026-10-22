//! Port of `transformers/safe_division_transformer.py`, a rule of the `Expression`
//! phase: a division gets Pine's semantics through `safe_convert.safe_div` (inside a
//! function body, for numeric-typed operands, written out as a fast-path expression
//! that falls back to the call), user code's `%` becomes `safe_convert.safe_mod`, and
//! an augmented division or remainder is written out as the plain assignment of the
//! same call.

use pyne_ast::walk::{Visited, copy_location, walk};
use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, ExprContext, Kind, NodeId, Operator, Value,
};

use crate::phase::{Rule, run_rules, single};
use crate::pine_type_rules::{
    BOOL, FLOAT, INT, STR, TYPELESS, get_ty, set_ty, stamp_lowering, ty_stamp,
};
use crate::steps::safe_convert::{
    binds_by_import, import_position, safe_convert_attr, safe_convert_import,
};
use crate::{Context, Error};

/// Operand types whose runtime values are native numbers, bools or na
const INLINE_TYPES: &[&str] = &[INT, FLOAT, BOOL, TYPELESS];

fn temp_name(counter: usize) -> String {
    format!("__div{counter}·__")
}

/// Whether a node is an int or float literal (a bool is not one).
fn is_numeric_constant(ast: &Ast, id: NodeId) -> bool {
    matches!(ast.kind(id), Kind::Constant { value: Constant::Int(_) | Constant::Float(_), .. })
}

/// Python's truth value of a numeric literal.
fn numeric_truthy(value: &Constant) -> bool {
    match value {
        Constant::Int(pyne_ast::Int::Small(number)) => *number != 0,
        Constant::Int(pyne_ast::Int::Big(_)) => true,
        Constant::Float(number) => *number != 0.0,
        _ => unreachable!("not a numeric literal"),
    }
}

fn load(ast: &mut Ast, name: &str) -> NodeId {
    ast.add(Kind::Name { id: name.into(), ctx: ExprContext::Load })
}

fn store(ast: &mut Ast, name: &str) -> NodeId {
    ast.add(Kind::Name { id: name.into(), ctx: ExprContext::Store })
}

/// A fresh read of a name or literal operand (`_reread`).
fn reread(ast: &mut Ast, id: NodeId) -> NodeId {
    match ast.kind(id) {
        Kind::Name { id: name, .. } => {
            let name = name.clone();
            load(ast, &name)
        }
        Kind::Constant { value, .. } => {
            let value = value.clone();
            ast.add(Kind::Constant { value, kind: None })
        }
        _ => unreachable!("reread of a {}", ast.name(id)),
    }
}

/// How the fallback reads an operand again.
enum Reference {
    /// A fresh read of the operand itself (a name or a literal)
    Reread(NodeId),
    /// The temporary the operand was bound to
    Temp(String),
}

impl Reference {
    fn node(&self, ast: &mut Ast) -> NodeId {
        match self {
            Reference::Reread(id) => reread(ast, *id),
            Reference::Temp(name) => load(ast, name),
        }
    }
}

fn is_name_or_constant(ast: &Ast, id: NodeId) -> bool {
    matches!(ast.kind(id), Kind::Name { .. } | Kind::Constant { .. })
}

pub struct SafeDivision {
    pine_mod: bool,
    has_division_operations: bool,
    counter: usize,
    func_depth: usize,
    blocked: usize,
    /// Per enclosing function body, the `blocked` count of the scope around it
    outer_blocked: Vec<usize>,
}

impl SafeDivision {
    pub fn new(ctx: &Context<'_>) -> SafeDivision {
        SafeDivision {
            pine_mod: ctx.user_code,
            has_division_operations: false,
            counter: 0,
            func_depth: 0,
            blocked: 0,
            outer_blocked: Vec::new(),
        }
    }

    fn temp(&mut self) -> String {
        self.counter += 1;
        temp_name(self.counter)
    }

    /// A walrus binding `value` to `name`.
    fn bind(ast: &mut Ast, name: &str, value: NodeId) -> NodeId {
        let target = store(ast, name);
        ast.add(Kind::NamedExpr { target, value })
    }

    /// Write one division out as the fast-path expression, `None` when it keeps its call.
    fn inline(&mut self, ast: &mut Ast, node: NodeId) -> Result<Option<NodeId>, Error> {
        let &Kind::BinOp { left, right, .. } = ast.kind(node) else { unreachable!() };
        if !INLINE_TYPES.contains(&get_ty(ast, left)) || !INLINE_TYPES.contains(&get_ty(ast, right))
        {
            return Ok(None);
        }
        let right_const = is_numeric_constant(ast, right);
        if matches!(ast.kind(right), Kind::Constant { .. }) && !right_const {
            return Ok(None);
        }
        if right_const {
            let Kind::Constant { value, .. } = ast.kind(right) else { unreachable!() };
            // A zero, nan or inf literal divisor is not worth a fast path
            let skip = match value {
                Constant::Float(number) => *number == 0.0 || !number.is_finite(),
                other => !numeric_truthy(other),
            };
            if skip {
                return Ok(None);
            }
        }

        // The left operand is re-read by the fallback, AFTER the right one ran: only a
        // literal, or a name no right operand can rebind, may be re-read
        let left_rereads = matches!(ast.kind(left), Kind::Constant { .. })
            || (matches!(ast.kind(left), Kind::Name { .. }) && is_name_or_constant(ast, right));
        let (left_first, left_ref) = if left_rereads {
            (left, Reference::Reread(left))
        } else {
            let temp = self.temp();
            (SafeDivision::bind(ast, &temp, left), Reference::Temp(temp))
        };
        let (right_first, right_ref) = if is_name_or_constant(ast, right) {
            (right, Reference::Reread(right))
        } else {
            let temp = self.temp();
            (SafeDivision::bind(ast, &temp, right), Reference::Temp(temp))
        };

        let divisor = if right_const {
            right_first
        } else {
            let zero = safe_convert_attr(ast, "zero_divisor");
            ast.add(Kind::BoolOp { op: BoolOperator::Or, values: vec![right_first, zero] })
        };
        // Every node that carries the quotient carries the division's type, the
        // fallback call included
        let ty = ty_stamp(ast, node)?.to_string();
        let result = self.temp();
        let quotient = ast.add(Kind::BinOp { left: left_first, op: Operator::Div, right: divisor });
        let quotient = set_ty(ast, quotient, &ty);
        let binding = SafeDivision::bind(ast, &result, quotient);
        let binding = set_ty(ast, binding, &ty);
        let reread_result = load(ast, &result);
        let reread_result = set_ty(ast, reread_result, &ty);
        let test = ast.add(Kind::Compare {
            left: binding,
            ops: vec![CmpOperator::Eq],
            comparators: vec![reread_result],
        });
        // A raw self-equality test, not a Pine comparison: the tolerance rewrite must
        // leave it alone
        ast.set_extra(test, "pine_exact", Value::Bool(true));
        let body = load(ast, &result);
        let body = set_ty(ast, body, &ty);
        let func = safe_convert_attr(ast, "safe_div");
        let args = vec![left_ref.node(ast), right_ref.node(ast)];
        let orelse = ast.add(Kind::Call { func, args, keywords: Vec::new() });
        let orelse = set_ty(ast, orelse, &ty);
        Ok(Some(ast.add(Kind::IfExp { test, body, orelse })))
    }

    /// Whether a `%` is Pine's remainder rather than Python's string formatting.
    fn is_pine_mod(&self, ast: &Ast, op: Operator, left: NodeId) -> bool {
        self.pine_mod
            && op == Operator::Mod
            && get_ty(ast, left) != STR
            && !matches!(ast.kind(left), Kind::Constant { value: Constant::Str(_), .. })
            && !matches!(ast.kind(left), Kind::JoinedStr { .. })
    }

    /// Evaluate one part of an augmented target once for its read and its store:
    /// the expressions for the read and for the store.
    fn bind_once(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        temps: &mut Vec<String>,
    ) -> (NodeId, NodeId) {
        if matches!(ast.kind(node), Kind::Constant { .. }) {
            return (node, reread(ast, node));
        }
        let temp = self.temp();
        let binding = SafeDivision::bind(ast, &temp, node);
        let read = load(ast, &temp);
        temps.push(temp);
        (binding, read)
    }

    fn leave_aug_assign(&mut self, ast: &mut Ast, node: NodeId) -> Result<Visited, Error> {
        let &Kind::AugAssign { target: old_target, op, value } = ast.kind(node) else {
            unreachable!()
        };
        let helper = if op == Operator::Div {
            "safe_div"
        } else if self.is_pine_mod(ast, op, old_target) {
            "safe_mod"
        } else {
            return Ok(Visited::Node(node));
        };
        let mut temps = Vec::new();
        let (current, target) = match ast.kind(old_target) {
            Kind::Name { id, .. } => {
                let id = id.clone();
                (load(ast, &id), old_target)
            }
            Kind::Attribute { value: receiver, attr, .. } => {
                let (receiver, attr) = (*receiver, attr.clone());
                let (read, stored) = self.bind_once(ast, receiver, &mut temps);
                let current = ast.add(Kind::Attribute {
                    value: read,
                    attr: attr.clone(),
                    ctx: ExprContext::Load,
                });
                let target =
                    ast.add(Kind::Attribute { value: stored, attr, ctx: ExprContext::Store });
                (current, target)
            }
            &Kind::Subscript { value: receiver, slice, .. }
                if !walk(ast, slice)
                    .into_iter()
                    .any(|part| matches!(ast.kind(part), Kind::Slice { .. })) =>
            {
                let (read, stored) = self.bind_once(ast, receiver, &mut temps);
                let (index, stored_index) = self.bind_once(ast, slice, &mut temps);
                let current =
                    ast.add(Kind::Subscript { value: read, slice: index, ctx: ExprContext::Load });
                let target = ast.add(Kind::Subscript {
                    value: stored,
                    slice: stored_index,
                    ctx: ExprContext::Store,
                });
                (current, target)
            }
            _ => return Ok(Visited::Node(node)),
        };
        self.has_division_operations = true;
        let ty = ty_stamp(ast, old_target)?.to_string();
        let current = set_ty(ast, current, &ty);
        let func = safe_convert_attr(ast, helper);
        let call = ast.add(Kind::Call { func, args: vec![current, value], keywords: Vec::new() });
        let call = stamp_lowering(ast, call, &ty);
        let assign =
            ast.add(Kind::Assign { targets: vec![target], value: call, type_comment: None });
        copy_location(ast, assign, node);
        if temps.is_empty() || (self.func_depth > 0 && self.blocked == 0) {
            return Ok(Visited::Node(assign));
        }
        let targets = temps
            .iter()
            .map(|temp| ast.add(Kind::Name { id: temp.clone(), ctx: ExprContext::Del }))
            .collect();
        let delete = ast.add(Kind::Delete { targets });
        copy_location(ast, delete, node);
        Ok(Visited::Many(vec![assign, delete]))
    }

    fn leave_bin_op(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Error> {
        let &Kind::BinOp { left, op, right } = ast.kind(node) else { unreachable!() };
        if self.is_pine_mod(ast, op, left) {
            self.has_division_operations = true;
            let ty = ty_stamp(ast, node)?.to_string();
            let func = safe_convert_attr(ast, "safe_mod");
            let call = ast.add(Kind::Call { func, args: vec![left, right], keywords: Vec::new() });
            let call = stamp_lowering(ast, call, &ty);
            copy_location(ast, call, node);
            return Ok(call);
        }
        if op != Operator::Div {
            return Ok(node);
        }
        // A literal division by a nonzero literal cannot raise and stays as is
        if matches!(ast.kind(left), Kind::Constant { .. })
            && let Kind::Constant { value: value @ (Constant::Int(_) | Constant::Float(_)), .. } =
                ast.kind(right)
            && numeric_truthy(value)
        {
            return Ok(node);
        }
        self.has_division_operations = true;
        let mut expression = None;
        if self.func_depth > 0 && self.blocked == 0 {
            expression = self.inline(ast, node)?;
        }
        let expression = match expression {
            Some(expression) => expression,
            None => {
                let func = safe_convert_attr(ast, "safe_div");
                ast.add(Kind::Call { func, args: vec![left, right], keywords: Vec::new() })
            }
        };
        // The replacement takes over the division's Pine type
        let ty = ty_stamp(ast, node)?.to_string();
        let expression = stamp_lowering(ast, expression, &ty);
        copy_location(ast, expression, node);
        Ok(expression)
    }

    fn leave_module(&mut self, ast: &mut Ast, node: NodeId) {
        if !self.has_division_operations {
            return;
        }
        let Kind::Module { body, .. } = ast.kind(node) else { return };
        // Only an import that binds the module itself to `safe_convert` provides the
        // name the emitted calls read
        for &stmt in body {
            let bound = match ast.kind(stmt) {
                Kind::ImportFrom { module, names, level } => {
                    let expected = if *level == Some(0) { "pynecore.core" } else { "core" };
                    module.as_deref() == Some(expected)
                        && names.iter().any(|&alias| {
                            matches!(ast.kind(alias), Kind::Alias { name, asname }
                                if name == "safe_convert"
                                    && matches!(asname.as_deref(), None | Some("safe_convert")))
                        })
                }
                Kind::Import { names } => binds_by_import(ast, names),
                _ => continue,
            };
            if bound {
                return;
            }
        }
        let position = import_position(ast, body);
        let import = safe_convert_import(ast);
        let Kind::Module { body, .. } = ast.kind_mut(node) else { unreachable!() };
        body.insert(position, import);
    }
}

impl Rule for SafeDivision {
    fn enter(&mut self, _ast: &mut Ast, class: &'static str, _node: NodeId) -> Result<(), Error> {
        // A temporary bound in a class body would become a class attribute, a walrus
        // in a lambda binds in the lambda's own scope, and one in a comprehension is
        // illegal in its iterable and binds to the containing scope elsewhere
        if matches!(
            class,
            "ClassDef" | "Lambda" | "ListComp" | "SetComp" | "DictComp" | "GeneratorExp"
        ) {
            self.blocked += 1;
        }
        Ok(())
    }

    fn leave(
        &mut self,
        ast: &mut Ast,
        class: &'static str,
        node: Visited,
    ) -> Result<Visited, Error> {
        match class {
            "ClassDef" | "Lambda" | "ListComp" | "SetComp" | "DictComp" | "GeneratorExp" => {
                self.blocked -= 1;
                Ok(node)
            }
            "AugAssign" => self.leave_aug_assign(ast, single(&node, class)?),
            "BinOp" => Ok(Visited::Node(self.leave_bin_op(ast, single(&node, class)?)?)),
            "Module" => {
                self.leave_module(ast, single(&node, class)?);
                Ok(node)
            }
            _ => Ok(node),
        }
    }

    fn enter_function_body(&mut self, _node: NodeId) {
        // Decorators and defaults are evaluated in the enclosing scope, so only the
        // body gets the function's own depth; the body is a scope of its own even
        // inside a class body
        self.outer_blocked.push(self.blocked);
        self.blocked = 0;
        self.func_depth += 1;
    }

    fn leave_function_body(&mut self, _node: NodeId) {
        self.func_depth -= 1;
        self.blocked = self.outer_blocked.pop().expect("a function body was entered");
    }
}

fn make(ctx: &Context<'_>) -> Box<dyn Rule> {
    Box::new(SafeDivision::new(ctx))
}

/// The rule on its own (sequential mode).
pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    run_rules(ast, module, vec![make(ctx)])
}

pub const RULE: crate::phase::RuleStep =
    crate::phase::RuleStep { name: "SafeDivision", make, user_code_only: false };
