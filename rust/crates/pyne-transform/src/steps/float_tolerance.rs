//! Port of `transformers/float_tolerance.py`, a rule of the `Expression` phase (user
//! code only): the comparison operators get TradingView's tolerant float semantics,
//! each rewritten into an arithmetic form over the difference with the bound on the
//! left, `==` / `!=` guarded by a runtime float test, and, in a module that keeps
//! Pine's three-state bool, a tail that answers na for an na operand.

use pyne_ast::walk::Visited;
use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, ExprContext, Kind, NodeId, Operator, UnaryOperator,
    Value,
};

use crate::phase::{Rule, run_rules, single};
use crate::pine_type_rules::{BOOL, stamp_lowering};
use crate::steps::pine_truthiness::truthy;
use crate::{Context, Error};

/// Pine's comparison tolerance, `pynecore.core.pine_compare.EPSILON`.
const EPSILON: f64 = 1e-10;

/// The alias the interned bool na is bound to in a script that keeps the three-state bool
const NA_BOOL_NAME: &str = "__pyne_cmp_na·__";

/// The tolerant ordering form of an operator: the bound and its comparison against
/// the difference.
fn ordering_form(op: CmpOperator) -> Option<(f64, CmpOperator)> {
    match op {
        CmpOperator::Lt => Some((-EPSILON, CmpOperator::Gt)),
        CmpOperator::Gt => Some((EPSILON, CmpOperator::Lt)),
        CmpOperator::LtE => Some((EPSILON, CmpOperator::GtE)),
        CmpOperator::GtE => Some((-EPSILON, CmpOperator::LtE)),
        _ => None,
    }
}

/// How many times the rewrite reads each operand of one comparison.
fn reads(op: CmpOperator) -> usize {
    match op {
        CmpOperator::Lt | CmpOperator::Gt => 1,
        CmpOperator::LtE | CmpOperator::GtE => 2,
        CmpOperator::Eq => 4,
        CmpOperator::NotEq => 3,
        _ => unreachable!("not a tolerant operator"),
    }
}

fn is_tolerant(op: CmpOperator) -> bool {
    matches!(
        op,
        CmpOperator::Lt
            | CmpOperator::Gt
            | CmpOperator::LtE
            | CmpOperator::GtE
            | CmpOperator::Eq
            | CmpOperator::NotEq
    )
}

/// Whether evaluating this operand can change what another operand reads.
fn may_rebind(ast: &Ast, id: NodeId) -> bool {
    !matches!(ast.kind(id), Kind::Name { .. } | Kind::Constant { .. })
}

/// A `str` / `bool` constant operand: the comparison is homogeneous non-float.
fn is_skippable_const(ast: &Ast, id: NodeId) -> bool {
    matches!(ast.kind(id), Kind::Constant { value: Constant::Str(_) | Constant::Bool(_), .. })
}

fn float(ast: &mut Ast, value: f64) -> NodeId {
    ast.add(Kind::Constant { value: Constant::Float(value), kind: None })
}

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.into(), ctx })
}

fn sub(ast: &mut Ast, left: NodeId, right: NodeId) -> NodeId {
    ast.add(Kind::BinOp { left, op: Operator::Sub, right })
}

fn bool_op(ast: &mut Ast, op: BoolOperator, values: Vec<NodeId>) -> NodeId {
    ast.add(Kind::BoolOp { op, values })
}

/// A comparison the rewrite emits: a bool by construction, marked so.
fn bool_compare(
    ast: &mut Ast,
    left: NodeId,
    ops: Vec<CmpOperator>,
    comparators: Vec<NodeId>,
) -> NodeId {
    let node = ast.add(Kind::Compare { left, ops, comparators });
    ast.set_extra(node, "pine_bool", Value::Bool(true));
    node
}

/// How an operand is read again.
#[derive(Clone)]
enum Reference {
    Constant(Constant),
    Name(String),
    /// A single-use operand: it is never read again
    Unreachable,
}

impl Reference {
    fn node(&self, ast: &mut Ast) -> Result<NodeId, Error> {
        match self {
            Reference::Constant(value) => {
                Ok(ast.add(Kind::Constant { value: value.clone(), kind: None }))
            }
            Reference::Name(id) => Ok(name(ast, id, ExprContext::Load)),
            Reference::Unreachable => {
                Err(Error::Unsupported("a single-use operand referenced more than once".into()))
            }
        }
    }
}

pub struct FloatTolerance {
    temp_counter: usize,
    na_bool: bool,
    na_bool_used: bool,
}

impl FloatTolerance {
    pub fn new(ctx: &Context<'_>) -> FloatTolerance {
        FloatTolerance { temp_counter: 0, na_bool: ctx.na_bool, na_bool_used: false }
    }

    /// `(first_use, reference)` for an operand: a name or a constant is re-read, an
    /// operand read only once stays in place, anything else is bound once by a walrus.
    fn binder(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        uses: usize,
        volatile: bool,
    ) -> (NodeId, Reference) {
        match ast.kind(node) {
            Kind::Constant { value, .. } => return (node, Reference::Constant(value.clone())),
            Kind::Name { id, .. } if !(volatile && uses > 1) => {
                return (node, Reference::Name(id.clone()));
            }
            _ => {}
        }
        if uses == 1 {
            return (node, Reference::Unreachable);
        }
        self.temp_counter += 1;
        let temp = format!("__cmp{}__", self.temp_counter);
        let target = name(ast, &temp, ExprContext::Store);
        (ast.add(Kind::NamedExpr { target, value: node }), Reference::Name(temp))
    }

    /// `operand.__class__ is float`.
    fn is_float(ast: &mut Ast, operand: NodeId) -> NodeId {
        let class = ast.add(Kind::Attribute {
            value: operand,
            attr: "__class__".into(),
            ctx: ExprContext::Load,
        });
        let float = name(ast, "float", ExprContext::Load);
        bool_compare(ast, class, vec![CmpOperator::Is], vec![float])
    }

    /// `not (operand == operand)`, the one operand node in both places.
    fn is_na(ast: &mut Ast, operand: NodeId) -> NodeId {
        let compare = bool_compare(ast, operand, vec![CmpOperator::Eq], vec![operand]);
        ast.add(Kind::UnaryOp { op: UnaryOperator::Not, operand: compare })
    }

    /// `(not (a == a) or not (b == b)) and na_bool`; `None` when neither operand can be na.
    fn na_tail(
        &mut self,
        ast: &mut Ast,
        operands: [(NodeId, &Reference); 2],
    ) -> Result<Option<NodeId>, Error> {
        let mut probes = Vec::new();
        for (operand, reference) in operands {
            if !matches!(ast.kind(operand), Kind::Constant { .. }) {
                let read = reference.node(ast)?;
                probes.push(FloatTolerance::is_na(ast, read));
            }
        }
        if probes.is_empty() {
            return Ok(None);
        }
        self.na_bool_used = true;
        let probe =
            if probes.len() == 1 { probes[0] } else { bool_op(ast, BoolOperator::Or, probes) };
        let na = name(ast, NA_BOOL_NAME, ExprContext::Load);
        Ok(Some(bool_op(ast, BoolOperator::And, vec![probe, na])))
    }

    /// Rewrite one `left op right` comparison into its tolerant form.
    fn rewrite_pair(
        &mut self,
        ast: &mut Ast,
        op: CmpOperator,
        (left, left_ref): (NodeId, &Reference),
        (right, right_ref): (NodeId, &Reference),
    ) -> Result<NodeId, Error> {
        if let Some((bound, cmp_op)) = ordering_form(op) {
            if matches!(op, CmpOperator::LtE | CmpOperator::GtE) {
                let raw = bool_compare(ast, left, vec![op], vec![right]);
                let bound = float(ast, bound);
                let left = left_ref.node(ast)?;
                let right = right_ref.node(ast)?;
                let difference = sub(ast, left, right);
                let tolerant = bool_compare(ast, bound, vec![cmp_op], vec![difference]);
                return Ok(bool_op(ast, BoolOperator::Or, vec![raw, tolerant]));
            }
            let bound = float(ast, bound);
            let difference = sub(ast, left, right);
            return Ok(bool_compare(ast, bound, vec![cmp_op], vec![difference]));
        }

        // `==` / `!=`: the tolerant branch is only reachable for real floats
        let right_bound = matches!(ast.kind(right), Kind::NamedExpr { .. });
        let guard_left = FloatTolerance::is_float(ast, left);
        let guard_right = FloatTolerance::is_float(ast, right);
        let guard = if right_bound {
            ast.add(Kind::BinOp { left: guard_left, op: Operator::BitOr, right: guard_right })
        } else {
            bool_op(ast, BoolOperator::Or, vec![guard_left, guard_right])
        };

        let tolerant = if op == CmpOperator::Eq {
            let a = left_ref.node(ast)?;
            let b = right_ref.node(ast)?;
            let exact = bool_compare(ast, a, vec![CmpOperator::Eq], vec![b]);
            let low = float(ast, -EPSILON);
            let a = left_ref.node(ast)?;
            let b = right_ref.node(ast)?;
            let difference = sub(ast, a, b);
            let high = float(ast, EPSILON);
            let within = bool_compare(
                ast,
                low,
                vec![CmpOperator::LtE, CmpOperator::LtE],
                vec![difference, high],
            );
            bool_op(ast, BoolOperator::Or, vec![exact, within])
        } else {
            self.temp_counter += 1;
            let diff = format!("__cmp{}__", self.temp_counter);
            let high = float(ast, EPSILON);
            let target = name(ast, &diff, ExprContext::Store);
            let a = left_ref.node(ast)?;
            let b = right_ref.node(ast)?;
            let difference = sub(ast, a, b);
            let bound = ast.add(Kind::NamedExpr { target, value: difference });
            let above = bool_compare(ast, high, vec![CmpOperator::Lt], vec![bound]);
            let low = float(ast, -EPSILON);
            let reread = name(ast, &diff, ExprContext::Load);
            let below = bool_compare(ast, low, vec![CmpOperator::Gt], vec![reread]);
            bool_op(ast, BoolOperator::Or, vec![above, below])
        };
        let a = left_ref.node(ast)?;
        let b = right_ref.node(ast)?;
        let exact = bool_compare(ast, a, vec![op], vec![b]);
        Ok(ast.add(Kind::IfExp { test: guard, body: tolerant, orelse: exact }))
    }

    fn leave_compare(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Error> {
        // An earlier pass may have emitted a comparison that already IS the tolerance
        if let Some(value) = ast.extra(node, "pine_exact") {
            match truthy(value) {
                Some(true) => return Ok(node),
                Some(false) => {}
                None => return Err(Error::Unsupported("an opaque pine_exact stamp".into())),
            }
        }
        let Kind::Compare { left, ops, comparators } = ast.kind(node) else { unreachable!() };
        if !ops.iter().all(|&op| is_tolerant(op)) {
            return Ok(node);
        }
        let ops = ops.clone();
        let operands: Vec<NodeId> =
            std::iter::once(*left).chain(comparators.iter().copied()).collect();
        if operands.iter().any(|&operand| is_skippable_const(ast, operand)) {
            return Ok(node);
        }

        // In a chain the inner operands are read by both of their clauses, and the na
        // tail reads each operand of every clause once more
        let mut uses = vec![0usize; operands.len()];
        let extra = usize::from(self.na_bool);
        for (i, &op) in ops.iter().enumerate() {
            let count = reads(op) + extra;
            uses[i] += count;
            uses[i + 1] += count;
        }
        let mut binders = Vec::with_capacity(operands.len());
        for (i, &operand) in operands.iter().enumerate() {
            let volatile = operands[i + 1..].iter().any(|&later| may_rebind(ast, later));
            binders.push(self.binder(ast, operand, uses[i], volatile));
        }
        let mut bound = vec![false; operands.len()];
        let mut clauses = Vec::with_capacity(ops.len());
        for (i, &op) in ops.iter().enumerate() {
            let mut pair = [NodeId::PLACEHOLDER; 2];
            for (slot, index) in [i, i + 1].into_iter().enumerate() {
                let (first, reference) = &binders[index];
                pair[slot] = if bound[index] { reference.node(ast)? } else { *first };
                bound[index] = true;
            }
            let mut clause =
                self.rewrite_pair(ast, op, (pair[0], &binders[i].1), (pair[1], &binders[i + 1].1))?;
            if self.na_bool
                && let Some(tail) = self.na_tail(
                    ast,
                    [(operands[i], &binders[i].1), (operands[i + 1], &binders[i + 1].1)],
                )?
            {
                clause = bool_op(ast, BoolOperator::Or, vec![clause, tail]);
            }
            clauses.push(clause);
        }
        // A chain evaluates the later operands only if the earlier comparisons hold
        let rewritten =
            if clauses.len() == 1 { clauses[0] } else { bool_op(ast, BoolOperator::And, clauses) };
        Ok(stamp_lowering(ast, rewritten, BOOL))
    }

    fn leave_module(&mut self, ast: &mut Ast, node: NodeId) {
        if !self.na_bool_used {
            return;
        }
        let Kind::Module { body, .. } = ast.kind(node) else { return };
        // After the docstring and the `__future__` block
        let mut position = 0;
        if let Some(&first) = body.first()
            && let Kind::Expr { value } = ast.kind(first)
            && matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
        {
            position = 1;
        }
        while position < body.len()
            && matches!(ast.kind(body[position]),
                Kind::ImportFrom { module: Some(module), .. } if module == "__future__")
        {
            position += 1;
        }
        let alias =
            ast.add(Kind::Alias { name: "na_bool".into(), asname: Some(NA_BOOL_NAME.into()) });
        let import = ast.add(Kind::ImportFrom {
            module: Some("pynecore.types.na".into()),
            names: vec![alias],
            level: Some(0),
        });
        let Kind::Module { body, .. } = ast.kind_mut(node) else { unreachable!() };
        body.insert(position, import);
    }
}

impl Rule for FloatTolerance {
    fn leave(
        &mut self,
        ast: &mut Ast,
        class: &'static str,
        node: Visited,
    ) -> Result<Visited, Error> {
        match class {
            "Compare" => Ok(Visited::Node(self.leave_compare(ast, single(&node, class)?)?)),
            "Module" => {
                self.leave_module(ast, single(&node, class)?);
                Ok(node)
            }
            _ => Ok(node),
        }
    }
}

fn make(ctx: &Context<'_>) -> Box<dyn Rule> {
    Box::new(FloatTolerance::new(ctx))
}

/// The rule on its own (sequential mode).
pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    run_rules(ast, module, vec![make(ctx)])
}

pub const RULE: crate::phase::RuleStep =
    crate::phase::RuleStep { name: "FloatTolerance", make, user_code_only: true };
