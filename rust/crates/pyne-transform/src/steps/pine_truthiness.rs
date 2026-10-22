//! Port of `transformers/pine_truthiness.py`: TradingView's tolerant float-to-bool
//! conversion in every bool context.
//!
//! A test that is not already a bool becomes
//! `(-EPSILON > x or EPSILON < x) if x.__class__ is float else not not x`, an
//! operand that is neither a name nor a constant bound once by a `__bool<n>__`
//! walrus inside the type guard. The bounds are marked `pine_exact`.

use pyne_ast::walk::{Transformer, Visited, generic_visit};
use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, ExprContext, Int, Kind, NodeId, UnaryOperator, Value,
};

use crate::{Context, Error};

/// Pine's comparison tolerance, `pynecore.core.pine_compare.EPSILON`.
const EPSILON: f64 = 1e-10;

/// Whether the expression provably evaluates to a Python bool already.
fn is_bool_valued(ast: &Ast, id: NodeId) -> bool {
    match ast.kind(id) {
        Kind::Compare { .. } | Kind::BoolOp { .. } => true,
        Kind::UnaryOp { op: UnaryOperator::Not, .. } => true,
        Kind::Constant { value, .. } => matches!(value, Constant::None | Constant::Bool(_)),
        &Kind::IfExp { body, orelse, .. } => {
            is_bool_valued(ast, body) && is_bool_valued(ast, orelse)
        }
        _ => false,
    }
}

/// Python's truth value of a stamped attribute; None for an object the step does not look into.
pub(crate) fn truthy(value: &Value) -> Option<bool> {
    Some(match value {
        Value::None => false,
        Value::Bool(flag) => *flag,
        Value::Int(Int::Small(number)) => *number != 0,
        Value::Int(Int::Big(_)) => true,
        Value::Float(number) => *number != 0.0,
        Value::Str(text) => !text.is_empty(),
        Value::Tuple(items) | Value::List(items) => !items.is_empty(),
        Value::Dict(items) => !items.is_empty(),
        Value::Node(_) => true,
        Value::Opaque(_) | Value::Native(_) => return None,
    })
}

fn not(ast: &mut Ast, operand: NodeId) -> NodeId {
    ast.add(Kind::UnaryOp { op: UnaryOperator::Not, operand })
}

fn float_constant(ast: &mut Ast, value: f64) -> NodeId {
    ast.add(Kind::Constant { value: Constant::Float(value), kind: None })
}

/// How the converted operand is read again: a fresh node per reference.
enum Reference {
    Name(String),
    Constant(Constant),
}

impl Reference {
    fn node(&self, ast: &mut Ast) -> NodeId {
        match self {
            Reference::Name(id) => ast.add(Kind::Name { id: id.clone(), ctx: ExprContext::Load }),
            Reference::Constant(value) => {
                ast.add(Kind::Constant { value: value.clone(), kind: None })
            }
        }
    }
}

struct PineTruthiness {
    temp_counter: usize,
    /// Why the pass cannot answer for the module, once it met such a construct
    unsupported: Option<String>,
}

impl PineTruthiness {
    /// Wrap one expression in Pine's float-to-bool conversion.
    fn convert(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        if is_bool_valued(ast, id) {
            return id;
        }
        // A name or a constant is side-effect free, so it may simply be re-read;
        // anything else can run user code, so it is bound once instead
        let (guarded, reference) = match ast.kind(id) {
            Kind::Name { id: name, .. } => (id, Reference::Name(name.clone())),
            Kind::Constant { value, .. } => (id, Reference::Constant(value.clone())),
            _ => {
                self.temp_counter += 1;
                let temp = format!("__bool{}__", self.temp_counter);
                let target = ast.add(Kind::Name { id: temp.clone(), ctx: ExprContext::Store });
                (ast.add(Kind::NamedExpr { target, value: id }), Reference::Name(temp))
            }
        };

        let left = float_constant(ast, -EPSILON);
        let right = reference.node(ast);
        let below =
            ast.add(Kind::Compare { left, ops: vec![CmpOperator::Gt], comparators: vec![right] });
        let left = float_constant(ast, EPSILON);
        let right = reference.node(ast);
        let above =
            ast.add(Kind::Compare { left, ops: vec![CmpOperator::Lt], comparators: vec![right] });
        // The bounds ARE the tolerance; the comparison rewrite must not widen them
        ast.set_extra(below, "pine_exact", Value::Bool(true));
        ast.set_extra(above, "pine_exact", Value::Bool(true));

        let class = ast.add(Kind::Attribute {
            value: guarded,
            attr: "__class__".to_string(),
            ctx: ExprContext::Load,
        });
        let float = ast.add(Kind::Name { id: "float".to_string(), ctx: ExprContext::Load });
        let test = ast.add(Kind::Compare {
            left: class,
            ops: vec![CmpOperator::Is],
            comparators: vec![float],
        });
        let body = ast.add(Kind::BoolOp { op: BoolOperator::Or, values: vec![below, above] });
        let reread = reference.node(ast);
        let inner = not(ast, reread);
        let orelse = not(ast, inner);
        ast.add(Kind::IfExp { test, body, orelse })
    }

    /// Force an `and` / `or` operand to a real bool where it could still be na.
    fn bool_value(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        if let &Kind::IfExp { body, orelse, .. } = ast.kind(id) {
            let body = self.bool_value(ast, body);
            let Kind::IfExp { body: slot, .. } = ast.kind_mut(id) else { unreachable!() };
            *slot = body;
            let orelse = self.bool_value(ast, orelse);
            let Kind::IfExp { orelse: slot, .. } = ast.kind_mut(id) else { unreachable!() };
            *slot = orelse;
            return id;
        }
        if !matches!(ast.kind(id), Kind::Compare { .. }) {
            return id;
        }
        match ast.extra(id, "pine_bool").map(truthy) {
            Some(Some(true)) => return id,
            Some(None) => {
                self.unsupported = Some("an opaque pine_bool stamp".into());
                return id;
            }
            Some(Some(false)) | None => {}
        }
        let inner = not(ast, id);
        not(ast, inner)
    }
}

impl Transformer for PineTruthiness {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        generic_visit(self, ast, id);
        match ast.kind(id) {
            &Kind::If { test, .. } => {
                let test = self.convert(ast, test);
                let Kind::If { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = test;
            }
            &Kind::While { test, .. } => {
                let test = self.convert(ast, test);
                let Kind::While { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = test;
            }
            &Kind::IfExp { test, .. } => {
                let test = self.convert(ast, test);
                let Kind::IfExp { test: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = test;
            }
            Kind::BoolOp { values, .. } => {
                let values = values.clone();
                let new_values: Vec<NodeId> = values
                    .into_iter()
                    .map(|value| {
                        let converted = self.convert(ast, value);
                        self.bool_value(ast, converted)
                    })
                    .collect();
                let Kind::BoolOp { values, .. } = ast.kind_mut(id) else { unreachable!() };
                *values = new_values;
            }
            &Kind::UnaryOp { op: UnaryOperator::Not, operand } => {
                let operand = self.convert(ast, operand);
                let Kind::UnaryOp { operand: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = operand;
            }
            _ => {}
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("PineTruthiness on a non-module".into()));
    }
    let mut pass = PineTruthiness { temp_counter: 0, unsupported: None };
    generic_visit(&mut pass, ast, module);
    match pass.unsupported {
        Some(message) => Err(Error::Unsupported(message)),
        None => Ok(module),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoHost;

    /// `a == 1 or b`, the comparison stamped `pine_bool`.
    fn module(ast: &mut Ast, pine_bool: Value) -> NodeId {
        let a = ast.add(Kind::Name { id: "a".into(), ctx: ExprContext::Load });
        let one = ast.add(Kind::Constant { value: Constant::Int(Int::Small(1)), kind: None });
        let compare =
            ast.add(Kind::Compare { left: a, ops: vec![CmpOperator::Eq], comparators: vec![one] });
        ast.set_extra(compare, "pine_bool", pine_bool);
        let b = ast.add(Kind::Name { id: "b".into(), ctx: ExprContext::Load });
        let test = ast.add(Kind::BoolOp { op: BoolOperator::Or, values: vec![compare, b] });
        let statement = ast.add(Kind::Expr { value: test });
        ast.add(Kind::Module { body: vec![statement], type_ignores: vec![] })
    }

    #[test]
    fn an_opaque_pine_bool_is_left_to_python() {
        let ctx = Context {
            path: String::new(),
            pyne_mode: None,
            source: String::new(),
            user_code: true,
            na_bool: false,
            sequential: false,
            security_slice_disabled: false,
            host: &NoHost,
            slot_layout: Default::default(),
        };
        let mut ast = Ast::new();
        let root = module(&mut ast, Value::Opaque(0));
        assert!(matches!(run(&mut ast, root, &ctx), Err(Error::Unsupported(_))));
        let mut ast = Ast::new();
        let root = module(&mut ast, Value::Bool(true));
        assert!(run(&mut ast, root, &ctx).is_ok());
    }
}
