//! The nodes the security protocol is written with (the builders of
//! `SecurityTransformer`). They carry no location, as the Python builders.

use pyne_ast::walk::clone;
use pyne_ast::{Ast, BoolOperator, CmpOperator, Constant, ExprContext, Int, Kind, NodeId};

use super::Security;

/// `ast.Constant(value=...)`.
pub(crate) fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

/// A str constant.
pub(crate) fn str_constant(ast: &mut Ast, value: &str) -> NodeId {
    constant(ast, Constant::Str(value.to_string()))
}

/// An int constant.
pub(crate) fn int_constant(ast: &mut Ast, value: i64) -> NodeId {
    constant(ast, Constant::Int(Int::Small(value)))
}

/// A `Load` name.
pub(crate) fn load(ast: &mut Ast, id: &str) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx: ExprContext::Load })
}

/// A `Load` list of str constants.
pub(crate) fn str_list<S: AsRef<str>>(ast: &mut Ast, items: &[S]) -> NodeId {
    let elts = items.iter().map(|item| str_constant(ast, item.as_ref())).collect();
    ast.add(Kind::List { elts, ctx: ExprContext::Load })
}

/// `lib.na`
pub(crate) fn lib_na(ast: &mut Ast) -> NodeId {
    let lib = load(ast, "lib");
    ast.add(Kind::Attribute { value: lib, attr: "na".into(), ctx: ExprContext::Load })
}

/// `lib.barmerge.gaps_off`
pub(crate) fn default_gaps(ast: &mut Ast) -> NodeId {
    let lib = load(ast, "lib");
    let barmerge =
        ast.add(Kind::Attribute { value: lib, attr: "barmerge".into(), ctx: ExprContext::Load });
    ast.add(Kind::Attribute { value: barmerge, attr: "gaps_off".into(), ctx: ExprContext::Load })
}

/// `name(*args)`
pub(crate) fn func_call(ast: &mut Ast, name: &str, args: Vec<NodeId>) -> NodeId {
    let func = load(ast, name);
    ast.add(Kind::Call { func, args, keywords: Vec::new() })
}

/// `__active_security__ is None`
pub(crate) fn is_none_check(ast: &mut Ast) -> NodeId {
    let left = load(ast, "__active_security__");
    let none = constant(ast, Constant::None);
    ast.add(Kind::Compare { left, ops: vec![CmpOperator::Is], comparators: vec![none] })
}

/// `if __active_security__ is None: <body>`
pub(crate) fn guard_block(ast: &mut Ast, body: Vec<NodeId>) -> NodeId {
    let test = is_none_check(ast);
    ast.add(Kind::If { test, body, orelse: Vec::new() })
}

/// An expression statement.
pub(crate) fn expr_stmt(ast: &mut Ast, value: NodeId) -> NodeId {
    ast.add(Kind::Expr { value })
}

/// `__active_security__ is not None and sec_id in __active_security__`
///
/// `__active_security__` is the SET of contexts the running process serves -- one
/// security child can serve a whole group -- so the guard is a membership test,
/// like `__same_context__`. It stays None in the chart process (that is what the
/// signal / wait blocks test for), which the left conjunct guards against.
fn in_active_check(ast: &mut Ast, sec_id: &str) -> NodeId {
    let active = load(ast, "__active_security__");
    let none = constant(ast, Constant::None);
    let not_none = ast.add(Kind::Compare {
        left: active,
        ops: vec![CmpOperator::IsNot],
        comparators: vec![none],
    });
    let sid = str_constant(ast, sec_id);
    let active = load(ast, "__active_security__");
    let member =
        ast.add(Kind::Compare { left: sid, ops: vec![CmpOperator::In], comparators: vec![active] });
    ast.add(Kind::BoolOp { op: BoolOperator::And, values: vec![not_none, member] })
}

/// `sec_id in __same_context__`
fn in_same_context(ast: &mut Ast, sec_id: &str) -> NodeId {
    let sid = str_constant(ast, sec_id);
    let same = load(ast, "__same_context__");
    ast.add(Kind::Compare { left: sid, ops: vec![CmpOperator::In], comparators: vec![same] })
}

/// The arguments of one `__sec_signal__`: the sid, the symbol and timeframe (None
/// when absent), and the runtime lookahead when there is one.
fn signal_args(ast: &mut Ast, sec_id: &str, args: [Option<NodeId>; 3]) -> Vec<NodeId> {
    let [symbol, timeframe, lookahead] = args;
    let mut out = vec![str_constant(ast, sec_id)];
    for expr in [symbol, timeframe] {
        out.push(match expr {
            Some(expr) => clone(ast, expr, &mut Default::default()),
            None => constant(ast, Constant::None),
        });
    }
    // A lookahead that is not module-level evaluable (an input-derived Pine
    // "simple" value) is still passed here when its chain is hoistable -- the
    // binding it reads is moved above this block.
    if let Some(lookahead) = lookahead {
        out.push(clone(ast, lookahead, &mut Default::default()));
    }
    out
}

impl Security {
    /// The chart-context signal block for function start (`_signal_block`).
    pub(crate) fn signal_block(&self, ast: &mut Ast, sec_ids: &[String]) -> NodeId {
        let mut body = Vec::with_capacity(sec_ids.len());
        for sid in sec_ids {
            let args = signal_args(ast, sid, self.signal_args[sid]);
            let call = func_call(ast, "__sec_signal__", args);
            body.push(expr_stmt(ast, call));
        }
        guard_block(ast, body)
    }

    /// A single inline signal for a runtime-dependent symbol / timeframe /
    /// lookahead (`_inline_signal`).
    pub(crate) fn inline_signal(&self, ast: &mut Ast, sec_id: &str) -> NodeId {
        let args = signal_args(ast, sec_id, self.signal_args[sec_id]);
        let call = func_call(ast, "__sec_signal__", args);
        let stmt = expr_stmt(ast, call);
        guard_block(ast, vec![stmt])
    }
}

/// The chart-context wait block for function end (`_wait_block`).
pub(crate) fn wait_block(ast: &mut Ast, sec_ids: &[String]) -> NodeId {
    let test = is_none_check(ast);
    let body = sec_ids
        .iter()
        .map(|sid| {
            let sid = str_constant(ast, sid);
            let call = func_call(ast, "__sec_wait__", vec![sid]);
            expr_stmt(ast, call)
        })
        .collect();
    ast.add(Kind::If { test, body, orelse: Vec::new() })
}

/// The security-context write block (`_write_block`). It fires in the security
/// process serving the sid, and in the chart process when the sid is same-context.
pub(crate) fn write_block(ast: &mut Ast, sec_id: &str, expression: NodeId) -> NodeId {
    let active = in_active_check(ast, sec_id);
    let same = in_same_context(ast, sec_id);
    let test = ast.add(Kind::BoolOp { op: BoolOperator::Or, values: vec![active, same] });
    let sid = str_constant(ast, sec_id);
    let call = func_call(ast, "__sec_write__", vec![sid, expression]);
    let stmt = expr_stmt(ast, call);
    ast.add(Kind::If { test, body: vec![stmt], orelse: Vec::new() })
}

/// `__sec_read__("sec_id", <default>)` (`_sec_read_call`): `lib.na`, or an N-tuple
/// of it when the target unpacks the result -- Pine's `request.security()` returns
/// a tuple of `na` on no-data bars, and a scalar would crash the unpack.
pub(crate) fn sec_read_call(ast: &mut Ast, sec_id: &str, tuple_len: Option<i64>) -> NodeId {
    let default = match tuple_len {
        None => lib_na(ast),
        Some(len) => {
            let elts = (0..len).map(|_| lib_na(ast)).collect();
            ast.add(Kind::Tuple { elts, ctx: ExprContext::Load })
        }
    };
    let sid = str_constant(ast, sec_id);
    func_call(ast, "__sec_read__", vec![sid, default])
}

/// The lower-timeframe read (`_sec_read_call_ltf`): `__sec_read__("sec_id", [])`, or
/// wrapped in `__ltf_unzip__(..., arity)` to transpose a tuple expression's
/// row-major intrabar buffer into `arity` column arrays.
pub(crate) fn sec_read_call_ltf(ast: &mut Ast, sec_id: &str, arity: Option<i64>) -> NodeId {
    let sid = str_constant(ast, sec_id);
    let empty = ast.add(Kind::List { elts: Vec::new(), ctx: ExprContext::Load });
    let read = func_call(ast, "__sec_read__", vec![sid, empty]);
    match arity {
        None => read,
        Some(arity) => {
            let arity = int_constant(ast, arity);
            func_call(ast, "__ltf_unzip__", vec![read, arity])
        }
    }
}
