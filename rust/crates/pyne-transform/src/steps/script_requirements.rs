//! Port of `transformers/script_requirements.py`: the broker capabilities a strategy
//! needs, deduced from the keyword arguments of its `strategy.*` order calls, are
//! injected into its `@script.strategy(...)` decorator as
//! `_broker_requirements=ScriptRequirements(...)`, imported at the top of the module.

use pyne_ast::locations::fix_locations;
use pyne_ast::walk::{children, future_end, walk_statements};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// The `ScriptRequirements` flags, in the order the keywords are emitted.
#[derive(Clone, Copy)]
enum Flag {
    Market,
    Limit,
    Stop,
    Bracket,
    Trail,
    StrategyOrder,
    ExitOrders,
    PartialQtyBracketExit,
    MayGoShort,
}

const FLAG_NAMES: [&str; 9] = [
    "market_orders",
    "limit_orders",
    "stop_orders",
    "tp_sl_bracket",
    "trailing_stop",
    "strategy_order",
    "exit_orders",
    "partial_qty_bracket_exit",
    "may_go_short",
];

fn is_name(ast: &Ast, id: NodeId, name: &str) -> bool {
    matches!(ast.kind(id), Kind::Name { id, .. } if id == name)
}

/// The method of a `(lib.)strategy.<method>(...)` call.
fn strategy_call_name(ast: &Ast, call: NodeId) -> Option<&str> {
    let Kind::Call { func, .. } = ast.kind(call) else { return None };
    let Kind::Attribute { value: parent, attr: method, .. } = ast.kind(*func) else {
        return None;
    };
    if is_name(ast, *parent, "strategy") {
        return Some(method);
    }
    if let Kind::Attribute { value, attr, .. } = ast.kind(*parent)
        && attr == "strategy"
        && is_name(ast, *value, "lib")
    {
        return Some(method);
    }
    None
}

/// Whether `expr` is a chain a direction constant hangs off: `strategy`, `direction`,
/// `lib.strategy` and the `.direction` of any of them.
fn is_strategy_direction_base(ast: &Ast, expr: NodeId) -> bool {
    match ast.kind(expr) {
        Kind::Name { id, .. } => id == "strategy" || id == "direction",
        Kind::Attribute { value, attr, .. } => match attr.as_str() {
            "direction" => is_strategy_direction_base(ast, *value),
            "strategy" => is_name(ast, *value, "lib"),
            _ => false,
        },
        _ => false,
    }
}

/// Whether the `direction` of an entry or order call is a constant short.
fn direction_is_constant_short(ast: &Ast, call: NodeId) -> bool {
    let Kind::Call { args, keywords, .. } = ast.kind(call) else { return false };
    let mut expr = args.get(1).copied();
    for &keyword in keywords {
        if let Kind::Keyword { arg: Some(arg), value } = ast.kind(keyword)
            && arg == "direction"
        {
            expr = Some(*value);
        }
    }
    let Some(expr) = expr else { return false };
    matches!(ast.kind(expr), Kind::Attribute { value, attr, .. }
        if attr == "short" && is_strategy_direction_base(ast, *value))
}

/// Whether a decorator is a `@(lib.)script.strategy(...)` call.
fn is_script_strategy_decorator(ast: &Ast, node: NodeId) -> bool {
    let Kind::Call { func, .. } = ast.kind(node) else { return false };
    let Kind::Attribute { value: parent, attr, .. } = ast.kind(*func) else { return false };
    if attr != "strategy" {
        return false;
    }
    if is_name(ast, *parent, "script") {
        return true;
    }
    matches!(ast.kind(*parent), Kind::Attribute { value, attr, .. }
        if attr == "script" && is_name(ast, *value, "lib"))
}

fn decorators(ast: &Ast, id: NodeId) -> Option<&[NodeId]> {
    match ast.kind(id) {
        Kind::FunctionDef { decorator_list, .. }
        | Kind::AsyncFunctionDef { decorator_list, .. } => Some(decorator_list),
        _ => None,
    }
}

struct Requirements {
    flags: [bool; 9],
    strategy_decorator: Option<NodeId>,
}

impl Requirements {
    fn set(&mut self, flag: Flag) {
        self.flags[flag as usize] = true;
    }

    fn apply_entry_or_order(&mut self, kws: &[&str], is_strategy_order: bool) {
        let has_limit = kws.contains(&"limit");
        let has_stop = kws.contains(&"stop");
        if is_strategy_order {
            self.set(Flag::StrategyOrder);
        }
        if has_limit && has_stop {
            // Pine has no stop-limit entry: a both-set order is two OCO legs, the
            // limit leg resting natively and the stop leg fired as a market order
            self.set(Flag::Limit);
            self.set(Flag::Market);
        } else if has_limit {
            self.set(Flag::Limit);
        } else if has_stop {
            self.set(Flag::Stop);
        } else {
            self.set(Flag::Market);
        }
    }

    fn apply_exit(&mut self, kws: &[&str]) {
        let has = |name: &str| kws.contains(&name);
        let has_limit = has("limit");
        let has_stop = has("stop");
        let has_profit_ticks = has("profit");
        let has_loss_ticks = has("loss");
        let has_trail = has("trail_offset") || has("trail_price") || has("trail_points");
        let has_qty = has("qty");
        let has_bracket_leg =
            has_limit || has_stop || has_profit_ticks || has_loss_ticks || has_trail;

        if (has_limit && has_stop) || (has_profit_ticks && has_loss_ticks) {
            self.set(Flag::Bracket);
            self.set(Flag::Limit);
            self.set(Flag::Stop);
        } else {
            if has_limit || has_profit_ticks {
                self.set(Flag::Limit);
            }
            if has_stop || has_loss_ticks {
                self.set(Flag::Stop);
            }
        }
        if has_trail {
            self.set(Flag::Trail);
        }
        // An explicit exit qty with any bracket leg cannot be proven to cover the
        // whole position at compile time
        if has_qty && has_bracket_leg {
            self.set(Flag::PartialQtyBracketExit);
        }
    }

    fn visit_call(&mut self, ast: &Ast, call: NodeId) {
        let Some(name) = strategy_call_name(ast, call) else { return };
        let Kind::Call { keywords, .. } = ast.kind(call) else { unreachable!() };
        let kws: Vec<&str> = keywords
            .iter()
            .filter_map(|&keyword| match ast.kind(keyword) {
                Kind::Keyword { arg: Some(arg), .. } => Some(arg.as_str()),
                _ => None,
            })
            .collect();
        match name {
            "entry" | "order" => {
                self.apply_entry_or_order(&kws, name == "order");
                if direction_is_constant_short(ast, call) {
                    self.set(Flag::MayGoShort);
                }
            }
            "exit" => {
                self.apply_exit(&kws);
                self.set(Flag::ExitOrders);
            }
            "close" | "close_all" => {
                self.set(Flag::Market);
                self.set(Flag::ExitOrders);
            }
            _ => {}
        }
    }

    /// The visit of the Python pass, in its pre-order: the strategy decorator the
    /// module ends up with is the last one a definition carries.
    fn visit(&mut self, ast: &Ast, root: NodeId) {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(decorators) = decorators(ast, id)
                && let Some(&found) =
                    decorators.iter().find(|&&dec| is_script_strategy_decorator(ast, dec))
            {
                self.strategy_decorator = Some(found);
            }
            if matches!(ast.kind(id), Kind::Call { .. }) {
                self.visit_call(ast, id);
            }
            stack.extend(children(ast, id).into_iter().rev());
        }
    }

    /// Append `_broker_requirements=ScriptRequirements(...)` and add the import.
    fn inject(&self, ast: &mut Ast, module: NodeId, decorator: NodeId) {
        let mut keywords = Vec::new();
        for (index, &on) in self.flags.iter().enumerate() {
            if on {
                let value = ast.add(Kind::Constant { value: Constant::Bool(true), kind: None });
                keywords
                    .push(ast.add(Kind::Keyword { arg: Some(FLAG_NAMES[index].into()), value }));
            }
        }
        let func = ast.add(Kind::Name { id: "ScriptRequirements".into(), ctx: ExprContext::Load });
        let requirements = ast.add(Kind::Call { func, args: Vec::new(), keywords });
        let keyword = ast
            .add(Kind::Keyword { arg: Some("_broker_requirements".into()), value: requirements });
        let Kind::Call { keywords, .. } = ast.kind(decorator) else { unreachable!() };
        let mut kept: Vec<NodeId> = keywords
            .iter()
            .copied()
            .filter(|&kw| {
                !matches!(ast.kind(kw), Kind::Keyword { arg: Some(arg), .. }
                    if arg == "_broker_requirements")
            })
            .collect();
        kept.push(keyword);
        let Kind::Call { keywords, .. } = ast.kind_mut(decorator) else { unreachable!() };
        *keywords = kept;

        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let imported = body.iter().any(|&stmt| match ast.kind(stmt) {
            Kind::ImportFrom { module: Some(name), names, .. }
                if name == "pynecore.core.broker.models" =>
            {
                names.iter().any(|&alias| {
                    matches!(ast.kind(alias), Kind::Alias { name, .. } if name == "ScriptRequirements")
                })
            }
            _ => false,
        });
        if imported {
            return;
        }
        let at = future_end(ast, body);
        let alias = ast.add(Kind::Alias { name: "ScriptRequirements".into(), asname: None });
        let import = ast.add(Kind::ImportFrom {
            module: Some("pynecore.core.broker.models".into()),
            names: vec![alias],
            level: Some(0),
        });
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        body.insert(at, import);
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("ScriptRequirements on a non-module".into()));
    }
    // The requirements only ever land on a strategy decorator, and a definition can
    // only stand among statements
    let strategy = walk_statements(ast, module).into_iter().any(|stmt| {
        decorators(ast, stmt)
            .is_some_and(|list| list.iter().any(|&dec| is_script_strategy_decorator(ast, dec)))
    });
    if !strategy {
        return Ok(module);
    }
    let mut pass = Requirements { flags: [false; 9], strategy_decorator: None };
    pass.visit(ast, module);
    let decorator = pass.strategy_decorator.expect("a strategy decorator was found");
    pass.inject(ast, module, decorator);
    fix_locations(ast, module, 1, 0)?;
    Ok(module)
}
