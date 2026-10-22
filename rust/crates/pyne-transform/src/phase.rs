//! Port of `transformers/phase.py`: several rules over ONE traversal of the tree.
//!
//! A [`Rule`] is a pass written as per-node hooks. [`run_rules`] runs several of them
//! in a single post-order traversal: at every node the rules' `enter` hooks run in
//! order, then the children are visited (by every rule), then the `leave` hooks run
//! in the same order, each handed what the previous one answered. The `body` of a
//! `def` or `async def` is bracketed by the function-body hooks; the decorators,
//! defaults and annotations, visited outside them, belong to the enclosing scope.
//! A rule run alone is a phase of one rule, which is what sequential mode runs.

use pyne_ast::walk::{Transformer, Visited, visit_field};
use pyne_ast::{Ast, NodeId};

use crate::{Context, Error};

/// One rule of a phase. A hook is keyed by the class name of the node it was
/// dispatched for (`enter_<Class>` / `leave_<Class>` in Python); a rule answers the
/// classes it has no hook for with a no-op.
pub trait Rule {
    /// `enter_<class>`: before the node's children are visited.
    fn enter(&mut self, ast: &mut Ast, class: &'static str, node: NodeId) -> Result<(), Error> {
        let _ = (ast, class, node);
        Ok(())
    }

    /// `leave_<class>`: after the children, handed what the previous rule's leave
    /// answered for the node (the node itself when nothing changed); answers what
    /// replaces it.
    fn leave(
        &mut self,
        ast: &mut Ast,
        class: &'static str,
        node: Visited,
    ) -> Result<Visited, Error> {
        let _ = (ast, class);
        Ok(node)
    }

    /// `enter_function_body`: before the `body` of a `def` or `async def`.
    fn enter_function_body(&mut self, node: NodeId) {
        let _ = node;
    }

    /// `leave_function_body`: after the `body` of a `def` or `async def`.
    fn leave_function_body(&mut self, node: NodeId) {
        let _ = node;
    }
}

/// The node a leave hook was handed, or `Unsupported` when an earlier rule replaced
/// it with a list or removed it (the Python hook would be handed that list).
pub fn single(node: &Visited, class: &str) -> Result<NodeId, Error> {
    match node {
        Visited::Node(id) => Ok(*id),
        _ => Err(Error::Unsupported(format!("a {class} rule hook handed a list"))),
    }
}

/// One rule as a phase declares it (`pipeline.RuleStep`).
pub struct RuleStep {
    pub name: &'static str,
    /// Builds the rule for one module
    pub make: fn(&Context<'_>) -> Box<dyn Rule>,
    /// Only user code gets this rule, never pynecore's own `@pyne` lib modules
    pub user_code_only: bool,
}

/// Rules that run in one traversal, equivalent to running them in this order
/// (`pipeline.Phase`).
pub struct Phase {
    pub name: &'static str,
    pub rules: &'static [RuleStep],
}

impl Phase {
    /// Run the rules that apply to the module: in one traversal, or one pass per
    /// rule when the context asks for sequential mode.
    pub fn run(
        &self,
        ast: &mut Ast,
        mut module: NodeId,
        ctx: &Context<'_>,
    ) -> Result<NodeId, Error> {
        let rules = self.rules.iter().filter(|rule| ctx.user_code || !rule.user_code_only);
        if ctx.sequential {
            for rule in rules {
                module = run_rules(ast, module, vec![(rule.make)(ctx)])?;
            }
            Ok(module)
        } else {
            let rules = rules.map(|rule| (rule.make)(ctx)).collect();
            run_rules(ast, module, rules)
        }
    }
}

struct Walker {
    rules: Vec<Box<dyn Rule>>,
    /// Why the phase cannot answer for the module, once a hook said so
    error: Option<Error>,
}

impl Walker {
    fn visit_node(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let class = ast.name(id);
        for rule in &mut self.rules {
            rule.enter(ast, class, id)?;
        }
        let function = matches!(class, "FunctionDef" | "AsyncFunctionDef");
        let fields = ast.kind(id).class().fields;
        for (index, field) in fields.iter().enumerate() {
            if function && field.name == "body" {
                for rule in &mut self.rules {
                    rule.enter_function_body(id);
                }
                visit_field(self, ast, id, index);
                for rule in &mut self.rules {
                    rule.leave_function_body(id);
                }
            } else {
                visit_field(self, ast, id, index);
            }
            if let Some(error) = self.error.take() {
                return Err(error);
            }
        }
        let mut node = Visited::Node(id);
        for rule in &mut self.rules {
            node = rule.leave(ast, class, node)?;
        }
        Ok(node)
    }
}

impl Transformer for Walker {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        match self.visit_node(ast, id) {
            Ok(visited) => visited,
            Err(error) => {
                self.error = Some(error);
                Visited::Node(id)
            }
        }
    }
}

/// Run several rules over a tree in one traversal (`phase.run_phase`).
pub fn run_rules(ast: &mut Ast, root: NodeId, rules: Vec<Box<dyn Rule>>) -> Result<NodeId, Error> {
    let mut walker = Walker { rules, error: None };
    match walker.visit_node(ast, root)? {
        Visited::Node(id) => Ok(id),
        _ => Err(Error::Unsupported("a phase replaced the module with a list".into())),
    }
}
