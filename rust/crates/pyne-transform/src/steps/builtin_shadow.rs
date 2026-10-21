//! Port of `transformers/builtin_shadow.py`: resolve workdir library imports whose
//! alias shadows a built-in namespace.
//!
//! An attribute read through such an alias that the library does not serve is
//! rewritten to the canonical built-in form (`lib.ta.valuewhen`). Which members a
//! library serves is the host's knowledge ([`crate::Host`]): it is asked where the
//! Python step imports the library and where it tests a member.

use std::collections::HashMap;

use pyne_ast::walk::{Transformer, Visited, copy_location, generic_visit};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error, Host, Unsupported};

/// Pine namespaces whose pynecore module has a different name
fn namespace_rename(name: &str) -> &str {
    match name {
        "str" => "string",
        other => other,
    }
}

struct BuiltinShadow<'h> {
    host: &'h dyn Host,
    /// alias -> (library module name, builtin namespace)
    aliases: HashMap<String, (String, String)>,
    /// The first host error: the step then declines the module
    error: Option<Unsupported>,
}

impl BuiltinShadow<'_> {
    fn visit_scope(&mut self, ast: &mut Ast, id: NodeId) {
        let args = match ast.kind(id) {
            Kind::FunctionDef { args, .. }
            | Kind::AsyncFunctionDef { args, .. }
            | Kind::Lambda { args, .. } => *args,
            _ => unreachable!(),
        };
        let Kind::Arguments { posonlyargs, args: positional, vararg, kwonlyargs, kwarg, .. } =
            ast.kind(args)
        else {
            unreachable!()
        };
        let mut masked = Vec::new();
        for &param in
            posonlyargs.iter().chain(positional).chain(kwonlyargs).chain(vararg).chain(kwarg)
        {
            let Kind::Arg { arg, .. } = ast.kind(param) else { unreachable!() };
            if let Some(entry) = self.aliases.remove(arg) {
                masked.push((arg.clone(), entry));
            }
        }
        generic_visit(self, ast, id);
        self.aliases.extend(masked);
    }

    fn visit_attribute(&mut self, ast: &mut Ast, id: NodeId) -> Result<NodeId, Unsupported> {
        generic_visit(self, ast, id);
        let Kind::Attribute { value, attr, ctx } = ast.kind(id) else { unreachable!() };
        let (attr, ctx) = (attr.clone(), *ctx);
        let Kind::Name { id: base, .. } = ast.kind(*value) else { return Ok(id) };
        if ctx != ExprContext::Load {
            return Ok(id);
        }
        let Some((library, namespace)) = self.aliases.get(base).cloned() else { return Ok(id) };

        // The library serves the name -> the access stays on the library
        if self.host.library_serves(&library, &attr)? {
            return Ok(id);
        }

        // Built-in namespace fallback -> canonical lib.<namespace>.<attr> form; the
        // nested-key check covers sub-namespaces (e.g. strategy.commission)
        let module = format!("lib.{namespace}");
        let known = |attr: &str| -> Result<bool, Unsupported> {
            Ok(self.host.module_property(&module, attr)?
                || self.host.has_module_properties(&format!("{module}.{attr}"))?)
        };
        let mut attr = attr;
        if !known(&attr)? && attr.ends_with("__ren__") {
            // A trigger-named member is emitted under its canonical image; when the
            // library does not serve it, the built-in namespace knows the bare name only
            attr.truncate(attr.len() - "__ren__".len());
        }
        if !known(&attr)? {
            // In neither -> leave it to fail at runtime, as before
            return Ok(id);
        }
        let lib = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
        let inner =
            ast.add(Kind::Attribute { value: lib, attr: namespace, ctx: ExprContext::Load });
        let outer = ast.add(Kind::Attribute { value: inner, attr, ctx });
        copy_location(ast, outer, id);
        Ok(outer)
    }
}

impl Transformer for BuiltinShadow<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. } => {
                self.visit_scope(ast, id);
            }
            Kind::Attribute { .. } => match self.visit_attribute(ast, id) {
                Ok(new) => return Visited::Node(new),
                Err(err) => {
                    self.error.get_or_insert(err);
                }
            },
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("BuiltinShadow on a non-module".into()));
    };
    let mut aliases = HashMap::new();
    for &stmt in body {
        let Kind::Import { names } = ast.kind(stmt) else { continue };
        for &alias in names {
            let Kind::Alias { name: module_name, asname: Some(asname) } = ast.kind(alias) else {
                continue;
            };
            if !module_name.starts_with("lib.") {
                continue;
            }
            let mut name = asname.as_str();
            // --strict compilation suffixes every global binding
            if let Some(stripped) = name.strip_suffix("__global__") {
                name = stripped;
            }
            // A builtin-named alias is emitted under its canonical image (``import ...
            // as ta`` compiles to ``as ta__ren__``); stripping the suffix recovers it
            if let Some(stripped) = name.strip_suffix("__ren__") {
                name = stripped;
            }
            let namespace = namespace_rename(name);
            if !ctx.host.has_module_properties(&format!("lib.{namespace}"))? {
                continue;
            }
            if !ctx.host.import_library(module_name)? {
                continue;
            }
            aliases.insert(asname.clone(), (module_name.clone(), namespace.to_string()));
        }
    }
    if aliases.is_empty() {
        return Ok(module);
    }
    let mut shadow = BuiltinShadow { host: ctx.host, aliases, error: None };
    generic_visit(&mut shadow, ast, module);
    match shadow.error {
        Some(err) => Err(err.into()),
        None => Ok(module),
    }
}
