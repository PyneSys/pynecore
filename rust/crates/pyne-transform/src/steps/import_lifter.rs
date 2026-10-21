//! Port of `transformers/import_lifter.py`: lift the `pynecore.lib` imports standing
//! directly in a function body to module level.

use pyne_ast::walk::{Transformer, Visited, generic_visit, walk_statements};
use pyne_ast::{Ast, Kind, NodeId};

use crate::{Context, Unsupported};

fn is_lib_import(ast: &Ast, id: NodeId) -> bool {
    match ast.kind(id) {
        Kind::ImportFrom { module: Some(module), .. } => {
            module == "pynecore.lib" || module.starts_with("pynecore.lib.")
        }
        _ => false,
    }
}

fn imports_equal(ast: &Ast, first: NodeId, second: NodeId) -> bool {
    let (
        Kind::ImportFrom { module: module1, names: names1, .. },
        Kind::ImportFrom { module: module2, names: names2, .. },
    ) = (ast.kind(first), ast.kind(second))
    else {
        return false;
    };
    module1 == module2
        && names1.len() == names2.len()
        && names1.iter().zip(names2).all(|(n1, n2)| match (ast.kind(*n1), ast.kind(*n2)) {
            (
                Kind::Alias { name: name1, asname: asname1 },
                Kind::Alias { name: name2, asname: asname2 },
            ) => name1 == name2 && asname1 == asname2,
            _ => false,
        })
}

struct ImportLifter {
    lifted: Vec<NodeId>,
}

impl Transformer for ImportLifter {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        generic_visit(self, ast, id);
        if let Kind::FunctionDef { body, .. } = ast.kind(id) {
            let body = body.clone();
            let mut new_body = Vec::with_capacity(body.len());
            for stmt in body {
                if is_lib_import(ast, stmt) {
                    if !self.lifted.iter().any(|&lifted| imports_equal(ast, stmt, lifted)) {
                        self.lifted.push(stmt);
                    }
                } else {
                    new_body.push(stmt);
                }
            }
            let Kind::FunctionDef { body, .. } = ast.kind_mut(id) else { unreachable!() };
            *body = new_body;
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Unsupported> {
    // Only a lib import standing directly in a function body is lifted: without
    // one there is nothing to visit
    let candidate = walk_statements(ast, module).into_iter().any(|stmt| match ast.kind(stmt) {
        Kind::FunctionDef { body, .. } => body.iter().any(|&inner| is_lib_import(ast, inner)),
        _ => false,
    });
    if !candidate {
        return Ok(module);
    }
    let mut lifter = ImportLifter { lifted: Vec::new() };
    generic_visit(&mut lifter, ast, module);
    if lifter.lifted.is_empty() {
        return Ok(module);
    }
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Unsupported("ImportLifter on a non-module".into()));
    };
    // Insert the lifted imports after the docstring, if there is one
    let docstring = body.first().is_some_and(|&first| match ast.kind(first) {
        Kind::Expr { value } => matches!(ast.kind(*value), Kind::Constant { .. }),
        _ => false,
    });
    let position = usize::from(docstring);
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.splice(position..position, lifter.lifted);
    Ok(module)
}
