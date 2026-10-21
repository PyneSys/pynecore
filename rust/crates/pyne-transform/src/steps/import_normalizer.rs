//! Port of `transformers/import_normalizer.py`: normalize `pynecore.lib` imports to
//! `from pynecore import lib`, rewrite every reference to its fully qualified
//! `lib.x.y` spelling and import the submodules those spellings need.

use std::collections::{BTreeSet, HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, copy_location, generic_visit, walk};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Host, Unsupported};

/// Attributes of `lib` that are classes, not submodules
const NON_MODULE_ATTRS: [&str; 2] = ["input", "script"];

fn is_lib_module(module: &str) -> bool {
    module == "pynecore.lib" || module.starts_with("pynecore.lib.")
}

fn is_lib_import(ast: &Ast, id: NodeId) -> bool {
    matches!(ast.kind(id), Kind::ImportFrom { module: Some(module), .. } if is_lib_module(module))
}

fn is_lib_module_import(ast: &Ast, id: NodeId) -> bool {
    match ast.kind(id) {
        Kind::Import { names } => {
            names.iter().any(|&alias| is_lib_module(alias_name(ast, alias).0))
        }
        _ => false,
    }
}

fn alias_name(ast: &Ast, id: NodeId) -> (&str, Option<&str>) {
    let Kind::Alias { name, asname } = ast.kind(id) else { unreachable!("not an alias") };
    (name, asname.as_deref())
}

/// Whether a statement is `from pynecore import lib` (among other names).
fn imports_lib(ast: &Ast, id: NodeId) -> bool {
    match ast.kind(id) {
        Kind::ImportFrom { module: Some(module), names, .. } if module == "pynecore" => {
            names.iter().any(|&alias| alias_name(ast, alias).0 == "lib")
        }
        _ => false,
    }
}

fn full_path(module: &str, name: &str) -> Vec<String> {
    if module == "pynecore.lib" {
        return vec!["lib".into(), name.into()];
    }
    let mut path = vec!["lib".to_string()];
    path.extend(module.split('.').skip(2).map(str::to_string));
    path.push(name.into());
    path
}

/// `a.b.c` for a path, every node at the position of the name it replaces.
fn dotted(ast: &mut Ast, path: &[String], node: NodeId) -> NodeId {
    let mut result = ast.add(Kind::Name { id: path[0].clone(), ctx: ExprContext::Load });
    copy_location(ast, result, node);
    for part in &path[1..] {
        result =
            ast.add(Kind::Attribute { value: result, attr: part.clone(), ctx: ExprContext::Load });
        copy_location(ast, result, node);
    }
    result
}

/// The names of every `Store`-context `Name` in a statement that is no import or definition.
fn stored_names(ast: &Ast, stmt: NodeId) -> HashSet<String> {
    match ast.kind(stmt) {
        Kind::ImportFrom { .. }
        | Kind::Import { .. }
        | Kind::FunctionDef { .. }
        | Kind::AsyncFunctionDef { .. }
        | Kind::ClassDef { .. } => HashSet::new(),
        _ => walk(ast, stmt)
            .into_iter()
            .filter_map(|node| match ast.kind(node) {
                Kind::Name { id, ctx: ExprContext::Store } => Some(id.clone()),
                _ => None,
            })
            .collect(),
    }
}

/// Whether a top-level statement binds `name` to something other than a lib module.
fn rebinds(ast: &Ast, stmt: NodeId, name: &str, stored: &HashSet<String>) -> bool {
    match ast.kind(stmt) {
        Kind::ImportFrom { names, .. } => names.iter().any(|&alias| {
            let (alias_name, asname) = alias_name(ast, alias);
            asname.unwrap_or(alias_name) == name
        }),
        Kind::Import { names } => names.iter().any(|&alias| {
            let (alias_name, asname) = alias_name(ast, alias);
            asname.unwrap_or_else(|| alias_name.split('.').next().unwrap()) == name
                && !alias_name.starts_with("pynecore.lib")
        }),
        Kind::FunctionDef { name: def, .. }
        | Kind::AsyncFunctionDef { name: def, .. }
        | Kind::ClassDef { name: def, .. } => def == name,
        _ => stored.contains(name),
    }
}

struct ImportNormalizer<'h> {
    host: &'h dyn Host,
    /// imported name -> full path
    import_map: HashMap<String, Vec<String>>,
    names_to_replace: HashSet<String>,
    needs_lib_import: bool,
    /// A lib-module alias's bindings in source order: (top-level statement index,
    /// lib path), or `None` for a statement rebinding the name to something else
    alias_bindings: HashMap<String, Vec<(usize, Option<String>)>>,
    stmt_index: usize,
    /// alias -> module path
    module_imports: HashMap<String, String>,
    /// module path -> exposed names, in insertion order
    wildcard_imports: Vec<(String, HashSet<String>)>,
    required_submodules: BTreeSet<String>,
    function_parameters: HashSet<String>,
    /// The parameters of the function being entered, whose defaults were visited already
    defaults_visited: Option<NodeId>,
}

impl ImportNormalizer<'_> {
    fn handle_wildcard_import(&mut self, module: &str) -> Result<(), Unsupported> {
        let exposed: HashSet<String> =
            self.host.star_exports(module)?.unwrap_or_default().into_iter().collect();
        if exposed.is_empty() {
            // The Python step raises the SyntaxError
            return Err(Unsupported(format!("wildcard import of {module} without __all__")));
        }
        let mut path: Vec<String> = vec!["lib".into()];
        path.extend(module.split('.').skip(2).map(str::to_string));
        for name in &exposed {
            let mut full = path.clone();
            full.push(name.clone());
            self.import_map.insert(name.clone(), full);
            self.names_to_replace.insert(name.clone());
        }
        // A dict assignment: a module seen again keeps its place
        match self.wildcard_imports.iter_mut().find(|(existing, _)| existing == module) {
            Some(entry) => entry.1 = exposed,
            None => self.wildcard_imports.push((module.to_string(), exposed)),
        }
        Ok(())
    }

    fn handle_import_from(&mut self, ast: &Ast, stmt: NodeId) -> Result<(), Unsupported> {
        if !is_lib_import(ast, stmt) {
            return Ok(());
        }
        self.needs_lib_import = true;
        let Kind::ImportFrom { module, names, .. } = ast.kind(stmt) else { unreachable!() };
        let module = module.clone().unwrap_or_default();
        if module.starts_with("pynecore.lib.") {
            self.required_submodules.insert(module.split('.').nth(2).unwrap().to_string());
        }
        if names.iter().any(|&alias| alias_name(ast, alias).0 == "*") {
            return self.handle_wildcard_import(&module);
        }
        for &alias in names {
            let (name, asname) = alias_name(ast, alias);
            let bound = asname.unwrap_or(name).to_string();
            self.import_map.insert(bound.clone(), full_path(&module, name));
            self.names_to_replace.insert(bound);
        }
        Ok(())
    }

    fn handle_import(&mut self, ast: &Ast, stmt: NodeId) {
        if !is_lib_module_import(ast, stmt) {
            return;
        }
        self.needs_lib_import = true;
        let Kind::Import { names } = ast.kind(stmt) else { unreachable!() };
        for &alias in names {
            let (name, asname) = alias_name(ast, alias);
            if !is_lib_module(name) {
                continue;
            }
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() <= 2 {
                self.module_imports.insert(asname.unwrap_or("lib").to_string(), "lib".into());
            } else {
                self.required_submodules.insert(parts[2].to_string());
                let bound = asname.unwrap_or(parts[parts.len() - 1]).to_string();
                self.module_imports.insert(bound, format!("lib.{}", parts[2..].join(".")));
            }
        }
    }

    fn visit_name(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        let Kind::Name { id: name, ctx } = ast.kind(id) else { unreachable!() };
        if *ctx != ExprContext::Load || self.function_parameters.contains(name) {
            return id;
        }
        let name = name.clone();
        if self.names_to_replace.contains(&name) {
            let path = self.import_map[&name].clone();
            return dotted(ast, &path, id);
        }
        if let Some(target) = self.module_imports.get(&name) {
            let mut target = Some(target.clone());
            for (index, bound) in self.alias_bindings.get(&name).into_iter().flatten() {
                if *index > self.stmt_index {
                    break;
                }
                target = bound.clone();
            }
            return match target {
                Some(target) => {
                    let path: Vec<String> = target.split('.').map(str::to_string).collect();
                    dotted(ast, &path, id)
                }
                None => id,
            };
        }
        for (module, exposed) in &self.wildcard_imports {
            if exposed.contains(&name) {
                let mut path: Vec<String> = vec!["lib".into()];
                path.extend(module.split('.').skip(2).map(str::to_string));
                path.push(name.clone());
                return dotted(ast, &path, id);
            }
        }
        id
    }

    fn visit_attribute(&mut self, ast: &mut Ast, id: NodeId) {
        generic_visit(self, ast, id);
        let mut parts = Vec::new();
        let mut current = id;
        while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
            parts.push(attr.clone());
            current = *value;
        }
        if matches!(ast.kind(current), Kind::Name { id, .. } if id == "lib") && !parts.is_empty() {
            self.needs_lib_import = true;
            if parts.len() >= 2 {
                let module_name = parts.last().unwrap();
                if !NON_MODULE_ATTRS.contains(&module_name.as_str()) {
                    self.required_submodules.insert(module_name.clone());
                }
            }
        }
    }

    fn visit_function_def(&mut self, ast: &mut Ast, id: NodeId) {
        let old_parameters = self.function_parameters.clone();
        let Kind::FunctionDef { args, .. } = *ast.kind(id) else { unreachable!() };
        // Parameter defaults are evaluated in the ENCLOSING scope: visit them before
        // the parameter names enter ``function_parameters``
        let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(args).clone() else {
            unreachable!()
        };
        let defaults: Vec<NodeId> =
            defaults.into_iter().map(|default| self.visit_expr(ast, default)).collect();
        let kw_defaults: Vec<Option<NodeId>> = kw_defaults
            .into_iter()
            .map(|default| default.map(|default| self.visit_expr(ast, default)))
            .collect();
        let Kind::Arguments {
            defaults: d,
            kw_defaults: k,
            args: positional,
            posonlyargs,
            kwonlyargs,
            vararg,
            kwarg,
        } = ast.kind_mut(args)
        else {
            unreachable!()
        };
        *d = defaults;
        *k = kw_defaults;
        let params: Vec<NodeId> = positional
            .iter()
            .chain(posonlyargs.iter())
            .chain(kwonlyargs.iter())
            .chain(vararg.iter())
            .chain(kwarg.iter())
            .copied()
            .collect();
        for param in params {
            let Kind::Arg { arg, .. } = ast.kind(param) else { unreachable!() };
            self.function_parameters.insert(arg.clone());
        }
        self.defaults_visited = Some(args);
        generic_visit(self, ast, id);
        self.function_parameters = old_parameters;
    }

    fn visit_arguments(&mut self, ast: &mut Ast, id: NodeId) {
        if self.defaults_visited != Some(id) {
            generic_visit(self, ast, id);
            return;
        }
        self.defaults_visited = None;
        let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
            ast.kind(id).clone()
        else {
            unreachable!()
        };
        for arg in posonlyargs.into_iter().chain(args).chain(vararg).chain(kwonlyargs).chain(kwarg)
        {
            self.visit(ast, arg);
        }
    }

    fn visit_expr(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        match self.visit(ast, id) {
            Visited::Node(new) => new,
            other => panic!("an expression visit answered {other:?}"),
        }
    }
}

impl Transformer for ImportNormalizer<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::Name { .. } => return Visited::Node(self.visit_name(ast, id)),
            Kind::Attribute { .. } => self.visit_attribute(ast, id),
            Kind::FunctionDef { .. } => self.visit_function_def(ast, id),
            Kind::Arguments { .. } => self.visit_arguments(ast, id),
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Unsupported> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Unsupported("ImportNormalizer on a non-module".into()));
    };
    let body = body.clone();
    let mut normalizer = ImportNormalizer {
        host: ctx.host,
        import_map: HashMap::new(),
        names_to_replace: HashSet::new(),
        needs_lib_import: false,
        alias_bindings: HashMap::new(),
        stmt_index: 0,
        module_imports: HashMap::new(),
        wildcard_imports: Vec::new(),
        required_submodules: BTreeSet::new(),
        function_parameters: HashSet::new(),
        defaults_visited: None,
    };

    // First collect all imports
    let mut has_lib_import = false;
    for &stmt in &body {
        if imports_lib(ast, stmt) {
            has_lib_import = true;
        }
        match ast.kind(stmt) {
            Kind::ImportFrom { module: Some(name), names, .. } => {
                if name == "pynecore"
                    && names.iter().any(|&alias| {
                        let (alias, asname) = alias_name(ast, alias);
                        alias == "lib" && asname.is_some()
                    })
                {
                    // The Python step raises the SyntaxError
                    return Err(Unsupported("'lib' imported as an alias".into()));
                }
                normalizer.handle_import_from(ast, stmt)?;
            }
            Kind::ImportFrom { .. } => normalizer.handle_import_from(ast, stmt)?,
            Kind::Import { .. } => normalizer.handle_import(ast, stmt),
            _ => {}
        }
    }

    // A lib-module alias rebound later in the module denotes that binding from there on
    for (index, &stmt) in body.iter().enumerate() {
        if let Kind::Import { names } = ast.kind(stmt) {
            for &alias in names {
                let (name, asname) = alias_name(ast, alias);
                if is_lib_module(name) {
                    let parts: Vec<&str> = name.split('.').collect();
                    let target = if parts.len() <= 2 {
                        "lib".to_string()
                    } else {
                        format!("lib.{}", parts[2..].join("."))
                    };
                    let bound = asname.unwrap_or(parts[parts.len() - 1]).to_string();
                    normalizer.alias_bindings.entry(bound).or_default().push((index, Some(target)));
                }
            }
        }
        if normalizer.module_imports.is_empty() {
            continue;
        }
        let stored = stored_names(ast, stmt);
        let names: Vec<String> = normalizer.module_imports.keys().cloned().collect();
        for name in names {
            if rebinds(ast, stmt, &name, &stored) {
                normalizer.alias_bindings.entry(name).or_default().push((index, None));
            }
        }
    }

    // Process the rest of the module to collect attribute usages
    let mut visited_body = Vec::with_capacity(body.len());
    for (index, stmt) in body.into_iter().enumerate() {
        normalizer.stmt_index = index;
        match normalizer.visit(ast, stmt) {
            Visited::Node(new) => visited_body.push(new),
            other => return Err(Unsupported(format!("a statement visit answered {other:?}"))),
        }
    }

    // Filter out old lib imports
    let mut new_body: Vec<NodeId> = visited_body
        .into_iter()
        .filter(|&stmt| match ast.kind(stmt) {
            Kind::ImportFrom { .. } => !is_lib_import(ast, stmt) || imports_lib(ast, stmt),
            Kind::Import { .. } => !is_lib_module_import(ast, stmt),
            _ => true,
        })
        .collect();

    let mut imports = Vec::new();
    if normalizer.needs_lib_import && !has_lib_import {
        let alias = ast.add(Kind::Alias { name: "lib".into(), asname: None });
        imports.push(ast.add(Kind::ImportFrom {
            module: Some("pynecore".into()),
            names: vec![alias],
            level: Some(0),
        }));
    }
    for submodule in &normalizer.required_submodules {
        let alias =
            ast.add(Kind::Alias { name: format!("pynecore.lib.{submodule}"), asname: None });
        imports.push(ast.add(Kind::Import { names: vec![alias] }));
    }

    // Insert imports after the docstring and the ``__future__`` block
    let mut position = usize::from(new_body.first().is_some_and(|&first| match ast.kind(first) {
        Kind::Expr { value } => matches!(ast.kind(*value), Kind::Constant { .. }),
        _ => false,
    }));
    while position < new_body.len()
        && matches!(ast.kind(new_body[position]), Kind::ImportFrom { module: Some(m), .. } if m == "__future__")
    {
        position += 1;
    }
    new_body.splice(position..position, imports);
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    *body = new_body;
    Ok(module)
}
