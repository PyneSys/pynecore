//! Port of `transformers/unused_series_detector.py`: a `Series[T]` annotation on a
//! variable or parameter that is never indexed becomes plain `T`.
//!
//! Scopes are named as the Python pass names them: a module-level function by its
//! name, a nested one by its dotted path, the module itself `__module__`. An index
//! of a name inside a nested function is credited to the nearest enclosing scope
//! that declares it as a Series, as the series pass resolves it.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::children;
use pyne_ast::{Ast, ExprContext, FieldRef, Kind, NodeId};

use crate::{Context, Error};

const MODULE_SCOPE: &str = "__module__";

type Scopes = HashMap<String, HashSet<String>>;

/// The scope of a function defined in `current`.
fn nested_scope(current: &str, name: &str) -> String {
    if current == MODULE_SCOPE { name.to_string() } else { format!("{current}.{name}") }
}

/// Whether an annotation is `Series[T]` or `<mod>.Series[T]`.
fn is_series_annotation(ast: &Ast, annotation: NodeId) -> bool {
    let Kind::Subscript { value, .. } = ast.kind(annotation) else { return false };
    match ast.kind(*value) {
        Kind::Name { id, .. } => id == "Series",
        Kind::Attribute { attr, .. } => attr == "Series",
        _ => false,
    }
}

/// The variables a module declares as Series and the ones it indexes, per scope.
#[derive(Default)]
struct Detector {
    series_vars: Scopes,
    indexed_vars: Scopes,
    scope: Vec<String>,
}

impl Detector {
    fn current(&self) -> &str {
        self.scope.last().map_or(MODULE_SCOPE, String::as_str)
    }

    fn declare(&mut self, name: &str) {
        let scope = self.current().to_string();
        self.series_vars.entry(scope).or_default().insert(name.to_string());
    }

    fn visit(&mut self, ast: &Ast, id: NodeId) {
        match ast.kind(id) {
            Kind::FunctionDef { name, .. } => {
                let scope = nested_scope(self.current(), name);
                self.scope.push(scope);
                self.generic_visit(ast, id);
                self.scope.pop();
            }
            Kind::AnnAssign { target, annotation, .. } => {
                if let Kind::Name { id: name, .. } = ast.kind(*target)
                    && is_series_annotation(ast, *annotation)
                {
                    self.declare(name);
                }
                self.generic_visit(ast, id);
            }
            Kind::Arg { arg, annotation, .. } => {
                if annotation.is_some_and(|annotation| is_series_annotation(ast, annotation)) {
                    self.declare(arg);
                }
            }
            Kind::Subscript { value, .. } => {
                // `Series[...]` is an annotation, not an index
                if let Kind::Name { id: name, .. } = ast.kind(*value)
                    && name != "Series"
                {
                    let scope = self.current().to_string();
                    self.indexed_vars.entry(scope).or_default().insert(name.clone());
                }
                self.generic_visit(ast, id);
            }
            _ => self.generic_visit(ast, id),
        }
    }

    fn generic_visit(&mut self, ast: &Ast, id: NodeId) {
        for child in children(ast, id) {
            self.visit(ast, child);
        }
    }

    /// The nearest scope, `scope` itself included, that declares `name` as a Series.
    fn series_owner(&self, scope: &str, name: &str) -> Option<String> {
        let declares = |candidate: &str| {
            self.series_vars.get(candidate).is_some_and(|vars| vars.contains(name))
        };
        let parts: Vec<&str> = scope.split('.').collect();
        for i in (1..=parts.len()).rev() {
            let candidate = parts[..i].join(".");
            if declares(&candidate) {
                return Some(candidate);
            }
        }
        declares(MODULE_SCOPE).then(|| MODULE_SCOPE.to_string())
    }
}

/// The second pass: drops the Series of every declaration its scope never indexes.
struct Optimizer<'a> {
    series_vars: &'a Scopes,
    indexed_vars: &'a Scopes,
    scope: Vec<String>,
}

impl Optimizer<'_> {
    fn current(&self) -> &str {
        self.scope.last().map_or(MODULE_SCOPE, String::as_str)
    }

    fn is_unused(&self, name: &str) -> bool {
        let scope = self.current();
        let declared = self.series_vars.get(scope).is_some_and(|vars| vars.contains(name));
        let indexed = self.indexed_vars.get(scope).is_some_and(|vars| vars.contains(name));
        declared && !indexed
    }

    /// `T` of `Series[T]`: the subscript's slice when it is a name, else the slice's
    /// `value`, else a fresh `float`.
    fn inner_type(ast: &mut Ast, annotation: NodeId) -> Result<NodeId, Error> {
        let Kind::Subscript { slice, .. } = ast.kind(annotation) else {
            return Ok(ast.add(Kind::Name { id: "float".into(), ctx: ExprContext::Load }));
        };
        let slice = *slice;
        let kind = ast.kind(slice);
        if matches!(kind, Kind::Name { .. }) {
            return Ok(slice);
        }
        let field = kind.class().fields.iter().position(|field| field.name == "value");
        match field.map(|index| kind.field(index)) {
            Some(FieldRef::Node(value)) | Some(FieldRef::OptNode(Some(value))) => Ok(*value),
            Some(_) => Err(Error::Unsupported(format!(
                "a Series annotation whose {} slice has a value that is no node",
                ast.name(slice)
            ))),
            None if ast.extra(slice, "value").is_some() => Err(Error::Unsupported(
                "a Series annotation whose slice carries a stamped value".into(),
            )),
            None => Ok(ast.add(Kind::Name { id: "float".into(), ctx: ExprContext::Load })),
        }
    }

    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Result<(), Error> {
        match ast.kind(id) {
            Kind::FunctionDef { name, .. } => {
                let scope = nested_scope(self.current(), name);
                self.scope.push(scope);
                self.generic_visit(ast, id)?;
                self.scope.pop();
            }
            &Kind::AnnAssign { target, annotation, .. } => {
                if let Kind::Name { id: name, .. } = ast.kind(target)
                    && self.is_unused(name)
                    && is_series_annotation(ast, annotation)
                {
                    let inner = Self::inner_type(ast, annotation)?;
                    let Kind::AnnAssign { annotation, .. } = ast.kind_mut(id) else {
                        unreachable!()
                    };
                    *annotation = inner;
                }
                self.generic_visit(ast, id)?;
            }
            Kind::Arg { arg, annotation: Some(annotation), .. } => {
                let annotation = *annotation;
                if self.is_unused(arg) && is_series_annotation(ast, annotation) {
                    let inner = Self::inner_type(ast, annotation)?;
                    let Kind::Arg { annotation, .. } = ast.kind_mut(id) else { unreachable!() };
                    *annotation = Some(inner);
                }
            }
            Kind::Arg { .. } => {}
            _ => self.generic_visit(ast, id)?,
        }
        Ok(())
    }

    fn generic_visit(&mut self, ast: &mut Ast, id: NodeId) -> Result<(), Error> {
        for child in children(ast, id) {
            self.visit(ast, child)?;
        }
        Ok(())
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("UnusedSeriesDetector on a non-module".into()));
    }
    let mut detector = Detector::default();
    detector.visit(ast, module);

    // A parent-scope series indexed only from a nested function stays a series
    let mut credits: Vec<(String, String)> = Vec::new();
    for (scope, names) in &detector.indexed_vars {
        for name in names {
            if let Some(owner) = detector.series_owner(scope, name)
                && owner != *scope
            {
                credits.push((owner, name.clone()));
            }
        }
    }
    for (owner, name) in credits {
        detector.indexed_vars.entry(owner).or_default().insert(name);
    }

    let all_indexed = detector.series_vars.iter().all(|(scope, names)| {
        let indexed = detector.indexed_vars.get(scope);
        names.iter().all(|name| indexed.is_some_and(|vars| vars.contains(name)))
    });
    if all_indexed {
        return Ok(module);
    }
    let mut optimizer = Optimizer {
        series_vars: &detector.series_vars,
        indexed_vars: &detector.indexed_vars,
        scope: Vec::new(),
    };
    optimizer.visit(ast, module)?;
    Ok(module)
}
