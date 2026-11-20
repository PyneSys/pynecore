//! Splitting the locals of one function into the variables they spell (`_Segments`).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::walk::{children, walk};
use pyne_ast::{Ast, Kind, NodeId};

use super::facts::{
    arg_name, args_of, def_name, scope_header, spells_persistent, spells_series, statement_lists,
    unchecked,
};
use crate::pine_type_rules::{UNKNOWN, elements_of, is_tuple, join, shape_conflict};

/// Separator between a local's name and its segment number in a binding key
/// (`SEGMENT_SEP`).
pub const SEGMENT_SEP: char = '@';

/// The name a binding key is spelled with in the source (`_plain_name`).
pub fn plain_name(key: &str) -> &str {
    key.split(SEGMENT_SEP).next().unwrap_or(key)
}

/// The key one segment of a name is bound under (`_segment_key`).
fn segment_key(name: &str, index: usize) -> String {
    if index == 0 { name.to_string() } else { format!("{name}{SEGMENT_SEP}{}", index + 1) }
}

/// Whether two types are ones no single variable holds (`_conflicting`).
pub fn conflicting(left: &str, right: &str) -> bool {
    if is_tuple(left) && is_tuple(right) {
        let (items, others) = (elements_of(left), elements_of(right));
        if items.len() == others.len() {
            return items.iter().zip(&others).any(|(item, other)| conflicting(item, other));
        }
    }
    if shape_conflict(left, right).is_some() {
        return true;
    }
    join(left, right) == UNKNOWN && !unchecked(left) && !unchecked(right)
}

/// What the scan knows about one local name (`_SegmentedName`).
#[derive(Default)]
struct SegmentedName {
    block: Option<usize>,
    current: usize,
    nodes: Vec<(NodeId, usize)>,
    parents: Vec<Option<usize>>,
}

/// The scan of one definition (`_Segments`).
pub struct Segments<'a> {
    ast: &'a Ast,
    excluded: HashSet<String>,
    names: IndexMap<String, SegmentedName>,
    blocks: usize,
    opened: HashMap<usize, Vec<String>>,
    /// The nested scopes the body holds: what each reads from outside itself is
    /// not split, which the caller settles with `exclude`
    pub scopes: Vec<NodeId>,
}

impl<'a> Segments<'a> {
    pub fn new(ast: &'a Ast, node: NodeId, body: &[NodeId]) -> Segments<'a> {
        let mut scan = Segments {
            ast,
            excluded: HashSet::new(),
            names: IndexMap::new(),
            blocks: 0,
            opened: HashMap::new(),
            scopes: Vec::new(),
        };
        if let Some(args) = args_of(ast, node) {
            for &arg in args.posonlyargs.iter().chain(args.args).chain(args.kwonlyargs) {
                scan.excluded.insert(arg_name(ast, arg));
            }
            for arg in [args.vararg, args.kwarg].into_iter().flatten() {
                scan.excluded.insert(arg_name(ast, arg));
            }
        }
        scan.body(body);
        scan
    }

    /// Take names out of the split.
    pub fn exclude<'n>(&mut self, names: impl IntoIterator<Item = &'n String>) {
        self.excluded.extend(names.into_iter().cloned());
    }

    /// The binding keys of the names that split (`result`): name node -> its key
    /// for every occurrence outside the name's first segment, and key -> the key
    /// of the segment it replaced.
    pub fn result(self) -> (HashMap<NodeId, String>, HashMap<String, String>) {
        let mut keys = HashMap::new();
        let mut parents = HashMap::new();
        for (name, found) in &self.names {
            if self.excluded.contains(name) || found.parents.len() < 2 {
                continue;
            }
            for &(node, index) in &found.nodes {
                if index != 0 {
                    keys.insert(node, segment_key(name, index));
                }
            }
            for (index, parent) in found.parents.iter().enumerate() {
                if let Some(parent) = parent {
                    parents.insert(segment_key(name, index), segment_key(name, *parent));
                }
            }
        }
        (keys, parents)
    }

    fn name_of(&self, node: NodeId) -> Option<String> {
        match self.ast.kind(node) {
            Kind::Name { id, .. } => Some(id.clone()),
            _ => None,
        }
    }

    fn plain(&mut self, node: NodeId) {
        let Some(name) = self.name_of(node) else { return };
        if self.excluded.contains(&name) {
            return;
        }
        match self.names.get_mut(&name) {
            Some(found) if found.block.is_some() => found.nodes.push((node, found.current)),
            _ => {
                self.excluded.insert(name);
            }
        }
    }

    fn declare(&mut self, node: NodeId, block: usize) {
        let Some(name) = self.name_of(node) else { return };
        if self.excluded.contains(&name) {
            return;
        }
        let found = self.names.entry(name.clone()).or_default();
        if found.block.is_some_and(|open| open != block) {
            found.nodes.push((node, found.current));
            return;
        }
        if found.block.is_none() {
            self.opened.entry(block).or_default().push(name);
            found.parents.push(None);
        } else {
            found.parents.push(Some(found.current));
        }
        found.block = Some(block);
        found.current = found.parents.len() - 1;
        found.nodes.push((node, found.current));
    }

    fn exclude_under(&mut self, node: NodeId) {
        for sub in walk(self.ast, node) {
            if let Kind::Name { id, .. } = self.ast.kind(sub) {
                self.excluded.insert(id.clone());
            }
        }
    }

    fn expr(&mut self, node: NodeId) {
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            match self.ast.kind(current) {
                Kind::Name { .. } => self.plain(current),
                Kind::FunctionDef { .. }
                | Kind::AsyncFunctionDef { .. }
                | Kind::Lambda { .. }
                | Kind::ClassDef { .. } => self.scope(current),
                Kind::ListComp { .. }
                | Kind::SetComp { .. }
                | Kind::DictComp { .. }
                | Kind::GeneratorExp { .. } => self.exclude_under(current),
                _ => stack.extend(children(self.ast, current).into_iter().rev()),
            }
        }
    }

    fn scope(&mut self, node: NodeId) {
        for header in scope_header(self.ast, node) {
            self.exclude_under(header);
        }
        self.scopes.push(node);
        if !matches!(self.ast.kind(node), Kind::Lambda { .. }) {
            self.excluded.insert(def_name(self.ast, node).to_string());
        }
    }

    fn target_names(&self, target: NodeId, out: &mut Vec<NodeId>) -> bool {
        match self.ast.kind(target) {
            Kind::Name { .. } => {
                out.push(target);
                true
            }
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                elts.iter().all(|&element| self.target_names(element, out))
            }
            _ => false,
        }
    }

    fn store(&mut self, target: NodeId, value: NodeId, block: usize) {
        let mut names = Vec::new();
        if !self.target_names(target, &mut names) {
            self.expr(target);
            return;
        }
        let read: HashSet<String> =
            walk(self.ast, value).into_iter().filter_map(|sub| self.name_of(sub)).collect();
        let mut seen: HashSet<String> = HashSet::new();
        for name in names {
            let Some(id) = self.name_of(name) else { continue };
            if read.contains(&id) || seen.contains(&id) {
                self.plain(name);
            } else {
                self.declare(name, block);
            }
            seen.insert(id);
        }
    }

    fn block(&mut self, stmts: &[NodeId], block: usize) {
        for &stmt in stmts {
            self.stmt(stmt, block);
        }
        for name in self.opened.remove(&block).unwrap_or_default() {
            if let Some(found) = self.names.get_mut(&name) {
                found.block = None;
            }
        }
    }

    fn new_block(&mut self) -> usize {
        self.blocks += 1;
        self.blocks
    }

    fn body(&mut self, stmts: &[NodeId]) {
        let block = self.new_block();
        self.block(stmts, block);
    }

    fn stmt(&mut self, stmt: NodeId, block: usize) {
        match self.ast.kind(stmt).clone() {
            Kind::Assign { targets, value, .. } => {
                self.expr(value);
                if targets.len() == 1 {
                    self.store(targets[0], value, block);
                } else {
                    for target in targets {
                        self.expr(target);
                    }
                }
            }
            Kind::AnnAssign { target, annotation, value, .. } => {
                self.expr(annotation);
                if let Some(value) = value {
                    self.expr(value);
                }
                match value {
                    Some(value)
                        if matches!(self.ast.kind(target), Kind::Name { .. })
                            && !spells_persistent(self.ast, Some(annotation))
                            && !spells_series(self.ast, Some(annotation)) =>
                    {
                        self.store(target, value, block);
                    }
                    _ => self.exclude_under(target),
                }
            }
            Kind::For { target, iter, body, orelse, .. }
            | Kind::AsyncFor { target, iter, body, orelse, .. } => {
                self.expr(iter);
                let inner = self.new_block();
                self.store(target, iter, inner);
                self.block(&body, inner);
                self.body(&orelse);
            }
            Kind::While { test, body, orelse } => {
                self.expr(test);
                self.body(&body);
                self.body(&orelse);
            }
            Kind::If { test, body, orelse } => {
                self.expr(test);
                self.body(&body);
                self.body(&orelse);
            }
            Kind::With { items, body, .. } | Kind::AsyncWith { items, body, .. } => {
                for item in items {
                    if let Kind::WithItem { context_expr, optional_vars } = self.ast.kind(item) {
                        let (context, vars) = (*context_expr, *optional_vars);
                        self.expr(context);
                        if let Some(vars) = vars {
                            self.expr(vars);
                        }
                    }
                }
                self.body(&body);
            }
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::ClassDef { .. } => {
                self.scope(stmt);
            }
            Kind::Import { names, .. } | Kind::ImportFrom { names, .. } => {
                for alias in names {
                    if let Kind::Alias { name, asname } = self.ast.kind(alias) {
                        let bound = asname.as_deref().unwrap_or(name);
                        self.excluded.insert(bound.split('.').next().unwrap_or("").to_string());
                    }
                }
            }
            Kind::Global { names } | Kind::Nonlocal { names } => self.excluded.extend(names),
            kind if matches!(kind, Kind::Delete { .. })
                || !statement_lists(self.ast, stmt).is_empty() =>
            {
                for sub in walk(self.ast, stmt) {
                    match self.ast.kind(sub) {
                        Kind::Name { id, .. } => {
                            self.excluded.insert(id.clone());
                        }
                        Kind::FunctionDef { .. }
                        | Kind::AsyncFunctionDef { .. }
                        | Kind::Lambda { .. }
                        | Kind::ClassDef { .. } => self.scope(sub),
                        Kind::ExceptHandler { name: Some(name), .. }
                        | Kind::MatchAs { name: Some(name), .. }
                        | Kind::MatchStar { name: Some(name) } => {
                            self.excluded.insert(name.clone());
                        }
                        Kind::MatchMapping { rest: Some(rest), .. } => {
                            self.excluded.insert(rest.clone());
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                for child in children(self.ast, stmt) {
                    self.expr(child);
                }
            }
        }
    }
}
