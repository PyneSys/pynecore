//! `PineType` (`pine_type_transformer.py`, `pine_type_infer.py`,
//! `pine_type_report.py`): Pine's static types, stamped on the nodes, and the
//! module's type table.
//!
//! The port mirrors `_Inference` method for method, split by concern:
//!
//! - this module: the walker's state, the scope plumbing, the stamps, the entry;
//! - [`facts`]: what the walk reads off the tree without typing anything;
//! - [`collect`]: the imports, the classes and the definitions, before the walk;
//! - [`stmts`]: statements, stores and the loop fixpoint;
//! - [`defs`]: definitions and their per-call-site contexts;
//! - [`exprs`]: expressions and their qualifiers;
//! - [`lib_calls`]: calls into the lib registry, its override forms and the
//!   implicit conversions and pins they decide;
//! - [`user_calls`]: calls to the module's own functions, groups, constructors
//!   and methods;
//! - [`imports`]: calls into other modules, through the host;
//! - [`finish`]: the per-instance channel and the stamps written at the end;
//! - [`report`]: where the types ran out (`unknown_diags`).
//!
//! Every map the Python walker keys by a node's pre-order id (`_pine_nid`) is keyed
//! by that id here too; a map keyed by the node object is keyed by its
//! [`NodeId`]. Iteration orders that reach the table or a stamp follow Python's
//! dict insertion order ([`IndexMap`]).

mod collect;
mod defs;
mod exprs;
pub mod facts;
mod finish;
mod imports;
mod lib_calls;
mod report;
mod stmts;
mod user_calls;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use indexmap::{IndexMap, IndexSet};
use pyne_ast::walk::walk_statements;
use pyne_ast::{Ast, Kind, NodeId, Unsupported, Value};

use crate::node_ids::assign_node_ids;
use crate::pine_qualifier::{CONST, SERIES};
use crate::pine_type_rules::{
    Classes, FactoryFields, TY_ATTR, UNKNOWN, elements_of, is_tuple, join, render_ty, set_ty,
    shape_conflict,
};
use crate::pine_type_table::{
    Binding, CallSite, ClassSig, ContextKey, ContextResult, DepRecord, Diag, ExportSig, FuncSig,
    ModuleInterface, PineTypeTable, TABLE_ATTR, Unknown, qualify,
};
use crate::{Context, Error, Host};

use facts::{BranchIndex, BranchPath, Pos, ScopeFacts, nid, unchecked};

/// How many times a loop body is re-inferred before the types are declared stable
/// (`_MAX_LOOP_PASSES`).
const MAX_LOOP_PASSES: usize = 5;

/// Ceiling on how many per-call-site contexts one module may produce (`_MAX_CONTEXTS`).
const MAX_CONTEXTS: usize = 500;

type R<T> = Result<T, Unsupported>;

/// The enclosing-scope names a definition's body reads, each sorted: the lexically resolved
/// ones and the `global` ones.
type FreeLists = (Rc<Vec<String>>, Rc<Vec<String>>);

/// One live lexical scope of the walk (`_Frame`).
#[derive(Debug, Default)]
struct Frame {
    scope: String,
    names: IndexMap<String, Binding>,
    declared_global: HashSet<String>,
    declared_nonlocal: HashSet<String>,
    declared_series: HashSet<String>,
    local_names: HashSet<String>,
    loop_counters: HashMap<String, String>,
    na_stores: HashMap<String, Vec<(NodeId, Vec<usize>)>>,
}

impl Frame {
    fn new(scope: &str) -> Frame {
        Frame { scope: scope.to_string(), ..Frame::default() }
    }
}

/// A module-level name an import binds, and the module it reaches (`_Import`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Import {
    module: String,
    attrs: Vec<String>,
}

/// A name bound to a library merged over a builtin namespace (`_Shadowed`).
#[derive(Clone, Debug)]
struct Shadowed {
    source: Import,
    namespace: String,
}

/// One call site's analysis identity (`_anchors`' key).
type Anchor = (String, Vec<String>, i64, Option<i64>, Option<i64>);

/// The walker (`_Inference`).
struct Engine<'a, 'h> {
    ast: &'a mut Ast,
    host: &'h dyn Host,
    module_path: String,
    qualify_windows: bool,
    // --- the table being built ---
    funcs: IndexMap<String, FuncSig>,
    /// Memo key -> index into `results`; `table.contexts`
    contexts: IndexMap<ContextKey, usize>,
    results: Vec<ContextResult>,
    calls: Vec<CallSite>,
    diags: Vec<Diag>,
    bindings: IndexMap<String, IndexMap<String, Binding>>,
    deps: IndexMap<String, DepRecord>,
    class_sigs_table: IndexMap<String, ClassSig>,
    extensions: IndexMap<String, IndexMap<String, ExportSig>>,
    call_pos: IndexMap<i64, (i64, i64)>,
    // --- the walk ---
    frames: Vec<Frame>,
    stack: Vec<usize>,
    lib_aliases: HashSet<String>,
    lib_names: HashMap<NodeId, Option<String>>,
    imports: IndexMap<String, Import>,
    shadowed: IndexMap<String, Shadowed>,
    import_positions: HashMap<String, HashSet<Pos>>,
    multi_imports: HashSet<String>,
    module_paths: HashMap<String, Option<String>>,
    interfaces: HashMap<String, Option<Rc<ModuleInterface>>>,
    annotated_returns: HashMap<String, bool>,
    overload_groups: IndexSet<String>,
    group_returns: HashMap<String, Vec<String>>,
    group_slot: HashMap<i64, usize>,
    defs: IndexMap<String, Vec<NodeId>>,
    rebound: HashMap<String, HashSet<String>>,
    walked: HashSet<i64>,
    context_stack: Vec<i64>,
    next_context: i64,
    in_progress: HashSet<(String, Vec<String>, Option<i64>)>,
    free: HashMap<i64, FreeLists>,
    scope_free_memo: HashMap<NodeId, facts::FreeNames>,
    facts_memo: HashMap<NodeId, Rc<ScopeFacts>>,
    returns_memo: HashMap<NodeId, Rc<Vec<NodeId>>>,
    called_memo: HashMap<NodeId, Rc<HashSet<String>>>,
    anchors: HashMap<Anchor, ContextKey>,
    ty: HashMap<i64, String>,
    pins: HashMap<i64, IndexMap<i64, Option<String>>>,
    widens: IndexMap<i64, IndexMap<i64, Option<String>>>,
    widen_nodes: IndexMap<i64, NodeId>,
    pin_sink: Vec<Option<usize>>,
    pending: Vec<Vec<(String, NodeId)>>,
    signed: HashSet<String>,
    pin_diags: HashSet<i64>,
    unvectored: HashSet<i64>,
    call_ctx: HashMap<i64, IndexMap<i64, i64>>,
    callee_key: HashMap<i64, String>,
    module_rebinds: IndexMap<String, Vec<(Pos, BranchPath)>>,
    module_def_positions: HashMap<String, Vec<Pos>>,
    branches: BranchIndex,
    loop_stack: Vec<NodeId>,
    classes: IndexMap<String, String>,
    classes_lookup: Classes,
    class_spellings: HashMap<String, String>,
    type_imports: HashMap<String, String>,
    class_sigs: HashMap<String, Rc<ClassSig>>,
    class_methods: IndexMap<String, IndexMap<String, NodeId>>,
    /// Class id -> method name -> (its key in `funcs`, the scope the class stands in),
    /// for the methods declared in this module's class bodies (`_members`)
    members: IndexMap<String, IndexMap<String, (String, String)>>,
    shape_diags: HashSet<i64>,
    shape_diags_named: HashSet<(i64, String)>,
    factory_fields: HashSet<NodeId>,
    factory: FactoryFields,
    pins_suppressed: Option<Diag>,
    bad_calls: HashSet<NodeId>,
    sec_writes: HashMap<String, Vec<NodeId>>,
    q: HashMap<i64, u8>,
    conds: Vec<u8>,
    call_q: HashMap<i64, u8>,
    series_len: IndexMap<i64, IndexMap<i64, bool>>,
    window_nodes: HashMap<i64, NodeId>,
}

impl<'a, 'h> Engine<'a, 'h> {
    fn new(ast: &'a mut Ast, tree: NodeId, host: &'h dyn Host, module_path: String) -> Self {
        let factory = FactoryFields::new(ast, tree);
        let mut bindings = IndexMap::new();
        bindings.insert(String::new(), IndexMap::new());
        let class_sigs = collect::lib_classes()
            .iter()
            .map(|(cid, sig)| (cid.clone(), Rc::new(sig.clone())))
            .collect();
        Engine {
            ast,
            host,
            module_path,
            qualify_windows: false,
            funcs: IndexMap::new(),
            contexts: IndexMap::new(),
            results: Vec::new(),
            calls: Vec::new(),
            diags: Vec::new(),
            bindings,
            deps: IndexMap::new(),
            class_sigs_table: IndexMap::new(),
            extensions: IndexMap::new(),
            call_pos: IndexMap::new(),
            frames: vec![Frame::new("")],
            stack: vec![0],
            lib_aliases: HashSet::new(),
            lib_names: HashMap::new(),
            imports: IndexMap::new(),
            shadowed: IndexMap::new(),
            import_positions: HashMap::new(),
            multi_imports: HashSet::new(),
            module_paths: HashMap::new(),
            interfaces: HashMap::new(),
            annotated_returns: HashMap::new(),
            overload_groups: IndexSet::new(),
            group_returns: HashMap::new(),
            group_slot: HashMap::new(),
            defs: IndexMap::new(),
            rebound: HashMap::new(),
            walked: HashSet::new(),
            context_stack: vec![0],
            next_context: 1,
            in_progress: HashSet::new(),
            free: HashMap::new(),
            scope_free_memo: HashMap::new(),
            facts_memo: HashMap::new(),
            returns_memo: HashMap::new(),
            called_memo: HashMap::new(),
            anchors: HashMap::new(),
            ty: HashMap::new(),
            pins: HashMap::new(),
            widens: IndexMap::new(),
            widen_nodes: IndexMap::new(),
            pin_sink: vec![None],
            pending: vec![Vec::new()],
            signed: HashSet::new(),
            pin_diags: HashSet::new(),
            unvectored: HashSet::new(),
            call_ctx: HashMap::new(),
            callee_key: HashMap::new(),
            module_rebinds: IndexMap::new(),
            module_def_positions: HashMap::new(),
            branches: BranchIndex::default(),
            loop_stack: Vec::new(),
            classes: IndexMap::new(),
            classes_lookup: Classes::new(),
            class_spellings: HashMap::new(),
            type_imports: HashMap::new(),
            class_sigs,
            class_methods: IndexMap::new(),
            members: IndexMap::new(),
            shape_diags: HashSet::new(),
            shape_diags_named: HashSet::new(),
            factory_fields: HashSet::new(),
            factory,
            pins_suppressed: None,
            bad_calls: HashSet::new(),
            sec_writes: HashMap::new(),
            q: HashMap::new(),
            conds: Vec::new(),
            call_q: HashMap::new(),
            series_len: IndexMap::new(),
            window_nodes: HashMap::new(),
        }
    }

    // --- node reading ----------------------------------------------------

    fn kind(&self, node: NodeId) -> &Kind {
        self.ast.kind(node)
    }

    fn nid(&self, node: NodeId) -> Option<i64> {
        nid(self.ast, node)
    }

    fn line(&self, node: NodeId) -> i64 {
        facts::line(self.ast, node)
    }

    fn col(&self, node: NodeId) -> i64 {
        facts::col(self.ast, node)
    }

    fn is_expr(&self, node: NodeId) -> bool {
        self.ast.kind(node).class().category == pyne_ast::Category::Expr
    }

    fn dotted(&self, node: NodeId) -> Option<String> {
        crate::pine_type_rules::dotted(self.ast, node)
    }

    fn unparse(&self, node: NodeId) -> R<String> {
        crate::unparse::unparse(self.ast, node)
    }

    fn unparse_prefix(&self, node: NodeId, limit: usize) -> R<String> {
        crate::unparse::unparse_prefix(self.ast, node, limit)
    }

    /// `_dotted(node) or ast.unparse(node)`.
    fn spelled(&self, node: NodeId) -> R<String> {
        match self.dotted(node) {
            Some(spelled) if !spelled.is_empty() => Ok(spelled),
            _ => self.unparse(node),
        }
    }

    fn annotation_type(&self, node: Option<NodeId>) -> R<String> {
        crate::pine_type_rules::annotation_type(self.ast, node, &self.classes_lookup)
    }

    // --- scope plumbing --------------------------------------------------

    fn top(&self) -> usize {
        *self.stack.last().expect("a frame")
    }

    fn scope(&self) -> String {
        self.frames[self.top()].scope.clone()
    }

    fn context(&self) -> i64 {
        *self.context_stack.last().expect("a context")
    }

    fn push_frame(&mut self, frame: Frame) -> usize {
        self.frames.push(frame);
        let index = self.frames.len() - 1;
        self.stack.push(index);
        index
    }

    /// The signature a call name resolves to, searching outward (`_resolve_func`):
    /// (key in `funcs`, declaring frame index, shadowed).
    fn resolve_func(&self, name: &str, node: NodeId) -> Option<(String, usize, bool)> {
        let mut shadowed = false;
        for index in (0..self.stack.len()).rev() {
            let scope = &self.frames[self.stack[index]].scope;
            if self.rebound.get(scope).is_some_and(|names| names.contains(name))
                && !self.reaches_def(scope, name, node)
            {
                shadowed = true;
            }
            let key = qualify(scope, name);
            if self.funcs.contains_key(&key) {
                return Some((key, index, shadowed));
            }
        }
        None
    }

    /// Whether a call reaches the definition despite a rebinding of its name
    /// (`_reaches_def`).
    fn reaches_def(&self, scope: &str, name: &str, node: NodeId) -> bool {
        if !scope.is_empty() || self.stack.len() > 1 {
            return false;
        }
        let Some(rebinds) = self.module_rebinds.get(name).filter(|binds| !binds.is_empty()) else {
            return false;
        };
        let at = (self.line(node), self.col(node));
        let path = self.branches.of(at);
        let spans: Vec<(Pos, Pos)> =
            self.loop_stack.iter().map(|&stmt| facts::span(self.ast, stmt)).collect();
        for (position, rebind_path) in rebinds {
            if spans.iter().any(|(start, end)| start <= position && position <= end) {
                return false;
            }
            if facts::mutually_exclusive(self.ast, rebind_path, &path) {
                continue;
            }
            if *position < at {
                return false;
            }
        }
        true
    }

    /// Where a `global`/`nonlocal` declaration sends one name (`_declared_home`): the
    /// frame whose names to use, `None` for an ordinary local.
    fn declared_home(&self, name: &str) -> Option<usize> {
        let current = &self.frames[self.top()];
        if current.declared_global.contains(name) {
            return Some(self.stack[0]);
        }
        if current.declared_nonlocal.contains(name) {
            let inner = &self.stack[1..self.stack.len().saturating_sub(1).max(1)];
            for &frame in inner.iter().rev() {
                if self.frames[frame].names.contains_key(name) {
                    return Some(frame);
                }
            }
            return Some(self.top());
        }
        None
    }

    /// The frame and the binding a name reads (`_lookup`).
    fn lookup_at(&self, name: &str) -> Option<usize> {
        let current = &self.frames[self.top()];
        if (!current.declared_global.is_empty() || !current.declared_nonlocal.is_empty())
            && let Some(home) = self.declared_home(name)
        {
            return self.frames[home].names.contains_key(name).then_some(home);
        }
        self.stack.iter().rev().copied().find(|&frame| self.frames[frame].names.contains_key(name))
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        self.lookup_at(name).and_then(|frame| self.frames[frame].names.get(name))
    }

    fn lookup_mut(&mut self, name: &str) -> Option<&mut Binding> {
        let frame = self.lookup_at(name)?;
        self.frames[frame].names.get_mut(name)
    }

    /// Record an assignment (`_bind`).
    #[allow(clippy::too_many_arguments)]
    fn bind(
        &mut self,
        name: &str,
        ty: &str,
        node: NodeId,
        unknown: Option<Unknown>,
        series: bool,
        qual: u8,
        persistent: bool,
    ) -> R<()> {
        let home = self.declared_home(name).unwrap_or_else(|| self.top());
        let line = self.line(node);
        let series = series || self.frames[self.top()].declared_series.contains(name);
        let decl = self.nid(node);
        if !self.frames[home].names.contains_key(name) {
            let mut binding = Binding::new(name);
            binding.ty = ty.to_string();
            binding.line = line;
            binding.unknown = unknown;
            binding.series = series;
            binding.qual = qual;
            binding.depth = self.conds.len() as i64;
            binding.persistent = persistent;
            binding.decl = decl;
            self.frames[home].names.insert(name.to_string(), binding);
            self.note_na_store(home, name, ty, node);
            return Ok(());
        }
        let conds = self.conds.clone();
        let (before, existing_unknown) = {
            let existing = self.frames[home].names.get_mut(name).expect("bound");
            existing.series = existing.series || series;
            existing.persistent = existing.persistent || persistent;
            if decl.is_some() && decl == existing.decl {
                existing.qual = existing.qual.max(qual);
            } else if existing.persistent {
                existing.qual = SERIES;
            } else {
                let depth = usize::try_from(existing.depth).unwrap_or(0);
                let enclosing = conds.get(depth..).and_then(|rest| rest.iter().max().copied());
                existing.qual = existing.qual.max(qual).max(enclosing.unwrap_or(CONST));
            }
            (existing.ty.clone(), existing.unknown.is_some())
        };
        let joined = self.joined(&before, ty, node, &format!("'{name}'"))?;
        let unknown_now = if joined == UNKNOWN && !existing_unknown {
            Some(
                self.shape_unknown(&before, ty, node)
                    .or(unknown)
                    .unwrap_or_else(|| self.unknown("joined-branches", node, name)),
            )
        } else {
            None
        };
        {
            let existing = self.frames[home].names.get_mut(name).expect("bound");
            existing.ty = joined.clone();
            if let Some(unknown) = unknown_now {
                existing.unknown = Some(unknown);
            }
        }
        self.note_na_store(home, name, ty, node);
        self.settle_na_stores(home, name, &joined)
    }

    /// Remember every typeless `na` a value stored into a name holds (`_note_na_store`).
    fn note_na_store(&mut self, frame: usize, name: &str, ty: &str, node: NodeId) {
        if !self.is_expr(node) || !self.stack.contains(&frame) {
            return;
        }
        let found = facts::typeless_nas(self.ast, node, ty);
        let stored = self.frames[frame].na_stores.entry(name.to_string()).or_default();
        for (value, path) in found {
            if !stored.iter().any(|(known, _)| *known == value) {
                stored.push((value, path));
            }
        }
    }

    /// Convert every typeless `na` stored into a name to the na of its type
    /// (`_settle_na_stores`).
    fn settle_na_stores(&mut self, frame: usize, name: &str, ty: &str) -> R<()> {
        if !self.stack.contains(&frame) {
            return Ok(());
        }
        let stored = self.frames[frame].na_stores.get(name).cloned().unwrap_or_default();
        for (value, path) in stored {
            let kind = facts::na_widen(&facts::element_at(ty, &path));
            self.record_widen(value, kind)?;
        }
        Ok(())
    }

    /// Join two types one expression may have, reporting a shape conflict (`_joined`).
    fn joined(&mut self, left: &str, right: &str, node: NodeId, what: &str) -> R<String> {
        let joined = join(left, right);
        if is_tuple(left) && is_tuple(right) {
            let lefts = elements_of(left);
            let rights = elements_of(right);
            if lefts.len() == rights.len() {
                let pairs: Vec<(String, String)> = lefts
                    .iter()
                    .zip(&rights)
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .collect();
                for (index, (item, other)) in pairs.iter().enumerate() {
                    self.joined(item, other, node, &format!("element {} of {what}", index + 1))?;
                }
                return Ok(joined);
            }
        }
        let mut conflict = shape_conflict(left, right).map(|(a, b)| (a.to_string(), b.to_string()));
        if conflict.is_none() && joined == UNKNOWN && !unchecked(left) && !unchecked(right) {
            conflict = Some((left.to_string(), right.to_string()));
        }
        let Some((first, second)) = conflict else { return Ok(joined) };
        let nid = self.nid(node);
        if let Some(nid) = nid
            && !self.shape_diags_named.insert((nid, what.to_string()))
        {
            return Ok(joined);
        }
        let origin = self.shape_unknown(&first, &second, node).unwrap_or_else(|| {
            self.unknown(
                "joined-branches",
                node,
                &format!("{} vs {}", render_ty(&first), render_ty(&second)),
            )
        });
        self.diag(
            &format!(
                "{what} gets both {} and {}, which are different types",
                render_ty(&first),
                render_ty(&second)
            ),
            node,
            Some(origin),
            "make both branches the same type",
        );
        Ok(joined)
    }

    /// The provenance of a type lost to a shape conflict (`_shape_unknown`).
    fn shape_unknown(&self, left: &str, right: &str, node: NodeId) -> Option<Unknown> {
        shape_conflict(left, right)?;
        Some(Unknown {
            reason: "shape-mismatch".into(),
            line: self.line(node),
            col: self.col(node),
            detail: format!("{} vs {}", render_ty(left), render_ty(right)),
        })
    }

    fn unknown(&self, reason: &str, node: NodeId, detail: &str) -> Unknown {
        Unknown {
            reason: reason.to_string(),
            line: self.line(node),
            col: self.col(node),
            detail: detail.to_string(),
        }
    }

    fn diag(&mut self, message: &str, node: NodeId, origin: Option<Unknown>, fix: &str) {
        self.diags.push(Diag {
            message: message.to_string(),
            line: self.line(node),
            col: self.col(node),
            origin,
            fix: fix.to_string(),
            end_line: 0,
            end_col: 0,
        });
    }

    /// Report a node once, however many walks reach it (`_node_diag`).
    fn node_diag(&mut self, message: &str, node: NodeId, reason: &str, detail: &str, fix: &str) {
        if let Some(nid) = self.nid(node)
            && !self.shape_diags.insert(nid)
        {
            return;
        }
        let origin = self.unknown(reason, node, detail);
        self.diag(message, node, Some(origin), fix);
    }

    // --- the per-node type view ------------------------------------------

    /// The type this walk gave a node (`_ty_of`).
    fn ty_of(&self, node: NodeId) -> String {
        if let Some(nid) = self.nid(node)
            && let Some(ty) = self.ty.get(&nid)
        {
            return ty.clone();
        }
        crate::pine_type_rules::get_ty(self.ast, node).to_string()
    }

    /// Record a node's type in this walk and join it into the stamp (`_stamp`).
    fn stamp(&mut self, node: NodeId, ty: &str) {
        if let Some(nid) = self.nid(node) {
            self.ty.insert(nid, ty.to_string());
        }
        self.join_stamp(node, ty);
    }

    fn join_stamp(&mut self, node: NodeId, ty: &str) {
        let stamped = match self.ast.extra(node, TY_ATTR) {
            Some(Value::Str(existing)) => join(existing, ty),
            _ => ty.to_string(),
        };
        set_ty(self.ast, node, &stamped);
    }

    // --- entry point -----------------------------------------------------

    fn run(&mut self, tree: NodeId) -> R<()> {
        let statements = walk_statements(self.ast, tree);
        for node in statements {
            match self.ast.kind(node) {
                Kind::ImportFrom { module: Some(module), names, .. } if module == "pynecore" => {
                    for &alias in names {
                        if let Kind::Alias { name, asname } = self.ast.kind(alias) {
                            self.lib_aliases.insert(asname.clone().unwrap_or(name.clone()));
                        }
                    }
                }
                Kind::Import { names } => {
                    for &alias in names {
                        if let Kind::Alias { name, asname } = self.ast.kind(alias)
                            && name.starts_with("pynecore.lib")
                        {
                            let bound = asname
                                .clone()
                                .unwrap_or_else(|| name.split('.').next().unwrap_or("").into());
                            self.lib_aliases.insert(bound);
                        }
                    }
                }
                _ => {}
            }
        }
        self.lib_aliases.insert("lib".to_string());

        let body = match self.kind(tree) {
            Kind::Module { body, .. } => body.clone(),
            _ => return Err(Unsupported("PineType of a non-module".into())),
        };
        self.branches = BranchIndex::new(self.ast, &body);
        let bound = facts::bound_positions(self.ast, &body);
        self.module_rebinds = bound
            .into_iter()
            .map(|(name, positions)| {
                let paths = positions.into_iter().map(|at| (at, self.branches.of(at))).collect();
                (name, paths)
            })
            .collect();
        self.module_def_positions = facts::module_defs(self.ast, &body)
            .into_iter()
            .map(|(name, defs)| (name, defs.into_iter().map(|(at, _)| at).collect()))
            .collect();
        self.rebound.insert(String::new(), self.module_rebinds.keys().cloned().collect());
        let scan = facts::scan_tree(self.ast, tree)?;
        self.sec_writes = scan.writes;
        self.collect_imports(&body)?;
        self.collect_classes(tree, &scan.annotations)?;
        self.collect(&body, "")?;
        self.body(&body)?;
        self.flush_pending()?;
        self.stamp_window_machines();
        self.stamp_widen_sites();
        self.stamp_instance_vectors(tree);
        self.suppress_pins(tree);
        self.report_unhandled_widens()?;
        self.sync_module_bindings();
        let reported = report::unknown_diags(self, tree)?;
        self.diags.extend(reported);
        Ok(())
    }

    /// The module scope's bindings, as the table holds them (`table.bindings['']` is
    /// the module frame's own names).
    fn sync_module_bindings(&mut self) {
        let names = self.frames[0].names.clone();
        self.bindings.insert(String::new(), names);
    }

    /// Finish the table: the groups, the exportable names, the published methods
    /// and the import records, in the order `run` writes them.
    fn finish(mut self, tree: NodeId) -> R<PineTypeTable> {
        let mut groups = IndexMap::new();
        for key in self.overload_groups.clone() {
            groups.insert(key.clone(), self.impl_sigs(&key)?);
        }
        let body = match self.kind(tree) {
            Kind::Module { body, .. } => body.clone(),
            _ => Vec::new(),
        };
        let overloads = self.overload_groups.clone();
        let exportable: BTreeSet<String> =
            facts::exportable_names(self.ast, &body, &self.module_rebinds, &|name| {
                overloads.contains(name)
            });
        self.publish_methods()?;
        self.record_imports(&body)?;
        self.sync_module_bindings();
        let contexts = self
            .contexts
            .iter()
            .map(|(key, &index)| (key.clone(), self.results[index].clone()))
            .collect();
        Ok(PineTypeTable {
            module_path: self.module_path.clone(),
            bindings: self.bindings,
            funcs: self.funcs,
            contexts,
            calls: self.calls,
            diags: self.diags,
            groups,
            deps: self.deps,
            classes: self.classes,
            class_sigs: self.class_sigs_table,
            extensions: self.extensions,
            pins_suppressed: self.pins_suppressed,
            exportable,
            call_pos: self.call_pos,
        })
    }
}

/// The module's source path (`getattr(tree, '_module_file_path', '')`).
fn module_file_path(ast: &Ast, tree: NodeId) -> R<String> {
    match ast.extra(tree, "_module_file_path") {
        None => Ok(String::new()),
        Some(Value::Str(path)) => Ok(path.clone()),
        Some(other) => Err(Unsupported(format!("a module path of {other:?}"))),
    }
}

/// Infer and stamp the Pine types of a module, and attach its table
/// (`PineTypeTransformer.visit`).
pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    let module_path = module_file_path(ast, module)?;
    assign_node_ids(ast, module, 0);
    ctx.host.begin_analysis(&module_path)?;
    let result = (|| {
        let mut engine = Engine::new(ast, module, ctx.host, module_path);
        engine.qualify_windows = ctx.user_code;
        engine.run(module)?;
        engine.finish(module)
    })();
    ctx.host.end_analysis()?;
    let table = result?;
    let value = ast.add_native(table);
    ast.set_extra(module, TABLE_ATTR, value);
    Ok(module)
}
