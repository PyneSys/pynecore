//! Port of `transformers/security.py`: every `lib.request.security[_lower_tf]` call
//! becomes the multiprocessing signal / write / read protocol, and the module gets
//! its `__security_contexts__` registry.
//!
//! Each call gets a sec id, `sec·<sha1(module path)[:8]>·<n>`, and:
//!
//! 1. at function start (chart context only), a `__sec_signal__` with its symbol,
//!    timeframe and runtime lookahead, when those can be evaluated there; inline
//!    before the write otherwise;
//! 2. at the call's statement, a write block (`__sec_write__`) for the security
//!    process (or a same-context chart), and the call replaced by `__sec_read__`;
//! 3. at function end (chart context only), a `__sec_wait__`.
//!
//! The parts, in the order the step runs them:
//!
//! - [`collection`]: the arguments of a call, the "simple chain" / hoistable
//!   binding rules and the history-reference rule behind `closed_shift`;
//! - [`builders`]: the nodes the protocol is written with;
//! - [`rewrite`]: the per-function lowering;
//! - [`lift`]: the cross-function signal lift;
//! - [`dependencies`]: the taint analysis behind `depends` / `late_reads` /
//!   `in_loop` (shared with the slice pass);
//! - [`groups`]: `always_read`, `signal_per_bar` and the feed groups;
//! - [`contexts`]: the reading of the `__security_contexts__` literal, for the
//!   steps after this one;
//! - this module: the module visit and the registry it emits.

pub mod builders;
pub mod collection;
pub mod contexts;
pub mod dependencies;
pub mod groups;
pub mod lift;
pub mod predicates;
pub mod rewrite;

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::walk::clone;
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, Value};

use self::dependencies::DependencyAnalyzer;
use self::predicates::has_security_call;
use crate::sha1::sha1_hex;
use crate::{Context, Error};

/// A context's registry entries, by key, in the order they were set (a Python dict).
pub(crate) type Entries = IndexMap<String, NodeId>;

/// What one function scope binds, as the passes after this one see it (`_SeriesScope`).
#[derive(Debug, Default)]
pub(crate) struct SeriesScope {
    /// Every name a series annotation or a `Series`-annotated parameter gives a
    /// history buffer to
    series: HashSet<String>,
    /// Every name the scope binds
    locals: HashSet<String>,
    /// The value each `__hist_N__` temp of this scope holds
    hist: HashMap<String, NodeId>,
}

/// How often a function binds or deletes each name, and what it declares `global` /
/// `nonlocal` (`_binding_counts`).
pub(crate) type BindingCounts = (HashMap<String, usize>, HashSet<String>);

/// The step's state over one module (`SecurityTransformer`).
pub(crate) struct Security {
    counter: usize,
    all_contexts: IndexMap<String, Entries>,
    /// Per sid, the symbol, timeframe and runtime lookahead its signal passes
    signal_args: HashMap<String, [Option<NodeId>; 3]>,
    needs_barmerge: bool,
    needs_ltf_unzip: bool,
    ltf_sec_ids: HashSet<String>,
    module_file: String,
    /// `module_file` as last hashed by [`Security::gen_id`], and its hash
    hashed_file: Option<String>,
    module_hash: String,
    /// Sids whose `__sec_signal__` is emitted in the function's top block
    top_sec_ids: HashSet<String>,
    /// Runtime-resolved sids kept out of the top block because their call is not
    /// reached on every run
    deferred_by_reach: HashSet<String>,
    /// The deferred sids that must stay in the top block anyway (the re-run)
    keep_top: HashSet<String>,
    /// Bindings of the function scope chain being processed, innermost last
    scopes: Vec<SeriesScope>,
    /// The binding counts [`Security::lowered_hoistable`] keeps per function
    binding_counts_memo: HashMap<NodeId, BindingCounts>,
}

impl Security {
    fn new(keep_top: HashSet<String>) -> Security {
        Security {
            counter: 0,
            all_contexts: IndexMap::new(),
            signal_args: HashMap::new(),
            needs_barmerge: false,
            needs_ltf_unzip: false,
            ltf_sec_ids: HashSet::new(),
            module_file: "<script>".to_string(),
            hashed_file: None,
            module_hash: String::new(),
            top_sec_ids: HashSet::new(),
            deferred_by_reach: HashSet::new(),
            keep_top,
            scopes: Vec::new(),
            binding_counts_memo: HashMap::new(),
        }
    }

    /// A fresh sec id. The module hash keeps sec ids unique across modules: the
    /// main script and any imported library may each have their own security
    /// calls, and their contexts are merged into one registry by the runner.
    fn gen_id(&mut self) -> String {
        if self.hashed_file.as_deref() != Some(self.module_file.as_str()) {
            self.hashed_file = Some(self.module_file.clone());
            self.module_hash = sha1_hex(self.module_file.as_bytes())[..8].to_string();
        }
        let sec_id = format!("sec\u{b7}{}\u{b7}{}", self.module_hash, self.counter);
        self.counter += 1;
        sec_id
    }

    /// The module visit (`visit_Module`); answers the module node the step hands back.
    fn visit_module(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Error> {
        if !has_security_call(ast, node) {
            // Every rewrite of this pass starts at a security call: a module
            // without one would only be walked to be handed back unchanged
            return Ok(node);
        }
        self.module_file = match ast.extra(node, "_module_file_path") {
            None => "<script>".to_string(),
            Some(Value::Str(path)) => path.clone(),
            Some(_) => {
                return Err(Error::Unsupported("a _module_file_path that is no str".into()));
            }
        };
        // Only a context deferred by reach can call for the re-run below, which
        // starts from the module as it is now: the copy is kept only when the
        // lowering is going to defer one.
        let original = if self.keep_top.is_empty() && self.may_defer(ast, node) {
            Some(clone(ast, node, &mut HashMap::new()))
        } else {
            None
        };
        self.visit_nested_defs(ast, node)?;

        if self.all_contexts.is_empty() {
            return Ok(node);
        }
        let (load_counts, signal_counts) = self.lift_helper_signals(ast, node);
        self.analyze_dependencies(ast, node)?;
        // Dependencies are only known on the lowered module. A deferred signal
        // that turns out to be a producer must stay in the top block after all, so
        // transform the module again with that decision fixed. Sids are counted in
        // visit order, so the second pass names every context the same way.
        let mut producers: HashSet<String> = HashSet::new();
        for entries in self.all_contexts.values() {
            let deps = entries["depends"];
            if let Kind::List { elts, .. } = ast.kind(deps) {
                for &dep in elts {
                    if let Kind::Constant { value: Constant::Str(sid), .. } = ast.kind(dep) {
                        producers.insert(sid.clone());
                    }
                }
            }
        }
        let keep_top: HashSet<String> =
            self.deferred_by_reach.intersection(&producers).cloned().collect();
        if !keep_top.is_empty() && self.keep_top.is_empty() {
            let Some(original) = original else {
                return Err(Error::Unsupported(
                    "security: a deferral the up-front check did not foresee".into(),
                ));
            };
            return Security::new(keep_top).visit_module(ast, original);
        }
        self.mark_always_read(ast, node);
        self.mark_signal_per_bar(ast, node, &signal_counts);
        // Groups are decided on the FINAL pass only: the re-run above starts from
        // the untransformed module and assigns its own.
        self.assign_groups(ast, node, &load_counts);

        // The barmerge import (this step runs after the import normalizer, so it
        // adds the import itself)
        if self.needs_barmerge {
            let alias = ast.add(Kind::Alias { name: "pynecore.lib.barmerge".into(), asname: None });
            let import = ast.add(Kind::Import { names: vec![alias] });
            module_body(ast, node).insert(0, import);
        }
        // __ltf_unzip__ transposes tuple-valued security_lower_tf() results
        if self.needs_ltf_unzip {
            let alias = ast.add(Kind::Alias { name: "__ltf_unzip__".into(), asname: None });
            let import = ast.add(Kind::ImportFrom {
                module: Some("pynecore.core.security".into()),
                names: vec![alias],
                level: Some(0),
            });
            module_body(ast, node).insert(0, import);
        }

        let target =
            ast.add(Kind::Name { id: "__security_contexts__".into(), ctx: ExprContext::Store });
        let contexts: Vec<(String, Entries)> = self.all_contexts.drain(..).collect();
        let mut keys = Vec::with_capacity(contexts.len());
        let mut values = Vec::with_capacity(contexts.len());
        for (sid, entries) in contexts {
            keys.push(Some(builders::constant(ast, Constant::Str(sid))));
            let mut entry_keys = Vec::with_capacity(entries.len());
            let mut entry_values = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                entry_keys.push(Some(builders::constant(ast, Constant::Str(key))));
                entry_values.push(value);
            }
            values.push(ast.add(Kind::Dict { keys: entry_keys, values: entry_values }));
        }
        let value = ast.add(Kind::Dict { keys, values });
        let assign = ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
        module_body(ast, node).push(assign);
        Ok(node)
    }

    /// Run the taint analysis on the lowered module and record its result
    /// (`_analyze_dependencies`): `depends` into every context, `late_reads` when
    /// there are any, and `in_loop` for a write that can run more than once per bar.
    fn analyze_dependencies(&mut self, ast: &mut Ast, module: NodeId) -> Result<(), Error> {
        let sids: Vec<String> = self.all_contexts.keys().cloned().collect();
        let mut analyzer = DependencyAnalyzer::new(ast, module, &sids);
        analyzer.run()?;
        let depends = std::mem::take(&mut analyzer.depends);
        let late_reads = std::mem::take(&mut analyzer.late_reads);
        let in_loop = std::mem::take(&mut analyzer.in_loop);
        let known: HashSet<&String> = sids.iter().collect();
        for sid in &sids {
            let mut deps: Vec<&String> = depends
                .get(sid)
                .map(|deps| deps.iter().filter(|dep| known.contains(dep)).collect())
                .unwrap_or_default();
            deps.sort();
            let list = builders::str_list(ast, &deps);
            let entries = &mut self.all_contexts[sid];
            entries.insert("depends".into(), list);
            let mut late: Vec<&String> = late_reads
                .get(sid)
                .map(|late| late.iter().filter(|dep| known.contains(dep)).collect())
                .unwrap_or_default();
            late.sort();
            if !late.is_empty() {
                let list = builders::str_list(ast, &late);
                self.all_contexts[sid].insert("late_reads".into(), list);
            }
            if in_loop.contains(sid) {
                let flag = builders::constant(ast, Constant::Bool(true));
                self.all_contexts[sid].insert("in_loop".into(), flag);
            }
        }
        Ok(())
    }
}

/// The module's statement list.
fn module_body(ast: &mut Ast, module: NodeId) -> &mut Vec<NodeId> {
    match ast.kind_mut(module) {
        Kind::Module { body, .. } => body,
        _ => unreachable!("the security step runs on a module"),
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("security: no module".into()));
    }
    Security::new(HashSet::new()).visit_module(ast, module)
}
