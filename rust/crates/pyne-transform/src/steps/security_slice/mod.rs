//! Port of `transformers/security_slice.py`: per-context backward slicing of
//! `main()` for the `request.security` children, and the `FinalizeCloneDefaults`
//! lowering step of the same module.
//!
//! A security child re-runs the script's whole `main()` on every bar of its own
//! feed, though only one `__sec_write__` block of it produces the value the chart
//! waits for. The step emits, per static security context, the backward slice of
//! `main()` for that write as an ordinary module-level function:
//!
//! ```text
//! def __sec_main_0__():
//!     <the backward slice of main() for sec·<hash>·0>
//! ```
//!
//! Contexts of one `group` share ONE clone holding every member's write block and
//! reads. The clone is recorded in `__security_contexts__[sid]['slice_main']`; the
//! child runs it instead of `main()`. The slice is a conservative
//! over-approximation: every uncertainty keeps a statement, and a construct the
//! pass cannot classify drops the optimization for the whole module. A clone is
//! also where a guarded write is made unconditional (`_force_conditional_writes`),
//! so a context the slice does not serve still gets a clone of the whole `main()`
//! whenever its write is guarded. `PYNE_NO_SECURITY_SLICE` reaches the port as
//! [`Context::security_slice_disabled`].
//!
//! The parts:
//!
//! - [`facts`]: what the statements of `main()` and of the functions it reaches
//!   read, write and mutate, the alias classes, and the slice itself;
//! - [`reach`]: whether a context's write runs on every execution of the child's
//!   code, and the `varip` scan behind `closed_shift`;
//! - [`forcer`]: the rewrite making the guarded writes unconditional;
//! - this module: the module visit, the units, the clones and the registry
//!   entries.
//!
//! The Python pass caches the walks of a statement by node identity; the caches
//! only ever serve code that does not change while it is asked about, so the port
//! recomputes what it does not cache and gets the same answers.

mod facts;
mod forcer;
mod reach;

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{clone, copy_location, descendants, fix_missing_locations};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

use self::facts::Collector;
use self::reach::{Reach, declares_varip, pynecore_imported_names, rebound_names};
use super::dynamic_default::is_script_entry;
use super::security::collection::is_function;
use super::security::contexts::{ctx_get, iter_contexts};
use super::security::dependencies::DependencyAnalyzer;
use crate::pine_type_rules::{OBJECT, STR, stamp_lowering};
use crate::{Context, Error};

/// Name shape of an emitted clone (`CLONE_PREFIX`, `CLONE_SUFFIX`). The security
/// protocol's own injected names (`__sec_write__`, `__sec_read__`, ...) share this
/// namespace; a script spelling one of them would already be broken today.
const CLONE_PREFIX: &str = "__sec_main_";
const CLONE_SUFFIX: &str = "__";

/// Pseudo-name standing for state the pass cannot see: another module's globals,
/// an unresolvable call. A statement that touches it both reads and writes it, so
/// keeping any one of them keeps them all.
const EXTERNAL: &str = "\0external";

/// The protocol calls the security split injects. None of them touches a name of
/// the script, so they carry no data dependency of their own.
const PROTOCOL_CALLS: [&str; 5] =
    ["__sec_read__", "__sec_write__", "__sec_signal__", "__sec_wait__", "__ltf_unzip__"];

/// Calls the earlier passes emit that stand for a plain expression: they run no
/// script code of their own, so an unresolved one is not opaque.
const MARKER_CALLS: [&str; 1] = ["inline_series"];

/// The annotations that declare a `varip` variable (`varip_types.VARIP_TYPES`).
const VARIP_TYPES: [&str; 2] = ["IBPersistent", "IBPersistentSeries"];

/// One member of a unit: its position among the contexts, its sid and its
/// context dict literal.
type Member = (usize, String, NodeId);

/// The contexts one clone is built for.
type Unit = Vec<Member>;

/// A unit and the indexes of the statements of `main()` its slice keeps.
type Plan = (Unit, Option<Vec<usize>>);

/// Whether a node is a function definition.
fn is_def(ast: &Ast, node: NodeId) -> bool {
    is_function(ast.kind(node))
}

/// Every descendant of a node, not entering nested function definitions
/// (`_walk_own`).
fn walk_own(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    descendants(ast, node, &is_function)
}

/// The node, then every node [`walk_own`] yields for it (`_OwnWalks.own`).
fn own(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut nodes = vec![node];
    nodes.extend(walk_own(ast, node));
    nodes
}

/// The sid argument of a `__sec_read__` / `__sec_write__` call node (`_protocol_sid`).
fn protocol_sid<'a>(ast: &'a Ast, node: NodeId, name: &str) -> Option<&'a str> {
    let Kind::Call { func, args, .. } = ast.kind(node) else { return None };
    if !matches!(ast.kind(*func), Kind::Name { id, .. } if id == name) {
        return None;
    }
    match ast.kind(*args.first()?) {
        Kind::Constant { value: Constant::Str(sid), .. } => Some(sid),
        _ => None,
    }
}

/// Whether the node itself or one of its own descendants is the target
/// (`_contains_write`).
fn contains_write(ast: &Ast, node: NodeId, hit: &dyn Fn(&Ast, NodeId) -> bool) -> bool {
    hit(ast, node) || walk_own(ast, node).into_iter().any(|sub| hit(ast, sub))
}

/// The nodes of a top-level statement that RUN when the statement runs
/// (`_stmt_nodes`). A nested `def` only binds a name here: what its body reads and
/// writes materializes at the call sites, through the function summaries.
fn stmt_nodes(ast: &Ast, stmt: NodeId) -> Vec<NodeId> {
    let (Kind::FunctionDef { args, decorator_list, .. }
    | Kind::AsyncFunctionDef { args, decorator_list, .. }) = ast.kind(stmt)
    else {
        return vec![stmt];
    };
    let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(*args) else {
        unreachable!("no arguments")
    };
    let mut nodes = decorator_list.clone();
    nodes.extend(defaults.iter().copied());
    nodes.extend(kw_defaults.iter().flatten().copied());
    nodes
}

/// The base `Name` id of an attribute chain (`_chain_root_name`, `_is_lib_chain`).
fn chain_root_name(ast: &Ast, mut node: NodeId) -> Option<&str> {
    while let Kind::Attribute { value, .. } = ast.kind(node) {
        node = *value;
    }
    match ast.kind(node) {
        Kind::Name { id, .. } => Some(id),
        _ => None,
    }
}

/// Whether `test` is the protocol's own `__active_security__` dispatch
/// (`_is_dispatch_test`).
fn is_dispatch_test(ast: &Ast, test: NodeId) -> bool {
    pyne_ast::walk::walk(ast, test)
        .into_iter()
        .any(|sub| matches!(ast.kind(sub), Kind::Name { id, .. } if id == "__active_security__"))
}

/// Union-find over names that may denote the SAME object (`_Aliases`). Only the
/// partition is ever read (through [`Aliases::find`]), so which name of a class
/// stands for it does not matter.
#[derive(Default)]
struct Aliases {
    parent: HashMap<String, String>,
}

impl Aliases {
    fn find(&self, name: &str) -> String {
        let mut root = name;
        while let Some(parent) = self.parent.get(root) {
            if parent == root {
                break;
            }
            root = parent;
        }
        root.to_string()
    }

    fn union(&mut self, a: &str, b: &str) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent.insert(rb, ra);
        }
    }

    /// The alias classes the names belong to.
    fn classes<'n>(&self, names: impl IntoIterator<Item = &'n String>) -> HashSet<String> {
        names.into_iter().map(|name| self.find(name)).collect()
    }
}

/// The step.
pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("SecuritySlice on a non-module".into()));
    }
    let mut reach = Reach::default();
    let sliced = emit_clones(ast, module, &mut reach, ctx.security_slice_disabled);
    if let Some(contexts) = find_contexts(ast, module) {
        finalize_closed_shift(ast, module, contexts, &sliced, &mut reach);
    }
    Ok(module)
}

/// `FinalizeCloneDefaults` (`finalize_clone_defaults`): make the generated clone
/// defaults inert after the input lowering passes.
pub fn finalize_clone_defaults(
    ast: &mut Ast,
    module: NodeId,
    _ctx: &Context<'_>,
) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("FinalizeCloneDefaults on a non-module".into()));
    };
    for statement in body.clone() {
        let &Kind::FunctionDef { ref name, args, .. } = ast.kind(statement) else { continue };
        if !(name.starts_with(CLONE_PREFIX) && name.ends_with(CLONE_SUFFIX)) {
            continue;
        }
        let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(args) else {
            unreachable!("no arguments")
        };
        let (defaults, kw_defaults) = (defaults.clone(), kw_defaults.clone());
        let none = |ast: &mut Ast, at: NodeId| {
            let constant = ast.add(Kind::Constant { value: Constant::None, kind: None });
            copy_location(ast, constant, at);
            constant
        };
        let defaults: Vec<NodeId> = defaults.into_iter().map(|at| none(ast, at)).collect();
        let kw_defaults: Vec<Option<NodeId>> =
            kw_defaults.into_iter().map(|at| at.map(|at| none(ast, at))).collect();
        let Kind::Arguments { defaults: slot, kw_defaults: kw_slot, .. } = ast.kind_mut(args)
        else {
            unreachable!()
        };
        *slot = defaults;
        *kw_slot = kw_defaults;
    }
    Ok(module)
}

// --- module inspection ---

fn module_body(ast: &Ast, module: NodeId) -> &Vec<NodeId> {
    match ast.kind(module) {
        Kind::Module { body, .. } => body,
        _ => unreachable!("no module"),
    }
}

/// The script's `main` entry point, if the module has one (`_find_main`).
fn find_main(ast: &Ast, module: NodeId) -> Option<NodeId> {
    module_body(ast, module).iter().copied().find(|&stmt| {
        matches!(ast.kind(stmt), Kind::FunctionDef { name, .. } if name == "main")
            && is_script_entry(ast, stmt)
    })
}

/// The `__security_contexts__` dict literal, if the module has one (`_find_contexts`).
fn find_contexts(ast: &Ast, module: NodeId) -> Option<NodeId> {
    module_body(ast, module).iter().find_map(|&stmt| {
        let Kind::Assign { targets, value, .. } = ast.kind(stmt) else { return None };
        if targets.len() != 1 || !matches!(ast.kind(*value), Kind::Dict { .. }) {
            return None;
        }
        matches!(ast.kind(targets[0]), Kind::Name { id, .. } if id == "__security_contexts__")
            .then_some(*value)
    })
}

/// A context's `depends` / `late_reads` sid list (`_sid_list`).
fn sid_list(ast: &Ast, ctx: NodeId, name: &str) -> HashSet<String> {
    let Some(value) = ctx_get(ast, ctx, name) else { return HashSet::new() };
    let Kind::List { elts, .. } = ast.kind(value) else { return HashSet::new() };
    elts.iter()
        .filter_map(|&elt| match ast.kind(elt) {
            Kind::Constant { value: Constant::Str(sid), .. } => Some(sid.clone()),
            _ => None,
        })
        .collect()
}

/// A context's compile-time group ordinal, if it has one (`_ctx_group`), spelled as
/// the decimal digits of the Python int (a bool is the int it equals).
fn ctx_group(ast: &Ast, ctx: NodeId) -> Option<String> {
    match ast.kind(ctx_get(ast, ctx, "group")?) {
        Kind::Constant { value: Constant::Int(value), .. } => Some(value.to_string()),
        Kind::Constant { value: Constant::Bool(value), .. } => Some(i64::from(*value).to_string()),
        _ => None,
    }
}

/// Why this context may not get a clone at all, or `None` when it may
/// (`_skip_reason`): a `request.security_lower_tf` context, the plain-OHLCV fast
/// path (which already skips `main()` entirely) and a write block that does not
/// sit in `main()` itself.
fn skip_reason(
    ast: &Ast,
    sid: &str,
    ctx: NodeId,
    written: &HashSet<String>,
) -> Option<&'static str> {
    if ctx_get(ast, ctx, "is_ltf").is_some() {
        return Some("lower_tf");
    }
    if ctx_get(ast, ctx, "ohlcv_fields").is_some() {
        return Some("ohlcv_passthrough");
    }
    if !written.contains(sid) {
        return Some("write_block_outside_main");
    }
    None
}

/// Split the module's contexts into the units one clone is built for (`_slice_units`).
///
/// A context without a group is a unit of its own. The contexts of one group share
/// a single clone. A member the pass cannot slice at all (an LTF context, a write
/// block outside `main()`) takes the whole group down with it; a plain-OHLCV member
/// does not, and a group of nothing but plain-OHLCV members gets no clone.
fn slice_units(
    ast: &Ast,
    entries: &[(usize, String, NodeId)],
    written: &HashSet<String>,
) -> Vec<Unit> {
    let mut reasons: HashMap<&str, Option<&str>> = HashMap::new();
    let mut groups: HashMap<String, Unit> = HashMap::new();
    let mut ordered: Vec<(&Member, Option<String>)> = Vec::new();
    for entry in entries {
        let (_, sid, ctx) = entry;
        reasons.insert(sid, skip_reason(ast, sid, *ctx, written));
        let group = ctx_group(ast, *ctx);
        if let Some(group) = &group {
            groups.entry(group.clone()).or_default().push(entry.clone());
        }
        ordered.push((entry, group));
    }
    let mut units = Vec::new();
    let mut emitted: HashSet<String> = HashSet::new();
    for (entry, group) in ordered {
        let Some(group) = group else {
            if reasons[entry.1.as_str()].is_none() {
                units.push(vec![entry.clone()]);
            }
            continue;
        };
        if !emitted.insert(group.clone()) {
            continue;
        }
        let members = &groups[&group];
        let blocking = members.iter().any(|(_, sid, _)| {
            matches!(reasons[sid.as_str()], Some(reason) if reason != "ohlcv_passthrough")
        });
        let sliceable = members.iter().any(|(_, sid, _)| reasons[sid.as_str()].is_none());
        if blocking || !sliceable {
            continue;
        }
        units.push(members.clone());
    }
    units
}

/// The units no sliced clone serves, grouped the way their children are
/// (`_whole_units`).
fn whole_units(
    ast: &Ast,
    entries: &[(usize, String, NodeId)],
    covered: &HashSet<String>,
) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    let mut by_group: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        if covered.contains(&entry.1) {
            continue;
        }
        match ctx_group(ast, entry.2) {
            None => units.push(vec![entry.clone()]),
            Some(group) => match by_group.get(&group) {
                Some(&position) => units[position].push(entry.clone()),
                None => {
                    by_group.insert(group, units.len());
                    units.push(vec![entry.clone()]);
                }
            },
        }
    }
    units
}

// --- the clones ---

/// Emit the clones and record what each served context's child runs
/// (`_emit_clones`): per sid, the statements of the clone its child runs (absent
/// for a context that got no clone, whose child runs the whole `main()`).
fn emit_clones(
    ast: &mut Ast,
    module: NodeId,
    reach: &mut Reach,
    disabled: bool,
) -> HashMap<String, Vec<NodeId>> {
    let (Some(contexts), Some(main)) = (find_contexts(ast, module), find_main(ast, module)) else {
        return HashMap::new();
    };
    let entries: Vec<(usize, String, NodeId)> = iter_contexts(ast, contexts)
        .into_iter()
        .enumerate()
        .map(|(index, (sid, ctx))| (index, sid, ctx))
        .collect();
    let main_body = match ast.kind(main) {
        Kind::FunctionDef { body, .. } => body.clone(),
        _ => unreachable!(),
    };

    // A sliced clone per unit the analysis can serve, then a clone of the WHOLE
    // ``main()`` for every other unit with a guarded write: forcing the write is a
    // matter of what the child computes, not of how much of ``main()`` it may skip,
    // so it cannot depend on the slice succeeding.
    let mut plans: Vec<Plan> =
        if disabled { Vec::new() } else { sliced_units(ast, module, main, &entries) };
    let covered: HashSet<String> =
        plans.iter().flat_map(|(unit, _)| unit.iter().map(|(_, sid, _)| sid.clone())).collect();
    let whole: Vec<usize> = (0..main_body.len()).collect();
    plans.extend(whole_units(ast, &entries, &covered).into_iter().map(|unit| (unit, None)));

    // A clone is a copy of ``main()``'s kept statements, so the writes it would
    // reach are judged on ``main()``'s own statements -- whose facts every unit
    // shares -- and a whole unit with nothing to force is dropped before paying for
    // the copy.
    let mut builds: Vec<(Unit, Vec<usize>, bool, HashSet<String>)> = Vec::new();
    for (unit, kept) in plans {
        match kept {
            None => {
                let guarded = guarded_sids(ast, &unit, &main_body, reach);
                if !guarded.is_empty() {
                    builds.push((unit, whole.clone(), true, guarded));
                }
            }
            Some(kept) => {
                let stmts: Vec<NodeId> = kept.iter().map(|&index| main_body[index]).collect();
                let guarded = guarded_sids(ast, &unit, &stmts, reach);
                builds.push((unit, kept, false, guarded));
            }
        }
    }

    let mut clones: Vec<NodeId> = Vec::new();
    let mut sliced: HashMap<String, Vec<NodeId>> = HashMap::new();
    for (unit, kept, is_whole, guarded) in builds {
        // The unit's first member names the clone: its index is its own, so no
        // other unit can pick the same name.
        let name = format!("{CLONE_PREFIX}{}{CLONE_SUFFIX}", unit[0].0);
        let clone = build_clone(ast, main, &main_body, &kept, &name);
        let forced =
            !guarded.is_empty() && forcer::force_conditional_writes(ast, module, clone, &guarded);
        if is_whole && !forced {
            // Nothing to gain: the clone would be ``main()`` itself.
            continue;
        }
        clones.push(clone);
        clones.extend(bind_defaults(ast, "main", &name));
        let body = match ast.kind(clone) {
            Kind::FunctionDef { body, .. } => body.clone(),
            _ => unreachable!(),
        };
        for (_, sid, ctx) in &unit {
            sliced.insert(sid.clone(), body.clone());
            // The type pass has already run, so the two new literals are stamped
            // here -- an unstamped node inside a stamped subtree loses the type of
            // everything above it.
            let key =
                ast.add(Kind::Constant { value: Constant::Str("slice_main".into()), kind: None });
            let key = stamp_lowering(ast, key, STR);
            let value = ast.add(Kind::Constant { value: Constant::Str(name.clone()), kind: None });
            let value = stamp_lowering(ast, value, STR);
            let Kind::Dict { keys, values } = ast.kind_mut(*ctx) else { unreachable!() };
            keys.push(Some(key));
            values.push(value);
        }
    }

    if !clones.is_empty() {
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        let insert_at = body.iter().position(|&stmt| stmt == main).expect("main in the module") + 1;
        body.splice(insert_at..insert_at, clones);
    }
    sliced
}

/// The units a backward slice serves, each with the statements it keeps
/// (`_sliced_units`); empty when the module cannot be sliced at all.
fn sliced_units(
    ast: &Ast,
    module: NodeId,
    main: NodeId,
    entries: &[(usize, String, NodeId)],
) -> Vec<Plan> {
    let scopes = DependencyAnalyzer::scopes_only(ast, module);
    let Some(main_key) = scopes.key_of.get(&main).cloned() else { return Vec::new() };
    let (reachable, unmodelled) = facts::reachable_scopes(ast, &scopes, &main_key);
    if reachable.iter().any(|key| unmodelled.contains(key)) {
        return Vec::new();
    }

    // An import and a definition are statements, so the statements alone hold every
    // one; the module's nodes are its own walk and the own walk of every definition.
    let statements = pyne_ast::walk::walk_statements(ast, module);
    let import_names = facts::module_imports(ast, &statements);
    let mut nodes = own(ast, module);
    for &stmt in &statements {
        if is_def(ast, stmt) {
            nodes.extend(own(ast, stmt));
        }
    }
    let rebound = rebound_names(ast, &nodes);
    let mut library_calls: HashSet<String> =
        pynecore_imported_names(ast, &statements).difference(&rebound).cloned().collect();
    library_calls.extend(MARKER_CALLS.iter().map(|name| name.to_string()));
    let mut collector = Collector::new(ast, &scopes, import_names, library_calls);
    // Module level binds names too (``store = array.new_float(1)`` followed by
    // ``alias = store``), and both names denote the same object at runtime, so its
    // bindings join the alias classes as well.
    collector.record_aliases("");
    for key in &reachable {
        collector.record_aliases(key);
    }
    let summaries = facts::function_summaries(&mut collector, &scopes, &reachable, &main_key);

    let main_body = match ast.kind(main) {
        Kind::FunctionDef { body, .. } => body.clone(),
        _ => unreachable!(),
    };
    let index = facts::build_index(ast, &main_body, &mut collector, &summaries, &main_key);
    if collector.unresolved {
        return Vec::new();
    }
    let written: HashSet<String> =
        index.iter().flat_map(|entry| entry.write_sids.iter().cloned()).collect();
    let definers = facts::definers(&index);

    let mut plans = Vec::new();
    for unit in slice_units(ast, entries, &written) {
        let mut keep_reads: HashSet<String> = HashSet::new();
        for (_, sid, ctx) in &unit {
            keep_reads.insert(sid.clone());
            keep_reads.extend(sid_list(ast, *ctx, "depends"));
            keep_reads.extend(sid_list(ast, *ctx, "late_reads"));
        }
        let write_sids: HashSet<String> = unit.iter().map(|(_, sid, _)| sid.clone()).collect();
        if let Some(kept) = facts::slice(&index, &definers, &write_sids, &keep_reads) {
            plans.push((unit, Some(kept)));
        }
    }
    plans
}

/// The unit's contexts whose write does not run on every execution of `stmts`
/// (`_guarded_sids`). A plain-OHLCV member is never among them: its child does not
/// run the clone at all.
fn guarded_sids(ast: &Ast, unit: &Unit, stmts: &[NodeId], reach: &mut Reach) -> HashSet<String> {
    unit.iter()
        .filter(|(_, sid, ctx)| {
            ctx_get(ast, *ctx, "ohlcv_fields").is_none() && !reach.write_reached(ast, stmts, sid)
        })
        .map(|(_, sid, _)| sid.clone())
        .collect()
}

/// Build the clone function from `main`'s signature and the kept body
/// (`_build_clone`).
///
/// The clone carries no `@script.*` decorator, but keeps `main`'s parameter list so
/// the child can call it with no arguments exactly like `main`. Only the KEPT
/// statements are copied, through one memo shared across the statements of the
/// clone, so an object hanging off several of them is copied once.
///
/// The Python pass builds several clones of one module from parts of `main()` it
/// copies and locates once (`_CloneSource`), which builds this very clone whenever
/// it is used (`_CloneSource.usable`), so the port always builds it this way.
fn build_clone(
    ast: &mut Ast,
    main: NodeId,
    main_body: &[NodeId],
    kept: &[usize],
    name: &str,
) -> NodeId {
    let mut memo = HashMap::new();
    // ``copy.copy(main)``: the same fields and stamps, the stamped values shared
    let shallow = ast[main].clone();
    let clone_id = ast.add(shallow.kind);
    ast[clone_id].loc = shallow.loc;
    ast[clone_id].extra = shallow.extra;
    let Kind::FunctionDef { args, returns, type_params, .. } = ast.kind(main).clone() else {
        unreachable!()
    };
    let args = clone(ast, args, &mut memo);
    let returns = returns.map(|returns| clone(ast, returns, &mut memo));
    let type_params: Vec<NodeId> =
        type_params.into_iter().map(|param| clone(ast, param, &mut memo)).collect();
    let mut body: Vec<NodeId> =
        kept.iter().map(|&index| clone(ast, main_body[index], &mut memo)).collect();
    if body.is_empty() {
        body.push(ast.add(Kind::Pass));
    }
    let Kind::FunctionDef {
        name: slot_name,
        args: slot_args,
        body: slot_body,
        decorator_list,
        returns: slot_returns,
        type_params: slot_params,
        ..
    } = ast.kind_mut(clone_id)
    else {
        unreachable!()
    };
    *slot_name = name.to_string();
    decorator_list.clear();
    *slot_args = args;
    *slot_returns = returns;
    *slot_params = type_params;
    *slot_body = body;
    copy_location(ast, clone_id, main);
    fix_missing_locations(ast, clone_id);
    clone_id
}

/// Statements rebinding the clone's parameter defaults to `main`'s
/// (`_bind_defaults`). The values `main` was bound with include the configured
/// inputs; the clone receives the same defaults without evaluating their
/// expressions again.
fn bind_defaults(ast: &mut Ast, main_name: &str, clone_name: &str) -> Vec<NodeId> {
    // The type pass is behind us, so every emitted expression carries its own
    // stamp; a function object's dunder is opaque runtime state.
    let place = |ast: &mut Ast, name: &str, attr: &str, ctx: ExprContext| {
        let value = ast.add(Kind::Name { id: name.to_string(), ctx: ExprContext::Load });
        let attribute = ast.add(Kind::Attribute { value, attr: attr.to_string(), ctx });
        stamp_lowering(ast, attribute, OBJECT)
    };
    ["__defaults__", "__kwdefaults__"]
        .into_iter()
        .map(|attr| {
            let target = place(ast, clone_name, attr, ExprContext::Store);
            let value = place(ast, main_name, attr, ExprContext::Load);
            let stmt = ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
            fix_missing_locations(ast, stmt);
            stmt
        })
        .collect()
}

/// Clear `closed_shift` wherever the child's rounds are observable
/// (`_finalize_closed_shift`): `varip` state in the code the child runs, and a
/// write that does not run on every round. Only the slicer knows WHAT the child
/// runs: its own clone's kept statements, or the whole `main()` when it got no
/// clone.
fn finalize_closed_shift(
    ast: &mut Ast,
    module: NodeId,
    contexts: NodeId,
    sliced: &HashMap<String, Vec<NodeId>>,
    reach: &mut Reach,
) {
    let candidates: Vec<(String, NodeId)> = iter_contexts(ast, contexts)
        .into_iter()
        .filter_map(|(sid, ctx)| ctx_closed_shift(ast, ctx).map(|value| (sid, value)))
        .collect();
    if candidates.is_empty() {
        return;
    }
    // One walk of the module, clones included, serves all three scans.
    let nodes = pyne_ast::walk::walk(ast, module);
    let mut funcs: HashMap<String, Vec<NodeId>> = HashMap::new();
    for &node in &nodes {
        if is_def(ast, node) {
            funcs
                .entry(super::security::collection::function_name(ast, node).to_string())
                .or_default()
                .push(node);
        }
    }
    let rebound = rebound_names(ast, &nodes);
    let pyne_imported = pynecore_imported_names(ast, &nodes);
    let main = find_main(ast, module);
    let mut whole: Option<bool> = None;
    // The members of a group share one clone body, so its scan is made once.
    let mut varip: HashMap<Vec<NodeId>, bool> = HashMap::new();
    let mut cleared = Vec::new();
    for (sid, value) in candidates {
        let observable = match sliced.get(&sid) {
            None => match main {
                None => true,
                Some(main) => {
                    let body = match ast.kind(main) {
                        Kind::FunctionDef { body, .. } => body.clone(),
                        _ => unreachable!(),
                    };
                    let declares = *whole.get_or_insert_with(|| {
                        declares_varip(ast, &body, &funcs, &rebound, &pyne_imported)
                    });
                    declares || !reach.write_reached(ast, &body, &sid)
                }
            },
            Some(stmts) => {
                let declares = *varip.entry(stmts.clone()).or_insert_with(|| {
                    declares_varip(ast, stmts, &funcs, &rebound, &pyne_imported)
                });
                declares || !reach.write_reached(ast, stmts, &sid)
            }
        };
        if observable {
            cleared.push(value);
        }
    }
    for value in cleared {
        let Kind::Constant { value: constant, .. } = ast.kind_mut(value) else { unreachable!() };
        *constant = Constant::Bool(false);
    }
}

/// The context's `closed_shift` literal, only while it says True (`_ctx_closed_shift`).
fn ctx_closed_shift(ast: &Ast, ctx: NodeId) -> Option<NodeId> {
    let value = ctx_get(ast, ctx, "closed_shift")?;
    matches!(ast.kind(value), Kind::Constant { value: Constant::Bool(true), .. }).then_some(value)
}
