//! The stamps written once every body is walked: the window machines, the
//! per-instance conversion sites, the instance vectors, and the pin suppression.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;
use pyne_ast::walk::{copy_location, walk};
use pyne_ast::{Constant, ExprContext, Int, Kind, NodeId, Value};

use super::facts::{self, WIDEN_HARMLESS, own_calls, per_context, vector_widen};
use super::{Engine, R};
use crate::pine_qualifier::{get_series_lens, set_series_len, set_series_lens};
use crate::pine_type_rules::{
    class_of, get_pins, get_varying, get_widen_site, get_widens, set_pin, set_pins, set_varying,
    set_vector, set_widen_site, set_widens,
};

/// One entry of an instance vector: what one varying site resolves to in one
/// instance.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Entry {
    None,
    Str(String),
    Bool(bool),
    Vector(Rc<Vec<Entry>>),
}

impl Entry {
    fn from_text(text: Option<&String>) -> Entry {
        text.map_or(Entry::None, |text| Entry::Str(text.clone()))
    }

    fn value(&self) -> Value {
        match self {
            Entry::None => Value::None,
            Entry::Str(text) => Value::Str(text.clone()),
            Entry::Bool(flag) => Value::Bool(*flag),
            Entry::Vector(items) => Value::Tuple(items.iter().map(Entry::value).collect()),
        }
    }
}

/// A call's widen sites per context id, or `None` when it has none.
type Widens = Option<Vec<(i64, Option<String>)>>;

/// The state `_stamp_instance_vectors`' two closures share.
struct Vectors<'e> {
    varying: IndexMap<String, Vec<NodeId>>,
    memo: HashMap<(String, i64), Rc<Vec<Entry>>>,
    widens: HashMap<NodeId, Widens>,
    series_len: &'e IndexMap<i64, IndexMap<i64, bool>>,
    pins: &'e HashMap<i64, IndexMap<i64, Option<String>>>,
    callee_key: &'e HashMap<i64, String>,
    call_ctx: &'e HashMap<i64, IndexMap<i64, i64>>,
}

impl Vectors<'_> {
    fn vector_of(
        &mut self,
        ast: &pyne_ast::Ast,
        key: &str,
        cid: i64,
        stack: &HashSet<(String, i64)>,
    ) -> Entry {
        let Some(sites) = self.varying.get(key).filter(|sites| !sites.is_empty()).cloned() else {
            return Entry::Vector(Rc::new(Vec::new()));
        };
        let token = (key.to_string(), cid);
        if let Some(found) = self.memo.get(&token) {
            return Entry::Vector(found.clone());
        }
        if stack.contains(&token) {
            return Entry::Vector(Rc::new(vec![Entry::None; sites.len()]));
        }
        let mut inner = stack.clone();
        inner.insert(token.clone());
        let built: Vec<Entry> =
            sites.iter().map(|&site| self.entry_of(ast, site, cid, &inner)).collect();
        let built = Rc::new(built);
        self.memo.insert(token, built.clone());
        Entry::Vector(built)
    }

    fn entry_of(
        &mut self,
        ast: &pyne_ast::Ast,
        site: NodeId,
        cid: i64,
        stack: &HashSet<(String, i64)>,
    ) -> Entry {
        let widens = self.widens.entry(site).or_insert_with(|| get_widens(ast, site)).clone();
        if let Some(widens) = widens {
            return match widens.iter().find(|(known, _)| *known == cid) {
                Some((_, kind)) => Entry::from_text(kind.as_ref()),
                None => Entry::None,
            };
        }
        let Some(site_id) = facts::nid(ast, site) else { return Entry::None };
        if get_series_lens(ast, site).is_some() {
            return self
                .series_len
                .get(&site_id)
                .and_then(|seen| seen.get(&cid))
                .map_or(Entry::None, |&flag| Entry::Bool(flag));
        }
        if get_pins(ast, site).is_some() {
            return self
                .pins
                .get(&site_id)
                .and_then(|seen| seen.get(&cid))
                .map_or(Entry::None, |pin| Entry::from_text(pin.as_ref()));
        }
        let site_callee = self.callee_key.get(&site_id).cloned();
        let target = self.call_ctx.get(&site_id).and_then(|seen| seen.get(&cid)).copied();
        match (site_callee, target) {
            (Some(callee), Some(target)) => self.vector_of(ast, &callee, target, stack),
            _ => Entry::None,
        }
    }
}

impl Engine<'_, '_> {
    /// Stamp each window call with the machine its length selects
    /// (`_stamp_window_machines`).
    pub(super) fn stamp_window_machines(&mut self) {
        for (nid, seen) in self.series_len.clone() {
            let node = self.window_nodes[&nid];
            let distinct: HashSet<bool> = seen.values().copied().collect();
            if distinct.len() == 1 {
                let value = *seen.values().next().expect("a verdict");
                set_series_len(self.ast, node, Some(value));
                set_series_lens(self.ast, node, None);
            } else if get_pins(self.ast, node).is_some() {
                set_series_len(self.ast, node, Some(seen.values().any(|&flag| flag)));
                set_series_lens(self.ast, node, None);
            } else {
                set_series_len(self.ast, node, None);
                let pairs: Vec<(i64, bool)> =
                    seen.iter().map(|(&cid, &flag)| (cid, flag)).collect();
                set_series_lens(self.ast, node, Some(&pairs));
            }
        }
    }

    /// Give every value whose conversion differs per context its per-instance site
    /// (`_stamp_widen_sites`).
    pub(super) fn stamp_widen_sites(&mut self) {
        for (nid, seen) in self.widens.clone() {
            let Some(&node) = self.widen_nodes.get(&nid) else { continue };
            let mut needed: Vec<&Option<String>> = Vec::new();
            for kind in seen.values() {
                if kind.as_deref() != Some(WIDEN_HARMLESS) && !needed.contains(&kind) {
                    needed.push(kind);
                }
            }
            if needed.len() < 2 {
                set_widen_site(self.ast, node, None);
                continue;
            }
            let module =
                self.ast.add(Kind::Name { id: "safe_convert".into(), ctx: ExprContext::Load });
            let func = self.ast.add(Kind::Attribute {
                value: module,
                attr: "implicit_convert".into(),
                ctx: ExprContext::Load,
            });
            let none = self.ast.add(Kind::Constant { value: Constant::None, kind: None });
            let site = self.ast.add(Kind::Call { func, args: vec![node, none], keywords: vec![] });
            copy_location(self.ast, site, node);
            let pairs: Vec<(i64, Option<String>)> =
                seen.iter().map(|(&cid, kind)| (cid, vector_widen(kind.as_deref()))).collect();
            set_widens(self.ast, site, Some(&pairs));
            set_widen_site(self.ast, node, Some(site));
        }
    }

    /// Report the per-instance conversions some instance cannot be given
    /// (`_report_unhandled_widens`).
    pub(super) fn report_unhandled_widens(&mut self) -> R<()> {
        let mut placed: HashSet<NodeId> = HashSet::new();
        let mut varying_sites: HashSet<i64> = HashSet::new();
        for definitions in self.defs.values() {
            for &definition in definitions {
                for site in get_varying(self.ast, definition).unwrap_or_default() {
                    placed.insert(site);
                    if let Some(nid) = facts::nid(self.ast, site) {
                        varying_sites.insert(nid);
                    }
                }
            }
        }
        let mut unvectored: HashSet<i64> = self.unvectored.clone();
        let mut frontier: Vec<i64> = unvectored.iter().copied().collect();
        while let Some(caller) = frontier.pop() {
            for site_nid in &varying_sites {
                let callee = self.call_ctx.get(site_nid).and_then(|seen| seen.get(&caller));
                if let Some(&callee) = callee
                    && unvectored.insert(callee)
                {
                    frontier.push(callee);
                }
            }
        }
        for (nid, node) in self.widen_nodes.clone() {
            let Some(site) = get_widen_site(self.ast, node) else { continue };
            if !placed.contains(&site) {
                set_widen_site(self.ast, node, None);
            } else {
                let unvectored_needs = get_widens(self.ast, site)
                    .unwrap_or_default()
                    .iter()
                    .any(|(cid, kind)| kind.is_some() && unvectored.contains(cid));
                let class_needs = self.widens.get(&nid).is_some_and(|seen| {
                    seen.values()
                        .any(|kind| kind.as_deref().is_some_and(|kind| class_of(kind).is_some()))
                });
                if !(unvectored_needs || class_needs) {
                    continue;
                }
            }
            let origin = self.unknown("context-dependent-widen", node, "");
            self.diag(
                "the implicit conversion of a value differs per call context, and not every \
                 instance can be given its own",
                node,
                Some(origin),
                "annotate the parameters so every call site passes the same types",
            );
        }
        Ok(())
    }

    /// Take every overload pin away from a module whose containers cannot be trusted
    /// (`_suppress_pins`).
    pub(super) fn suppress_pins(&mut self, tree: NodeId) {
        if self.pins_suppressed.is_none() {
            return;
        }
        for node in walk(self.ast, tree) {
            match self.kind(node) {
                Kind::Call { .. } => {
                    set_pin(self.ast, node, None);
                    set_pins(self.ast, node, None);
                    set_vector(self.ast, node, None);
                    if let Some(lens) = get_series_lens(self.ast, node) {
                        set_series_len(self.ast, node, Some(lens.iter().any(|&(_, flag)| flag)));
                        set_series_lens(self.ast, node, None);
                    }
                }
                Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                    set_varying(self.ast, node, None);
                }
                _ => {}
            }
        }
        for site in &mut self.calls {
            site.pin = None;
        }
    }

    /// Work out what a shared body resolves per instance, and stamp the wire
    /// (`_stamp_instance_vectors`).
    pub(super) fn stamp_instance_vectors(&mut self, tree: NodeId) {
        let targets: IndexMap<String, NodeId> = self
            .defs
            .iter()
            .filter(|(key, defs)| !self.overload_groups.contains(*key) && !defs.is_empty())
            .map(|(key, defs)| (key.clone(), *defs.last().expect("a definition")))
            .collect();
        let impls: Vec<NodeId> = self
            .overload_groups
            .iter()
            .flat_map(|key| self.defs.get(key).cloned().unwrap_or_default())
            .collect();
        let mut cids: HashMap<String, Vec<i64>> = HashMap::new();
        let mut impl_cids: HashMap<i64, Vec<i64>> = HashMap::new();
        for (memo_key, &index) in &self.contexts {
            let result = &self.results[index];
            cids.entry(result.key.clone()).or_default().push(result.cid);
            if self.overload_groups.contains(&result.key)
                && let Some(def) = memo_key.def
            {
                impl_cids.entry(def).or_default().push(result.cid);
            }
        }
        let own: IndexMap<String, Vec<NodeId>> =
            targets.iter().map(|(key, &node)| (key.clone(), own_calls(self.ast, node))).collect();
        let mut owner: HashMap<i64, String> = HashMap::new();
        for (key, calls) in &own {
            for &call in calls {
                if let Some(nid) = facts::nid(self.ast, call) {
                    owner.insert(nid, key.clone());
                }
            }
        }
        let mut impl_owner: HashMap<i64, i64> = HashMap::new();
        for &imp in &impls {
            let Some(impl_nid) = facts::nid(self.ast, imp) else { continue };
            for call in own_calls(self.ast, imp) {
                if let Some(nid) = facts::nid(self.ast, call) {
                    impl_owner.insert(nid, impl_nid);
                }
            }
        }
        let ast: &pyne_ast::Ast = self.ast;
        let varying: IndexMap<String, Vec<NodeId>> = own
            .iter()
            .map(|(key, calls)| {
                (
                    key.clone(),
                    calls.iter().copied().filter(|&call| per_context(ast, call)).collect(),
                )
            })
            .collect();
        let mut vectors = Vectors {
            varying,
            memo: HashMap::new(),
            widens: HashMap::new(),
            series_len: &self.series_len,
            pins: &self.pins,
            callee_key: &self.callee_key,
            call_ctx: &self.call_ctx,
        };
        let empty = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for (key, calls) in &own {
                let mut found: Vec<NodeId> = Vec::new();
                for &call in calls {
                    if per_context(ast, call) {
                        found.push(call);
                        continue;
                    }
                    let mut seen: HashSet<Entry> = HashSet::new();
                    for &cid in cids.get(key).map(Vec::as_slice).unwrap_or(&[]) {
                        seen.insert(vectors.entry_of(ast, call, cid, &empty));
                    }
                    if seen.len() > 1 {
                        found.push(call);
                    }
                }
                if found != vectors.varying[key] {
                    vectors.varying.insert(key.clone(), found);
                    vectors.memo.clear();
                    changed = true;
                }
            }
        }
        let mut stamps: Vec<(NodeId, Option<Vec<NodeId>>)> = Vec::new();
        for (key, sites) in &vectors.varying {
            stamps.push((targets[key], if sites.is_empty() { None } else { Some(sites.clone()) }));
        }
        let any_varying = vectors.varying.values().any(|sites| !sites.is_empty());
        let mut vector_stamps: Vec<(NodeId, Value)> = Vec::new();
        if any_varying {
            let per_instance: HashSet<NodeId> =
                vectors.varying.values().flatten().copied().collect();
            for node in walk(ast, tree) {
                if !matches!(ast.kind(node), Kind::Call { .. }) || per_instance.contains(&node) {
                    continue;
                }
                let nid = facts::nid(ast, node);
                let callee = nid.and_then(|nid| self.callee_key.get(&nid)).cloned();
                let Some(callee) = callee else { continue };
                let width = match vectors.varying.get(&callee) {
                    Some(sites) if !sites.is_empty() => sites.len(),
                    _ => continue,
                };
                let nid = nid.expect("a numbered call");
                let callers: Vec<i64> = if let Some(key) = owner.get(&nid) {
                    cids.get(key).cloned().unwrap_or_default()
                } else if let Some(impl_nid) = impl_owner.get(&nid) {
                    impl_cids.get(impl_nid).cloned().unwrap_or_default()
                } else {
                    vec![0]
                };
                let mut seen: Vec<Entry> = Vec::new();
                for cid in callers {
                    let entry = vectors.entry_of(ast, node, cid, &empty);
                    if !seen.contains(&entry) {
                        seen.push(entry);
                    }
                }
                if seen.len() != 1 {
                    continue;
                }
                let vector = seen.pop().expect("one vector");
                let all_none = Entry::Vector(Rc::new(vec![Entry::None; width]));
                if vector != Entry::None && vector != all_none {
                    vector_stamps.push((node, vector.value()));
                }
            }
        }
        for (node, sites) in stamps {
            set_varying(self.ast, node, sites.as_deref());
        }
        for (node, vector) in vector_stamps {
            set_vector(self.ast, node, Some(vector));
        }
    }
}

#[allow(dead_code)]
fn int(value: i64) -> Value {
    Value::Int(Int::Small(value))
}
