//! Port of `transformers/slot_layout.py`: the shared slot allocator of the lowering half.
//!
//! Every step that contributes per-instance state (persistent variables, series,
//! isolated call sites) allocates slots from the one [`ModuleLayout`] of the run
//! ([`crate::Context::slot_layout`]) while walking the module; the Python
//! `layout_emit.apply_layout` then materializes it. The layout is the
//! `PipelineContext.slot_layout` of the Python pipeline, field for field: a host
//! hands the Python object's state over before a run of steps that use it, and
//! writes the state the steps left back into the same object afterwards.
//!
//! Scope identifiers join the function-name path with the middle-dot separator
//! (`main`, `main·helper`); repeated definitions of one name within a scope
//! (`overload` implementations) are disambiguated with an ordinal suffix
//! (`highest·2`). The hidden state parameter is `__state__`, or the
//! scope-qualified `__state·main__` for a function that contains nested
//! definitions.

use indexmap::IndexMap;
use pyne_ast::walk::{child_statements, walk_statements};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

/// The hidden state parameter of a scope without nested definitions.
pub const DEFAULT_STATE_PARAM: &str = "__state__";

/// What a slot holds (`_Slot.kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    Var,
    Flag,
    Series,
    Child,
    Anchor,
    Pin,
}

impl SlotKind {
    /// The Python spelling.
    pub fn name(self) -> &'static str {
        match self {
            SlotKind::Var => "var",
            SlotKind::Flag => "flag",
            SlotKind::Series => "series",
            SlotKind::Child => "child",
            SlotKind::Anchor => "anchor",
            SlotKind::Pin => "pin",
        }
    }

    pub fn from_name(name: &str) -> Option<SlotKind> {
        Some(match name {
            "var" => SlotKind::Var,
            "flag" => SlotKind::Flag,
            "series" => SlotKind::Series,
            "child" => SlotKind::Child,
            "anchor" => SlotKind::Anchor,
            "pin" => SlotKind::Pin,
            _ => return None,
        })
    }
}

/// One slot of a scope's state vector (`_Slot`).
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub index: usize,
    pub kind: SlotKind,
    /// Debug name for the layout `names` tuple
    pub name: String,
    /// Template expression for the layout `init` tuple
    pub init: NodeId,
    /// Series slots only
    pub max_bars_back: Option<NodeId>,
    /// Series slots only: element type name (`float` / `int` / `bool`) or `None`
    pub series_elem: Option<String>,
    /// Var slots only: the declared scalar type name or `None`
    pub var_elem: Option<String>,
    /// Child and anchor slots only
    pub call_id: Option<String>,
    /// Child slots only
    pub in_loop: bool,
    /// Var and flag slots only
    pub varip: bool,
}

impl Slot {
    fn new(index: usize, kind: SlotKind, name: String, init: NodeId) -> Slot {
        Slot {
            index,
            kind,
            name,
            init,
            max_bars_back: None,
            series_elem: None,
            var_elem: None,
            call_id: None,
            in_loop: false,
            varip: false,
        }
    }
}

/// `ast.Constant(value=...)`, without location.
fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

/// Slot table of one scope (one function definition) (`ScopeLayout`).
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeLayout {
    pub scope: String,
    pub slots: Vec<Slot>,
    pub state_param: String,
    /// Slot holding the instance vector, for a scope whose body resolves
    /// something per instance; `None` for every other scope
    pub pin_slot: Option<usize>,
}

impl ScopeLayout {
    pub fn new(scope: &str) -> ScopeLayout {
        ScopeLayout {
            scope: scope.to_string(),
            slots: Vec::new(),
            state_param: DEFAULT_STATE_PARAM.to_string(),
            pin_slot: None,
        }
    }

    fn add(&mut self, slot: Slot) -> usize {
        let index = slot.index;
        self.slots.push(slot);
        index
    }

    /// Allocate a slot for a persistent variable (`add_var`).
    pub fn add_var(&mut self, name: &str, init: NodeId, varip: bool, elem: Option<&str>) -> usize {
        let mut slot = Slot::new(self.slots.len(), SlotKind::Var, name.to_string(), init);
        slot.varip = varip;
        slot.var_elem = elem.map(str::to_string);
        self.add(slot)
    }

    /// Allocate a lazy-init flag slot, init `False` (`add_flag`).
    pub fn add_flag(&mut self, ast: &mut Ast, name: &str, varip: bool) -> usize {
        let init = constant(ast, Constant::Bool(false));
        let mut slot = Slot::new(self.slots.len(), SlotKind::Flag, format!("{name}·flag"), init);
        slot.varip = varip;
        self.add(slot)
    }

    /// Allocate a series slot (`add_series`).
    pub fn add_series(
        &mut self,
        ast: &mut Ast,
        name: &str,
        max_bars_back: NodeId,
        elem: Option<&str>,
    ) -> usize {
        let init = constant(ast, Constant::None);
        let mut slot = Slot::new(self.slots.len(), SlotKind::Series, name.to_string(), init);
        slot.max_bars_back = Some(max_bars_back);
        slot.series_elem = elem.map(str::to_string);
        self.add(slot)
    }

    /// Allocate the scope's instance-vector slot, init the all-`None` vector (`add_pin`).
    pub fn add_pin(&mut self, ast: &mut Ast, count: usize) -> usize {
        let elts = (0..count).map(|_| constant(ast, Constant::None)).collect();
        let init = ast.add(Kind::Tuple { elts, ctx: ExprContext::Load });
        let index = self.add(Slot::new(self.slots.len(), SlotKind::Pin, "·pins".into(), init));
        self.pin_slot = Some(index);
        index
    }

    /// Allocate a child slot for an isolated call site (`add_child`).
    pub fn add_child(&mut self, ast: &mut Ast, call_id: &str, in_loop: bool) -> usize {
        self.add_call_site(ast, SlotKind::Child, call_id, in_loop)
    }

    /// Allocate an anchor slot for a uniform-path call site (`add_anchor`).
    pub fn add_anchor(&mut self, ast: &mut Ast, call_id: &str, in_loop: bool) -> usize {
        self.add_call_site(ast, SlotKind::Anchor, call_id, in_loop)
    }

    fn add_call_site(
        &mut self,
        ast: &mut Ast,
        kind: SlotKind,
        call_id: &str,
        in_loop: bool,
    ) -> usize {
        let init = constant(ast, Constant::None);
        let mut slot = Slot::new(self.slots.len(), kind, call_id.to_string(), init);
        slot.call_id = Some(call_id.to_string());
        slot.in_loop = in_loop;
        self.add(slot)
    }
}

/// Slot layouts of every scope in one module, shared by the steps (`ModuleLayout`);
/// the default is `ModuleLayout()`, a module whose series are not compacted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModuleLayout {
    /// Scope id -> its layout, in creation order
    pub scopes: IndexMap<String, ScopeLayout>,
    /// Definition -> its scope segment, for the module in `tree` (`_segments`)
    pub segments: IndexMap<NodeId, String>,
    /// Definition -> whether a `FunctionDef` stands anywhere inside it (`_nests`)
    pub nests: IndexMap<NodeId, bool>,
    /// The module the two maps above were computed for (`_tree`)
    pub tree: Option<NodeId>,
    /// True for `@pyne lib` modules, whose series are na-compacted windows
    pub compacted_series: bool,
}

impl ModuleLayout {
    pub fn new(compacted_series: bool) -> ModuleLayout {
        ModuleLayout {
            scopes: IndexMap::new(),
            segments: IndexMap::new(),
            nests: IndexMap::new(),
            tree: None,
            compacted_series,
        }
    }

    /// Build the definition -> scope-segment mapping for a module, once per module
    /// (`assign_scope_ids`).
    pub fn assign_scope_ids(&mut self, ast: &Ast, tree: NodeId) {
        if self.tree == Some(tree) {
            return;
        }
        self.tree = Some(tree);
        let (segments, nests) = scan_scopes(ast, tree);
        self.segments = segments;
        self.nests = nests;
    }

    /// Disambiguated scope segment of a function definition (`scope_segment`).
    pub fn scope_segment(&self, ast: &Ast, node: NodeId) -> String {
        match self.segments.get(&node) {
            Some(segment) => segment.clone(),
            None => def_name(ast, node).to_string(),
        }
    }

    /// Whether a `FunctionDef` stands anywhere inside a definition (`nests_definition`).
    pub fn nests_definition(&self, ast: &Ast, node: NodeId) -> bool {
        match self.nests.get(&node) {
            Some(&nests) => nests,
            None => walk_statements(ast, node)
                .into_iter()
                .any(|child| child != node && matches!(ast.kind(child), Kind::FunctionDef { .. })),
        }
    }

    /// The layout of a scope, created on demand (`scope`).
    pub fn scope(&mut self, scope_id: &str) -> &mut ScopeLayout {
        self.scopes.entry(scope_id.to_string()).or_insert_with(|| ScopeLayout::new(scope_id))
    }

    /// Whether a scope has any state slot (`state_carrying`).
    pub fn state_carrying(&self, scope_id: &str) -> bool {
        self.scopes.get(scope_id).is_some_and(|scope| !scope.slots.is_empty())
    }

    /// Name of a scope's hidden state parameter (`state_param`).
    pub fn state_param(&self, scope_id: &str) -> &str {
        match self.scopes.get(scope_id) {
            Some(scope) => &scope.state_param,
            None => DEFAULT_STATE_PARAM,
        }
    }
}

/// The scope layout of a function definition, qualifying the state parameter name
/// when the function contains nested definitions (`scope_for_function`).
pub fn scope_for_function<'l>(
    layout: &'l mut ModuleLayout,
    ast: &Ast,
    scope_id: &str,
    node: NodeId,
) -> &'l mut ScopeLayout {
    let nests = layout.nests_definition(ast, node);
    let scope = layout.scope(scope_id);
    if nests {
        scope.state_param = format!("__state·{scope_id}__");
    }
    scope
}

/// The scope segment of every function definition in a module (`collect_scope_segments`).
pub fn collect_scope_segments(ast: &Ast, tree: NodeId) -> IndexMap<NodeId, String> {
    scan_scopes(ast, tree).0
}

fn def_name(ast: &Ast, node: NodeId) -> &str {
    match ast.kind(node) {
        Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => name,
        other => panic!("a scope segment of a {}", other.name()),
    }
}

fn is_def(kind: &Kind) -> bool {
    matches!(kind, Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

/// The function definitions belonging to one scope in source order, not entering a
/// nested definition (`_scope_defs`).
fn scope_defs(ast: &Ast, scope_node: NodeId, out: &mut Vec<NodeId>) {
    for child in child_statements(ast, scope_node) {
        if is_def(ast.kind(child)) {
            out.push(child);
        } else {
            scope_defs(ast, child, out);
        }
    }
}

type Scan = (IndexMap<NodeId, String>, IndexMap<NodeId, bool>);

/// The scope segment of every function definition, and whether it nests one
/// (`_scan_scopes`).
fn scan_scopes(ast: &Ast, tree: NodeId) -> Scan {
    fn assign(ast: &Ast, scope_node: NodeId, scan: &mut Scan) -> bool {
        let mut counts: IndexMap<&str, usize> = IndexMap::new();
        let mut nested = false;
        let mut defs = Vec::new();
        scope_defs(ast, scope_node, &mut defs);
        for func in defs {
            let name = def_name(ast, func);
            let n = counts.get(name).copied().unwrap_or(0) + 1;
            counts.insert(name, n);
            let segment = if n == 1 { name.to_string() } else { format!("{name}·{n}") };
            scan.0.insert(func, segment);
            let inner = assign(ast, func, scan);
            scan.1.insert(func, inner);
            nested = nested || inner || matches!(ast.kind(func), Kind::FunctionDef { .. });
        }
        nested
    }

    let mut scan = (IndexMap::new(), IndexMap::new());
    assign(ast, tree, &mut scan);
    scan
}
