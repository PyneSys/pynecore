//! PyneCore's AST transform pipeline, native.
//!
//! The Python pipeline (`pynecore/transformers/pipeline.py`) is the reference;
//! every step here is a port of the Python step of the same name and must
//! produce the same tree for the same input, which the host checks
//! (`PYNE_AST_BACKEND=verify`). A step meeting something its port does not
//! cover answers [`Unsupported`], and the host runs the Python step instead.

pub mod steps;

pub use pyne_ast::Unsupported;
use pyne_ast::{Ast, NodeId};

/// What the steps of one module's transform share besides the tree: the
/// counterpart of `pipeline.PipelineContext`.
#[derive(Clone, Debug, Default)]
pub struct Context {
    /// Source path; the script / lib profile is picked from it
    pub path: String,
    /// The module's mode word, `None` for a hand-written script
    pub pyne_mode: Option<String>,
    /// Full module source, for error locations
    pub source: String,
    /// Whether the module is user code rather than one of pynecore's own lib modules
    pub user_code: bool,
}

/// One step: transforms the module and returns the (possibly replaced) module node.
pub type StepFn = fn(&mut Ast, NodeId, &Context) -> Result<NodeId, Unsupported>;

/// A native step, by the name `pipeline.py` declares it under.
pub struct Step {
    pub name: &'static str,
    pub run: StepFn,
}

/// Per step, the Python module it ports and the digest of that module's sources
/// at the time of the port (`ported.json`, maintained by `rust/tools/ported.py`).
/// The host compares the digests with its sources and runs a step natively only
/// while they match.
pub const PORTED: &str = include_str!("../ported.json");

/// Every step with a native port.
pub static STEPS: &[Step] = &[
    Step { name: "ImportLifter", run: steps::import_lifter::run },
    Step { name: "TypeCheckingStripper", run: steps::type_checking_stripper::run },
];

/// The native port of a step, if there is one.
pub fn step(name: &str) -> Option<&'static Step> {
    STEPS.iter().find(|step| step.name == name)
}

/// Run named steps in order.
pub fn run_steps(
    ast: &mut Ast,
    mut module: NodeId,
    names: &[&str],
    ctx: &Context,
) -> Result<NodeId, Unsupported> {
    for name in names {
        let step = step(name).ok_or_else(|| Unsupported(format!("no native step {name}")))?;
        module = (step.run)(ast, module, ctx)?;
    }
    Ok(module)
}
