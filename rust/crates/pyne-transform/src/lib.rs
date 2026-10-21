//! PyneCore's AST transform pipeline, native.
//!
//! The Python pipeline (`pynecore/transformers/pipeline.py`) is the reference;
//! every step here is a port of the Python step of the same name and must
//! produce the same tree for the same input, which the host checks
//! (`PYNE_AST_BACKEND=verify`). A step meeting something its port does not
//! cover answers [`Error::Unsupported`], and the host runs the Python step
//! instead; a step that finds the module invalid answers [`Error::Rejected`], and
//! the host lets the Python step raise the error the author sees (a browser host
//! shows the message).

pub mod data;
pub mod steps;

use std::fmt;

pub use pyne_ast::Unsupported;
use pyne_ast::{Ast, NodeId};

/// Why a native step did not hand back a tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A construct the port does not cover: the Python step runs instead
    Unsupported(String),
    /// The module is invalid where the Python step raises (a `SyntaxError` on
    /// a plot call outside the entry, a wildcard import without `__all__`, ...)
    Rejected(String),
}

impl From<Unsupported> for Error {
    fn from(err: Unsupported) -> Error {
        Error::Unsupported(err.0)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(message) => write!(f, "unsupported: {message}"),
            Error::Rejected(message) => write!(f, "rejected: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// What a step needs to know that the tree does not say, answered by whoever runs
/// the pipeline: the Python host from the live interpreter, a browser host from
/// data it was handed. Each question is asked exactly where the Python step reads
/// the live object, so a host that imports (Python) does so at the same moment.
///
/// The registry questions default to the data files compiled in (see [`data`]);
/// the Python host answers them from the registries the Python steps read, so a
/// registry a test patches reaches both backends alike.
pub trait Host {
    /// Import a library module whose alias shadows a builtin namespace
    /// (`builtin_shadow.import_library`); false when it cannot be imported.
    fn import_library(&self, module: &str) -> Result<bool, Unsupported> {
        let _ = module;
        Ok(false)
    }

    /// Whether an imported library module serves a name (`builtin_shadow.library_serves`).
    fn library_serves(&self, module: &str, attr: &str) -> Result<bool, Unsupported> {
        Err(Unsupported(format!("no host to ask whether {module} serves {attr}")))
    }

    /// Whether `module_properties.json` has an entry for a module
    /// (`module in module_property.module_properties()`).
    fn has_module_properties(&self, module: &str) -> Result<bool, Unsupported> {
        Ok(data::module_properties().contains_key(module))
    }

    /// Whether a module's `module_properties.json` entry names a member
    /// (`name in module_property.module_properties()[module]`).
    fn module_property(&self, module: &str, name: &str) -> Result<bool, Unsupported> {
        Ok(data::module_properties().get(module).is_some_and(|members| members.contains(name)))
    }

    /// The `__all__` of a lib module (`lib_registry.lib_star_exports().get(module)`).
    fn star_exports(&self, module: &str) -> Result<Option<Vec<String>>, Unsupported> {
        Ok(data::star_exports().get(module).cloned())
    }
}

/// A host that knows nothing: no library can be imported.
pub struct NoHost;

impl Host for NoHost {}

/// What the steps of one module's transform share besides the tree: the
/// counterpart of `pipeline.PipelineContext`.
pub struct Context<'h> {
    /// Source path; the script / lib profile is picked from it
    pub path: String,
    /// The module's mode word, `None` for a hand-written script
    pub pyne_mode: Option<String>,
    /// Full module source, for error locations
    pub source: String,
    /// Whether the module is user code rather than one of pynecore's own lib modules
    pub user_code: bool,
    /// Answers what the tree does not say
    pub host: &'h dyn Host,
}

/// One step: transforms the module and returns the (possibly replaced) module node.
pub type StepFn = fn(&mut Ast, NodeId, &Context<'_>) -> Result<NodeId, Error>;

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
    Step { name: "TypeErasure", run: steps::type_erasure::run },
    Step { name: "BuiltinShadow", run: steps::builtin_shadow::run },
    Step { name: "ImportNormalizer", run: steps::import_normalizer::run },
    Step { name: "PlotScope", run: steps::plot_scope::run },
    Step { name: "OuterWrite", run: steps::outer_write::run },
    Step { name: "SecurityDrawings", run: steps::security_drawings::run },
    Step { name: "DynamicDefault", run: steps::dynamic_default::run },
    Step { name: "InlineSeriesHoist", run: steps::inline_series_hoist::run },
    Step { name: "PineTruthiness", run: steps::pine_truthiness::run },
    Step { name: "PersistentSeries", run: steps::persistent_series::run },
    Step { name: "LibrarySeries", run: steps::lib_series::run },
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
    ctx: &Context<'_>,
) -> Result<NodeId, Error> {
    for name in names {
        let step =
            step(name).ok_or_else(|| Error::Unsupported(format!("no native step {name}")))?;
        module = (step.run)(ast, module, ctx)?;
    }
    Ok(module)
}
