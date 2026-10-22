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
pub mod dynamic_default_guard;
pub mod fdlibm;
pub mod node_ids;
pub mod phase;
pub mod pine_qualifier;
pub mod pine_type_rules;
pub mod pine_type_table;
pub mod sha1;
pub mod slot_layout;
pub mod steps;
pub mod unparse;

use std::cell::RefCell;
use std::fmt;

pub use pyne_ast::Unsupported;
use pyne_ast::{Ast, NodeId};
use slot_layout::ModuleLayout;

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
        Ok(data::module_properties().get(module).is_some_and(|members| members.contains_key(name)))
    }

    /// The `"type"` of a module's `module_properties.json` entry for a member, `None`
    /// when the module's entry does not name it
    /// (`module_property.module_properties()[module].get(name)["type"]`).
    fn module_property_type(
        &self,
        module: &str,
        name: &str,
    ) -> Result<Option<String>, Unsupported> {
        Ok(data::module_properties().get(module).and_then(|members| members.get(name)).cloned())
    }

    /// The `__all__` of a lib module (`lib_registry.lib_star_exports().get(module)`).
    fn star_exports(&self, module: &str) -> Result<Option<Vec<String>>, Unsupported> {
        Ok(data::star_exports().get(module).cloned())
    }

    /// A lib name's registry entry (`lib_registry.lib_types().get(name)`), `None` when
    /// the registry does not list it.
    fn lib_name(&self, name: &str) -> Result<Option<data::LibName>, Unsupported> {
        Ok(data::lib_names().get(name).cloned())
    }

    /// The `.py` source file of an imported module, found without executing anything
    /// (`export_capture.module_origin(name, importer, level)`); `None` when the module
    /// has no such source. `importer` is the importing module's path, read for a
    /// relative import (`level` > 0) and empty otherwise.
    fn module_origin(
        &self,
        name: &str,
        importer: &str,
        level: i64,
    ) -> Result<Option<String>, Unsupported> {
        let _ = (importer, level);
        Err(Unsupported(format!("no host to locate module {name}")))
    }

    /// A source file's bytes with the fingerprint they belong to, and its resolved
    /// path (`source_read.stable_source(path)` and `path.resolve()`); `None` when
    /// no such pair could be had.
    fn read_source(&self, path: &str) -> Result<Option<SourceRead>, Unsupported> {
        Err(Unsupported(format!("no host to read {path}")))
    }

    /// The interface an imported module publishes, from its resolved source path
    /// (`module_interface.lookup(path, analyse, pipeline_hash)`); `None` when there is
    /// none to be had. The host may analyse the module to answer.
    fn interface(
        &self,
        path: &str,
    ) -> Result<Option<pine_type_table::ModuleInterface>, Unsupported> {
        Err(Unsupported(format!("no host for the interface of {path}")))
    }

    /// Whether reaching a module closes an import cycle: its analysis is on the stack
    /// (`module_interface.closes_cycle(path)`).
    fn closes_cycle(&self, path: &str) -> Result<bool, Unsupported> {
        Err(Unsupported(format!("no host to ask whether {path} closes a cycle")))
    }

    /// Mark a module as being analysed until [`Host::end_analysis`]
    /// (`module_interface.analysing_scope(path)` entered); a host without
    /// cross-module analysis marks nothing.
    fn begin_analysis(&self, path: &str) -> Result<(), Unsupported> {
        let _ = path;
        Ok(())
    }

    /// Leave the analysing scope the last [`Host::begin_analysis`] entered.
    fn end_analysis(&self) -> Result<(), Unsupported> {
        Ok(())
    }

    /// The `.py` source a dotted module name resolves to through `find_spec`
    /// (`pine_type_infer.module_spec_origin(dotted)`); `None` when it is no readable
    /// Python module.
    fn module_spec_origin(&self, dotted: &str) -> Result<Option<String>, Unsupported> {
        Err(Unsupported(format!("no host to find module {dotted}")))
    }

    /// The source of a module below a package directory
    /// (`pine_type_infer.package_member(Path(directory), dotted)`).
    fn package_member(&self, directory: &str, dotted: &str) -> Result<Option<String>, Unsupported> {
        Err(Unsupported(format!("no host to find {dotted} below {directory}")))
    }

    /// Whether a module belongs to Python's standard library
    /// (`utils.stdlib_checker.is_stdlib(module)`).
    fn is_stdlib(&self, module: &str) -> Result<bool, Unsupported> {
        Err(Unsupported(format!("no host to ask whether {module} is stdlib")))
    }
}

/// A source file as [`Host::read_source`] read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRead {
    /// The file's resolved path (`str(path.resolve())`)
    pub resolved: String,
    /// The bytes read
    pub data: Vec<u8>,
    /// `st_mtime_ns` of the file the bytes were read from
    pub mtime_ns: i64,
    /// `st_size` of the file the bytes were read from
    pub size: i64,
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
    /// Whether the module keeps Pine's three-state bool (`FloatTolerance`)
    pub na_bool: bool,
    /// Whether the rules of a phase run as separate passes instead of in one traversal
    pub sequential: bool,
    /// Answers what the tree does not say
    pub host: &'h dyn Host,
    /// The shared slot allocator of the lowering half (see [`slot_layout`]); the
    /// steps of [`SLOT_LAYOUT_STEPS`] read and fill it
    pub slot_layout: RefCell<ModuleLayout>,
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
    Step { name: "ExportCapture", run: steps::export_capture::run },
    Step { name: "SecurityDrawings", run: steps::security_drawings::run },
    Step { name: "ConstFold", run: steps::const_fold::run },
    Step { name: "DynamicDefault", run: steps::dynamic_default::run },
    Step { name: "InlineSeriesHoist", run: steps::inline_series_hoist::run },
    Step { name: "PineTruthiness", run: steps::pine_truthiness::run },
    Step { name: "SecurityInstantiation", run: steps::security_instantiation::run },
    Step { name: "Security", run: steps::security::run },
    Step { name: "PersistentSeries", run: steps::persistent_series::run },
    Step { name: "LibrarySeries", run: steps::lib_series::run },
    Step { name: "ModuleProperty", run: steps::module_property::run },
    Step { name: "TaVariableHoist", run: steps::ta_variable_hoist::run },
    Step { name: "ClosureArguments", run: steps::closure_arguments::run },
    Step { name: "PineType", run: steps::pine_type::run },
    Step { name: "ImplicitConversion", run: steps::implicit_conversion::run },
    Step { name: "SecurityDefault", run: steps::security_default::run },
    Step { name: "ExportOnce", run: steps::export_once::run },
    Step { name: "UnusedSeriesDetector", run: steps::unused_series_detector::run },
    Step { name: "Series", run: steps::series::run },
    Step { name: "VerifyClosedShift", run: steps::security_closed_shift_check::run },
    Step { name: "Persistent", run: steps::persistent::run },
    Step { name: "ScriptRequirements", run: steps::script_requirements::run },
    Step { name: "Input", run: steps::input::run },
    // The rules of the phases, each on its own (sequential mode)
    Step { name: "SafeConvert", run: steps::safe_convert::run },
    Step { name: "SafeDivision", run: steps::safe_division::run },
    Step { name: "FloatTolerance", run: steps::float_tolerance::run },
    Step { name: "FixLocations", run: steps::locations::run },
];

/// The steps whose port reads or fills the run's [`Context::slot_layout`]. A host
/// hands the layout of its pipeline over before a run with one of these steps, and
/// takes back the state they left.
pub static SLOT_LAYOUT_STEPS: &[&str] = &["Series", "VerifyClosedShift", "Persistent"];

/// Whether a run of these steps reads or fills the slot layout.
pub fn uses_slot_layout(names: &[&str]) -> bool {
    names.iter().any(|name| SLOT_LAYOUT_STEPS.contains(name))
}

/// Every phase with a native port: its rules run in one traversal under the
/// phase's name. Each rule is a step of [`STEPS`] as well, ported from its own
/// module, so a host runs the phase natively only while every rule is current.
pub static PHASES: &[phase::Phase] = &[phase::Phase {
    name: "Expression",
    rules: &[steps::safe_convert::RULE, steps::safe_division::RULE, steps::float_tolerance::RULE],
}];

/// The native port of a phase, if there is one.
pub fn phase(name: &str) -> Option<&'static phase::Phase> {
    PHASES.iter().find(|phase| phase.name == name)
}

/// The steps whose port reads the module's
/// [`PineTypeTable`](pine_type_table::PineTypeTable). A host hands the table of a
/// run over to the tree natively ([`pyne_ast::Value::Native`]) when one of these
/// steps is in the run, and through untouched otherwise.
pub static TYPE_TABLE_READERS: &[&str] = &[];

/// Whether a run of these steps needs the module's type table natively.
pub fn reads_type_table(names: &[&str]) -> bool {
    names.iter().any(|name| TYPE_TABLE_READERS.contains(name))
}

/// The native port of a step, if there is one.
pub fn step(name: &str) -> Option<&'static Step> {
    STEPS.iter().find(|step| step.name == name)
}

/// Run named steps and phases in order.
pub fn run_steps(
    ast: &mut Ast,
    mut module: NodeId,
    names: &[&str],
    ctx: &Context<'_>,
) -> Result<NodeId, Error> {
    for name in names {
        module = match (step(name), phase(name)) {
            (Some(step), _) => (step.run)(ast, module, ctx)?,
            (None, Some(phase)) => phase.run(ast, module, ctx)?,
            (None, None) => return Err(Error::Unsupported(format!("no native step {name}"))),
        };
    }
    Ok(module)
}
