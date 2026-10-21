//! The data files of `pynecore/transformers`, compiled in.
//!
//! They are part of a step's source closure (see `transformers/native.py`): a
//! regenerated file changes the digest of every step that reads it, so a binary
//! built with the old copy is out of use until it is rebuilt.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde_json::Value;

/// `module_properties.json`: module -> its member names (`module_property.module_properties`).
pub fn module_properties() -> &'static HashMap<String, HashSet<String>> {
    static REGISTRY: OnceLock<HashMap<String, HashSet<String>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let text = include_str!("../../../../src/pynecore/transformers/module_properties.json");
        let document: HashMap<String, HashMap<String, Value>> =
            serde_json::from_str(text).expect("module_properties.json");
        document
            .into_iter()
            .map(|(module, members)| (module, members.into_keys().collect()))
            .collect()
    })
}
