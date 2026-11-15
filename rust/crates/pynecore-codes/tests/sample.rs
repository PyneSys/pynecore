//! The generator's Rust output for `scripts/errorcodes/sample.toml`, compiled against the
//! crate's types, and the lookups over PyneCore's own codes.

use pynecore_codes::{Code, Fix, Severity};

mod sample {
    include!("sample/registry.rs");
}

#[test]
fn sample_codes_are_generated_sorted_without_the_retired_one() {
    let codes: Vec<&str> = sample::ALL.iter().map(|c| c.code).collect();
    assert_eq!(codes, ["L0001", "S0001"]);
    assert_eq!(sample::NBSP_IN_VERSION_PRAGMA.severity, Severity::Warning);
}

#[test]
fn render_fills_parameters_in_order_and_unescapes_braces() {
    let code: &Code = &sample::UNEXPECTED_TOKEN;
    assert_eq!(code.params, ["expected", "found"]);
    assert_eq!(code.render(&["`)`", "`,`"]), "Expected `)`, found `,`; `{` opens nothing here.");
    assert_eq!(
        sample::NBSP_IN_VERSION_PRAGMA.render(&[]),
        "The version pragma contains a non-breaking space."
    );
}

#[test]
fn a_fix_renders_with_the_message_arguments() {
    let code: &Code = &sample::UNEXPECTED_TOKEN;
    assert_eq!(code.fix(&["`)`", "`,`"]), Some(("Write `)`".to_string(), "`)`".to_string())));
    assert_eq!(sample::NBSP_IN_VERSION_PRAGMA.fix(&[]), None);
}

#[test]
fn the_sample_has_no_run_time_code() {
    assert!(sample::RUNTIME.is_empty());
}

#[test]
fn lookup_finds_only_registered_codes() {
    assert_eq!(
        pynecore_codes::lookup("R0001"),
        pynecore_codes::ALL.iter().copied().find(|c| c.code == "R0001")
    );
    assert_eq!(pynecore_codes::lookup("S0001"), None);
}

#[test]
fn a_run_time_code_is_found_by_its_number() {
    assert_eq!(pynecore_codes::INDEX_OUT_OF_RANGE.id(), 820_010);
    assert_eq!(pynecore_codes::runtime(820_010), Some(&pynecore_codes::INDEX_OUT_OF_RANGE));
    assert!(pynecore_codes::RUNTIME.iter().all(|c| pynecore_codes::runtime(c.id()) == Some(*c)));
    assert!(pynecore_codes::RUNTIME.iter().all(|c| matches!(&c.code[..1], "R" | "I")));
    assert_eq!(pynecore_codes::runtime(pynecore_codes::EDGE_SYNTAX.id()), None);
}

#[test]
fn a_codes_number_is_read_off_its_spelling() {
    assert_eq!(pynecore_codes::id_of("R0010"), Some(820_010));
    assert!(pynecore_codes::ALL.iter().all(|c| pynecore_codes::id_of(c.code) == Some(c.id())));
    for bad in ["", "R001", "R00100", "r0010", "R00x0", "00010"] {
        assert_eq!(pynecore_codes::id_of(bad), None, "{bad:?}");
    }
}
