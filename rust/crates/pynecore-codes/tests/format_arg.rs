//! `format_arg` over the vectors of `scripts/errorcodes/format_arg.toml`, which PyneCore's
//! Python `format_arg` runs too.

use pynecore_codes::format_arg;

mod vectors {
    include!("format_arg/vectors.rs");
}

#[test]
fn format_arg_writes_every_vector_as_the_python_side_does() {
    for &(bits, text) in vectors::VECTORS {
        let v = f64::from_bits(bits);
        assert_eq!(format_arg(v), text, "{v:e}");
    }
}
