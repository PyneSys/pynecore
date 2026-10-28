fn main() {
    // macOS links an extension module against no libpython; the interpreter
    // that loads it provides the symbols
    pyo3_build_config::add_extension_module_link_args();
}
