"""
The diagnostic code generator (``codes/generate.py``): its Python and docs output
for ``codes/sample.toml``, its validation, the merge of two registries, and that
the committed outputs match ``codes/registry.toml``.
"""
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = _load('codes_generate', _REPO_ROOT / 'codes' / 'generate.py')


def __test_committed_outputs_match_the_registry__():
    """The generated Rust side and docs pages are what the registry gives today"""
    assert generate.stale(generate.outputs(), generate.DOCS) == [], \
        "regenerate with `python codes/generate.py` from the pynecore checkout root"


def __test_the_registry_holds_only_pynecore_domains__():
    """Every code of PyneCore's registry is a runtime, Pyne, Edge or internal code"""
    codes = generate.load(generate.REGISTRY)
    assert codes and {c.code[0] for c in codes} <= set(generate.DOMAINS)


def __test_sample_python_side_renders_messages__(tmp_path: Path):
    """The self-contained Python module renders a message from its parameters"""
    module_path = tmp_path / "sample_codes.py"
    sample = generate.load(generate.SAMPLE, generate.ALL_DOMAINS)
    module_path.write_text(generate.python(sample, "codes/sample.toml"), encoding="utf-8")
    module = _load("sample_codes", module_path)
    assert sorted(module.ALL) == ["L0001", "S0001"]
    code = module.UNEXPECTED_TOKEN
    assert (code.severity, code.params) == ("error", ("expected", "found"))
    assert code.render("`)`", "`,`") == "Expected `)`, found `,`; `{` opens nothing here."


def __test_sample_docs_have_a_page_per_code_and_an_index__():
    """A page per code, retired ones included, and an index listing them"""
    pages = generate.docs(generate.load(generate.SAMPLE, generate.ALL_DOMAINS), "Intro.")
    assert sorted(pages) == ["L0001.md", "S0001.md", "V0001.md", "index.md"]
    page = pages["S0001.md"]
    assert page.startswith("# S0001: Unexpected token\n")
    assert "Expected {expected}, found {found}" in page
    assert "Quick fix: Write {expected}" in page
    assert "The parser expected `expected`" in page
    assert "| 6.0.0 | 6.1.0   |" in pages["V0001.md"]
    assert pages["index.md"].startswith("# Diagnostic codes\n\nIntro.\n")
    assert "[V0001](V0001.md) | info (retired)" in pages["index.md"]


def __test_rust_side_reexports_and_lists_another_crates_codes__():
    """With ``reexport``, the other crate's codes are re-exported and sorted into ALL and RUNTIME"""
    ours = generate.load(generate.SAMPLE, generate.ALL_DOMAINS)
    theirs = generate.load(generate.REGISTRY)
    text = generate.rust(ours, "codes/sample.toml", ("pynecore_codes", theirs))
    assert "pub use pynecore_codes::{EDGE_SYNTAX, " in text
    all_line = next(line for line in text.splitlines() if line.startswith("pub static ALL"))
    assert all_line.index("&EDGE_SYNTAX") < all_line.index("&NBSP_IN_VERSION_PRAGMA") \
        < all_line.index("&DIVISION_BY_ZERO") < all_line.index("&UNEXPECTED_TOKEN")
    assert "pub static DIVISION_BY_ZERO" not in text


def __test_a_code_in_two_registries_is_rejected__():
    """A code or a name in two registries is an error"""
    sample = generate.load(generate.SAMPLE, generate.ALL_DOMAINS)
    assert [c.code for c in generate.merge(sample, generate.load(generate.REGISTRY))][:2] \
        == ["E0001", "E0002"]
    with pytest.raises(ValueError, match="in two registries"):
        generate.merge(sample, sample)


_BAD = [
    ('[X0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"\nexplanation="e"',
     "domain letter"),
    ('[S0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"\nexplanation="e"',
     "domain letter"),
    ('[R0001]\nname="a"\nseverity="fatal"\ntitle="t"\nmessage="m"\nexplanation="e"',
     "severity"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="{x:>3}"\nexplanation="e"',
     "placeholder"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"', "fields"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"\nexplanation="e"\n'
     '[R0002]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"\nexplanation="e"', "unique"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m"\nfix="x"\nexplanation="e"',
     "both"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m {a}"\nfix_title="t"\nfix="{b}"\n'
     'explanation="e"', "only the message's parameters"),
    ('[R0001]\nname="a"\nseverity="error"\ntitle="t"\nmessage="m {a}"\npython_message="{b}"\n'
     'explanation="e"', "python_message needs python"),
]


@pytest.mark.parametrize(("table", "error"), _BAD)
def __test_malformed_entries_are_rejected__(tmp_path: Path, table: str, error: str):
    """A malformed entry, or a code of PyneComp's domains, is rejected"""
    path = tmp_path / "bad.toml"
    path.write_text(table, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        generate.load(path)
