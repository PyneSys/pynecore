"""Generate the Rust side and the docs pages of PyneCore's diagnostic codes from
``codes/registry.toml``.

    python codes/generate.py          write every generated file
    python codes/generate.py --check  fail when a generated file is stale

The registry format is described in its header. The Rust side holds what a
reported diagnostic needs (code, severity, title, message template and its
parameters); the explanation and the introducing release are only in the docs.
A retired code gets a docs page and no constant.

The functions below are also PyneComp's generator: it loads its own registry
and this one with :func:`load`, and writes its outputs over both.
"""

import argparse
import re
import string
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "codes" / "registry.toml"
SAMPLE = ROOT / "codes" / "sample.toml"
RUST_ROOT = ROOT / "rust"
CRATE = RUST_ROOT / "crates" / "pynecore-codes"
RUST = CRATE / "src" / "registry.rs"
SAMPLE_RUST = CRATE / "tests" / "sample" / "registry.rs"
DOCS = ROOT / "docs" / "reference" / "codes"

#: The domains PyneCore's registry holds: R runtime, P Pyne code rules,
#: E Pyne Edge profile, I internal error.
DOMAINS = "RPEI"
#: Every domain of the Pyne chain; PyneComp's registry holds the others.
ALL_DOMAINS = "SNTCLVRPEI"

NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
SEVERITIES = ("error", "warning", "info")
FIELDS = {"name", "severity", "title", "message", "since", "explanation", "retired", "fix_title",
          "fix", "python", "python_message"}
OPTIONAL = {"since", "retired", "fix_title", "fix", "python", "python_message"}

#: The domains of the codes a run raises (``RUNTIME``).
RUNTIME_DOMAINS = "RI"


@dataclass(frozen=True)
class Code:
    code: str
    name: str
    severity: str
    title: str
    message: str
    params: tuple[str, ...]
    since: str | None
    explanation: str
    retired: str | None
    #: The quick fix: its title and the text that replaces the diagnostic's
    #: range, both templates of the message's parameters.
    fix_title: str | None = None
    fix: str | None = None
    #: The built-in exception classes the code's exception derives from: one
    #: class, or several where different built-ins raise different ones for
    #: the same condition.
    python: tuple[str, ...] = ()


def _params(message: str) -> tuple[str, ...]:
    """The ``{name}`` placeholders in order of first appearance."""
    params: list[str] = []
    for _, field, spec, conversion in string.Formatter().parse(message):
        if field is None:
            continue
        if not NAME_RE.fullmatch(field) or spec or conversion:
            raise ValueError(f"placeholder {{{field}}}: only {{name}} is allowed")
        if field not in params:
            params.append(field)
    return tuple(params)


def load(path: Path, domains: str = DOMAINS) -> list[Code]:
    """The registry's codes sorted by code; a malformed entry, or a code outside
    ``domains``, is an error."""
    code_re = re.compile(f"[{domains}]\\d{{4}}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    codes: list[Code] = []
    names: set[str] = set()
    for code, row in sorted(data.items()):
        where = f"{path.name} [{code}]"
        if not code_re.fullmatch(code):
            raise ValueError(f"{where}: a code is a domain letter of {domains} and four digits")
        if not isinstance(row, dict) or set(row) - FIELDS or OPTIONAL | set(row) != FIELDS:
            raise ValueError(f"{where}: fields are {sorted(FIELDS)} ({sorted(OPTIONAL)} optional)")
        python = row.get("python", [])
        if isinstance(python, str):
            python = [python]
        if not all(isinstance(v, str) for k, v in row.items() if k != "python") or not (
                isinstance(python, list) and all(isinstance(v, str) for v in python)):
            raise ValueError(f"{where}: every field is a string "
                             "(python: a string or a list of them)")
        if not NAME_RE.fullmatch(row["name"]) or row["name"] in names:
            raise ValueError(f"{where}: name {row['name']!r} is not a unique snake_case name")
        if row["severity"] not in SEVERITIES:
            raise ValueError(f"{where}: severity is one of {SEVERITIES}")
        for field in ("since", "retired"):
            if field in row and not VERSION_RE.fullmatch(row[field]):
                raise ValueError(f"{where}: {field} is a release like 6.0.0")
        try:
            params = _params(row["message"])
            fix_params = _params(row.get("fix_title", "")) + _params(row.get("fix", ""))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        if ("fix" in row) != ("fix_title" in row):
            raise ValueError(f"{where}: a quick fix has both fix_title and fix")
        if set(fix_params) - set(params):
            raise ValueError(f"{where}: a quick fix uses only the message's parameters")
        if "python_message" in row:
            try:
                py_params = _params(row["python_message"])
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            if "python" not in row or set(py_params) - set(params):
                raise ValueError(f"{where}: python_message needs python and uses only the "
                                 "message's parameters")
        names.add(row["name"])
        codes.append(Code(code, row["name"], row["severity"], row["title"].strip(), row["message"],
                          params, row.get("since"), row["explanation"].strip(), row.get("retired"),
                          row.get("fix_title"), row.get("fix"), tuple(python)))
    return codes


def merge(*registries: list[Code]) -> list[Code]:
    """The codes of several registries sorted by code; a code or a name in two
    of them is an error."""
    seen_codes: set[str] = set()
    seen_names: set[str] = set()
    for codes in registries:
        for c in codes:
            if c.code in seen_codes or c.name in seen_names:
                raise ValueError(f"{c.code} ({c.name}) is in two registries")
            seen_codes.add(c.code)
            seen_names.add(c.name)
    return sorted((c for codes in registries for c in codes), key=lambda c: c.code)


def _rust_str(text: str) -> str:
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{{{ord(ch):x}}}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def header(sources: str) -> str:
    """The first line of a generated file, ``sources`` naming the registries it
    is generated from."""
    return f"@generated by codes/generate.py from {sources}. Do not edit; regenerate."


def rust(codes: list[Code], sources: str, reexport: tuple[str, list[Code]] | None = None) -> str:
    """Constants of ``super::Code``, and ``ALL`` and ``RUNTIME`` sorted by code.

    ``reexport`` is a crate and codes of its own: they are re-exported from it
    and listed in ``ALL`` and ``RUNTIME`` with the constants written here.
    """
    live = [c for c in codes if c.retired is None]
    lines = [f"// {header(sources)}", ""]
    external: list[Code] = []
    if reexport is not None:
        crate, theirs = reexport
        external = [c for c in theirs if c.retired is None]
        names = ", ".join(c.name.upper() for c in external)
        lines += [f"pub use {crate}::{{{names}}};", ""]
    for c in live:
        params = ", ".join(_rust_str(p) for p in c.params)
        fix = ("None" if c.fix is None else
               f"Some(super::Fix {{ title: {_rust_str(c.fix_title)}, "
               f"replacement: {_rust_str(c.fix)} }})")
        lines += [
            f"/// {c.code}: {c.title}",
            f"pub static {c.name.upper()}: super::Code = super::Code {{",
            f"    code: {_rust_str(c.code)},",
            f"    severity: super::Severity::{c.severity.capitalize()},",
            f"    title: {_rust_str(c.title)},",
            f"    message: {_rust_str(c.message)},",
            f"    params: &[{params}],",
            f"    fix: {fix},",
            "};",
            "",
        ]
    every = sorted(live + external, key=lambda c: c.code)
    items = ", ".join(f"&{c.name.upper()}" for c in every)
    lines += ["/// Every reported code, sorted by `code`.",
              f"pub static ALL: &[&super::Code] = &[{items}];", ""]
    run = ", ".join(f"&{c.name.upper()}" for c in every if c.code[0] in RUNTIME_DOMAINS)
    lines += [
        "/// The codes a compiled script's run raises (`R`, `I`), sorted by `code`: a",
        "/// host that renders run errors links these alone.",
        f"pub static RUNTIME: &[&super::Code] = &[{run}];",
        "",
    ]
    return "\n".join(lines)


def python(codes: list[Code], sources: str) -> str:
    """A self-contained module of ``Code`` tuples and ``ALL``: what a tool that
    reports codes without PyneCore installed (the PyneIDE linter) reads."""
    live = [c for c in codes if c.retired is None]
    lines = [
        f"# {header(sources)}",
        '"""The diagnostic codes of the Pyne chain (``codes/registry.toml``)."""',
        "",
        "from typing import NamedTuple",
        "",
        "",
        "class Code(NamedTuple):",
        "    code: str",
        "    severity: str",
        "    title: str",
        "    message: str",
        "    params: tuple[str, ...]",
        "    #: The quick fix's title and the text that replaces the diagnostic's range.",
        "    fix_title: str | None = None",
        "    fix: str | None = None",
        "",
        "    def render(self, *args: str) -> str:",
        '        """The message with ``args[i]`` in place of ``params[i]``."""',
        "        return self.message.format(**dict(zip(self.params, args)))",
        "",
        "",
    ]
    for c in live:
        params = "".join(f"{p!r}, " for p in c.params).rstrip(" ")
        fix = "" if c.fix is None else f", {c.fix_title!r}, {c.fix!r}"
        lines.append(f"{c.name.upper()} = Code({c.code!r}, {c.severity!r}, {c.title!r}, "
                     f"{c.message!r}, ({params}){fix})")
    if live:
        lines.append("")
    items = "".join(f"{c.name.upper()}, " for c in live).rstrip(" ")
    table = f"{{c.code: c for c in ({items})}}" if live else "{}"
    lines += ["#: Every reported code by its code.", f"ALL: dict[str, Code] = {table}", ""]
    return "\n".join(lines)


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    out.insert(1, "|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return out


def docs(codes: list[Code], intro: str) -> dict[str, str]:
    """``index.md`` (``intro`` under its title) and one page per code, by file name."""
    pages: dict[str, str] = {}
    for c in codes:
        fields = [("Severity", c.severity), ("Since", c.since), ("Retired", c.retired),
                  ("PyneCore raises", ", ".join(f"`{p}`" for p in c.python))]
        fields = [(name, value) for name, value in fields if value]
        rows = [[name for name, _ in fields], [value for _, value in fields]]
        page = [f"# {c.code}: {c.title}", "", *_table(rows), "", "Message:", "",
                "```text", c.message, "```", ""]
        if c.fix_title is not None:
            page += [f"Quick fix: {c.fix_title}", ""]
        pages[f"{c.code}.md"] = "\n".join(page + [c.explanation, ""])
    index = ["# Diagnostic codes", "", intro, ""]
    if codes:
        rows = [["Code", "Severity", "Title"]]
        rows += [[f"[{c.code}]({c.code}.md)", c.severity + (" (retired)" if c.retired else ""),
                  c.title] for c in codes]
        index += _table(rows)
    else:
        index.append("No codes yet.")
    pages["index.md"] = "\n".join(index) + "\n"
    return pages


def rustfmt(code: str, edition: str, workspace: Path) -> str:
    """``code`` as ``cargo fmt`` leaves it, so formatting the workspace keeps it
    current: ``edition`` is the crate's, and ``workspace`` the Cargo workspace
    root, whose ``rustfmt.toml`` applies."""
    done = subprocess.run(["rustfmt", "--edition", edition, "--emit", "stdout"], input=code,
                          capture_output=True, text=True, cwd=workspace, check=True)
    return done.stdout


def stale(files: dict[Path, str], docs_dir: Path) -> list[Path]:
    """Of ``files`` (path -> content), those that differ on disk, and the docs
    pages under ``docs_dir`` that are no longer generated."""
    changed = [p for p, text in files.items()
               if not p.is_file() or p.read_text(encoding="utf-8") != text]
    extra = [p for p in docs_dir.glob("*.md") if p not in files] if docs_dir.is_dir() else []
    return changed + extra


def run(files: dict[Path, str], docs_dir: Path, root: Path, check: bool) -> int:
    """``--check``: report the stale files; otherwise write them and remove the
    docs pages no longer generated."""
    paths = stale(files, docs_dir)
    if check:
        for p in paths:
            print(f"stale: {p.relative_to(root)}", file=sys.stderr)
        return 1 if paths else 0
    for p in paths:
        if p in files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(files[p], encoding="utf-8")
        else:
            p.unlink()
        print(f"wrote {p.relative_to(root)}" if p in files else f"removed {p.relative_to(root)}")
    return 0


def outputs() -> dict[Path, str]:
    """Every file generated from PyneCore's registry, and its content."""
    codes = load(REGISTRY)
    sample = load(SAMPLE, ALL_DOMAINS)
    files = {RUST: rustfmt(rust(codes, "codes/registry.toml"), "2024", RUST_ROOT),
             SAMPLE_RUST: rustfmt(rust(sample, "codes/sample.toml"), "2024", RUST_ROOT)}
    pages = docs(codes, "Every code PyneCore reports.")
    files.update({DOCS / name: text for name, text in pages.items()})
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail when a generated file is stale")
    args = ap.parse_args()
    return run(outputs(), DOCS, ROOT, args.check)


if __name__ == "__main__":
    sys.exit(main())
