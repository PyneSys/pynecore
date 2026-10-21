"""The native transform extension (``rust/crates/pyne-transform-py``)."""
import ast
from typing import Any

__version__: str


class Unsupported(Exception):
    """A construct the native transform does not handle; the Python path runs instead."""


def parse(source: str) -> ast.Module:
    """Parse a module with the native parser into Python ``ast`` objects."""


def roundtrip(tree: ast.AST) -> ast.AST:
    """Convert a tree into the native arena and back."""


def run(tree: ast.Module, steps: list[str], context: dict[str, Any]) -> ast.Module:
    """Run native steps over a module, in order; the input tree is left as it was."""


def steps() -> list[str]:
    """The names of the steps with a native port."""


def ported_json() -> str:
    """The ``ported.json`` manifest the extension was built with."""
