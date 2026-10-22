"""The native transform extension (``rust/crates/pyne-transform-py``)."""
import ast
from typing import Any

from .pine_type_table import PineTypeTable

__version__: str


class Unsupported(Exception):
    """A construct the native transform does not handle; the Python path runs instead."""


class Rejected(Exception):
    """The module is invalid; the Python step raises the error the author sees."""


def parse(source: str) -> ast.Module:
    """Parse a module with the native parser into Python ``ast`` objects."""


def roundtrip(tree: ast.AST) -> ast.AST:
    """Convert a tree into the native arena and back."""


def dump(tree: ast.AST, include_attributes: bool = False) -> str:
    """``ast.dump(tree, include_attributes=...)`` written by the native side."""


def run(tree: ast.Module, steps: list[str], context: dict[str, Any]) -> ast.Module:
    """Run native steps over a module, in order; the input tree is left as it was."""


def steps() -> list[str]:
    """The names of the steps with a native port."""


def ported_json() -> str:
    """The ``ported.json`` manifest the extension was built with."""


def _rules(name: str, *args: Any) -> Any:
    """One function of the native ``pine_type_rules`` port, by its Python name (tests only)."""


def _qualifier(name: str, *args: Any) -> Any:
    """One function of the native ``pine_qualifier`` port, by its Python name (tests only)."""


def _data(section: str) -> dict[str, Any]:
    """A section of the registries compiled into the extension (tests only)."""


def _fdlibm(name: str, x: bytes, y: bytes | None = None) -> bytes:
    """One function of the native ``core.fdlibm`` port over packed float64 arguments
    (``y``: the exponents of ``pow``), the results packed alike (tests only)."""


def _quantize_embed(values: bytes) -> bytes:
    """The native ``const_fold.quantize_embed`` of packed float64 values (tests only)."""


def _const_eval(expr: ast.expr) -> int | float | None:
    """The native ``quantize_embed(_try_eval(expr, {}))``, None where the fold leaves
    ``expr`` in place (tests only)."""


def _type_table(table: PineTypeTable) -> PineTypeTable:
    """A ``PineTypeTable`` converted to its native form and back, as the bridge converts
    the table of a run whose steps read it (tests only)."""


def _node_ids(tree: ast.AST, start: int = 0) -> tuple[ast.AST, int]:
    """The tree numbered by the native ``assign_node_ids``, a new tree, and the next
    unused id (tests only)."""
