"""
Run the analysing half of the transform on one source file.

The type tests inspect a module's type table and diagnostics as its transform
finds them, without the emission. The loader itself never needs only that half
(an imported module's interface comes from its transform, see
``import_hook.compile_interface``), so the entry point lives with the tests.
"""
import ast
from pathlib import Path

from pynecore.core import import_hook
from pynecore.transformers.module_interface import stable_source
from pynecore.transformers.pine_type_table import PineTypeTable
from pynecore.transformers.pine_type_transformer import module_table

__all__ = ['analyse_module']


# noinspection PyProtectedMember
def analyse_module(path: str | Path) -> tuple[ast.Module, PineTypeTable, tuple[int, int]] | None:
    """
    Analyse one module from its source, compiling and running nothing.

    :param path: Path to the ``.py`` source.
    :return: The analysed tree, its type table and the fingerprint of the bytes
             they were derived from, or None when the file is not readable, not
             parseable, not Pyne code, or does not analyse.
    """
    path = Path(path)
    stable = stable_source(path)
    if stable is None:
        return None
    data, fingerprint = stable
    try:
        source = data.decode('utf-8')
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return None
    is_pyne_module, pyne_mode = import_hook._module_mode(tree)
    if not is_pyne_module:
        return None
    try:
        analysed = import_hook._with_nesting_headroom(
            lambda module: import_hook._analyse_tree(module, source, path, pyne_mode),
            tree, source)
    except (SyntaxError, RecursionError):
        return None
    table = module_table(analysed)
    return None if table is None else (analysed, table, fingerprint)
