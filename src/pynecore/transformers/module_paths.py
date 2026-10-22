"""
Where an imported module's source lives, found without executing the module.

Shared by the type pass, which reads an imported module's interface, and the
call-site isolation, which routes a call into it; a leaf, so neither depends on
the other for it.
"""
import importlib.util
from pathlib import Path

__all__ = ['module_spec_origin', 'package_member']


def module_spec_origin(dotted: str) -> str | None:
    """
    The source file ``find_spec`` resolves a dotted module name to.

    :param dotted: Dotted module name
    :return: Its ``.py`` source path, or None when it is not a readable Python module
    """
    try:
        spec = importlib.util.find_spec(dotted)
    except Exception:  # noqa: any resolution failure means "not a module here"
        return None
    if spec is not None and spec.origin and spec.origin.endswith('.py'):
        return spec.origin
    return None


def package_member(directory: Path, dotted: str) -> str | None:
    """
    The source of a module below a package directory, found without importing.

    The candidates are tried in the import hook's order, ``name.py`` before
    ``name/__init__.py``.

    :param directory: The package directory the name is relative to
    :param dotted: Dotted module name below it; empty for the package itself
    :return: Its source path, or None when there is no such file
    """
    if dotted:
        target = directory.joinpath(*dotted.split('.'))
        candidates = (target.parent / f'{target.name}.py', target / '__init__.py')
    else:
        candidates = (directory / '__init__.py',)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
