# The import hook must install before anything that can transitively pull in
# ``pynecore.lib``: the lib package imports its @pyne submodules during its own
# init, and those must already load through the transforming loader.
from .core import import_hook
from .types import Series, Persistent, PersistentSeries
from .core.pine_range import pine_range, pine_loop

from pathlib import Path as _Path


def docs_path() -> _Path:
    """
    Return the documentation directory of the installed PyneCore version.

    A built install carries it inside the package (``pynecore/docs/``); an editable
    install uses the ``docs/`` of the source tree it runs from.

    :return: The directory holding the Markdown documentation
    :raises FileNotFoundError: If neither location holds the documentation
    """
    package_dir = _Path(__file__).parent
    for candidate in (package_dir / 'docs', package_dir.parent.parent / 'docs'):
        if (candidate / 'README.md').is_file():
            return candidate
    raise FileNotFoundError("PyneCore documentation not found")
