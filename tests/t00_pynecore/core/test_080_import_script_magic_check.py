"""
Regression tests for the ``@pyne`` magic-docstring pre-check in
:func:`pynecore.core.script_runner.import_script`.

The check once read a 1KB head and required the CLOSED docstring inside it
(``\"\"\".*?@pyne.*?\"\"\"``), so a valid script whose module docstring closed
past the first kilobyte was rejected with "must have a magic doc comment"
(observed live: a bot's docstring grew past 1KB in an edit and its next
scheduled run died on import). The pre-check now delegates to the import
hook's head detector, which matches a docstring that BEGINS with ``@pyne``
without needing the closing quotes in the window.
"""
from pathlib import Path

import pytest

from pynecore.core.script_runner import import_script


def _write_script(path: Path, *, docstring_body: str) -> Path:
    path.write_text(
        f'"""\n{docstring_body}\n"""\n'
        'from pynecore.lib import script\n'
        '\n'
        '\n'
        '@script.indicator("magic check probe")\n'
        'def main():\n'
        '    pass\n'
    )
    return path


def __test_a_docstring_longer_than_the_old_1kb_window_imports__(tmp_path: Path) -> None:
    padding = "x" * 1500
    script = _write_script(
        tmp_path / "long_docstring_probe.py",
        docstring_body=f"@pyne\n\n{padding}",
    )
    module = import_script(script)
    assert hasattr(module, "main")


def __test_a_script_without_the_magic_comment_is_rejected__(tmp_path: Path) -> None:
    script = tmp_path / "not_pyne.py"
    script.write_text('"""ordinary module"""\n\n\ndef main():\n    pass\n')
    with pytest.raises(ImportError, match="magic doc comment"):
        import_script(script)
