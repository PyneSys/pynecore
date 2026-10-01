"""
Regression tests for :func:`pynecore.core.script_runner.import_script`: the
``@pyne`` magic-docstring pre-check, and which file a call executes.

The check once read a 1KB head and required the CLOSED docstring inside it
(``\"\"\".*?@pyne.*?\"\"\"``), so a valid script whose module docstring closed
past the first kilobyte was rejected with "must have a magic doc comment"
(observed live: a bot's docstring grew past 1KB in an edit and its next
scheduled run died on import). The pre-check now delegates to the import
hook's head detector, which matches a docstring that BEGINS with ``@pyne``
without needing the closing quotes in the window.

The script was once imported by its bare file stem, so a second script with the
same file name got the first one's cached module and silently ran the wrong
script. Every call now executes exactly the requested file.
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


def __test_scripts_sharing_a_file_name_load_their_own_file__(tmp_path: Path) -> None:
    first = tmp_path / "first" / "same_name_probe.py"
    second = tmp_path / "second" / "same_name_probe.py"
    for path, title in ((first, "first probe"), (second, "second probe")):
        path.parent.mkdir()
        path.write_text(
            '"""\n@pyne\n"""\n'
            'from pynecore.lib import script\n'
            '\n'
            '\n'
            f'@script.indicator("{title}")\n'
            'def main():\n'
            '    pass\n'
        )

    first_module = import_script(first)
    second_module = import_script(second)

    assert Path(str(first_module.__file__)).resolve() == first.resolve()
    assert Path(str(second_module.__file__)).resolve() == second.resolve()
    assert first_module.main.script.title == "first probe"
    assert second_module.main.script.title == "second probe"


def __test_a_repeated_import_runs_the_script_again__(tmp_path: Path) -> None:
    script = _write_script(tmp_path / "repeated_probe.py", docstring_body="@pyne")
    module = import_script(script)
    again = import_script(script)
    assert again is not module
    assert again.__name__ == module.__name__
