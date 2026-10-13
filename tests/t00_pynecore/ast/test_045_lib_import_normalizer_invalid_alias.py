"""
Import normalizer - invalid alias.

The script under test is not Pyne code that compiles, so it is not this module: a
test runner may load a test module through the import hook, which would raise the
expected error at collection instead of inside the test.
"""
from pathlib import Path

import pytest

from pynecore.core.import_hook import PyneLoader

__test_helper_script = '''"""
@pyne
"""
from pynecore import lib as lib2


def main():
    print(lib2.close)
'''


def __test_import_normalizer_invalid_alias__(tmp_path: Path):
    """
    Import normalizer - invalid alias
    """
    path = tmp_path / 'invalid_alias_script.py'
    path.write_text(__test_helper_script, encoding='utf-8')
    with pytest.raises(SyntaxError, match="'lib' must be imported as itself, not as an alias"):
        PyneLoader(path.stem, str(path)).source_to_code(__test_helper_script, str(path))
