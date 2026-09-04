"""
The loader reads the script's bool na choice off its decorator, in every spelling
the script may use, and refuses a choice it cannot read before the module runs.
"""
import ast
from pathlib import Path

import pytest

from pynecore.core.import_hook import _script_bool_na


def _choice(source: str) -> bool | None:
    return _script_bool_na(ast.parse(source), Path('script.py'))


def __test_every_decorator_spelling_is_recognized__():
    """script., lib.script., pynecore.lib.script. and aliases of both"""
    body = "def main():\n    pass\n"
    assert _choice("from pynecore.lib import script\n@script.indicator(na_bool=True)\n" + body) is True
    assert _choice("from pynecore import lib\n@lib.script.strategy('s', na_bool=True)\n" + body) is True
    assert _choice("import pynecore.lib as pl\n@pl.script.library('l', na_bool=True)\n" + body) is True
    assert _choice("import pynecore\n@pynecore.lib.script.indicator(na_bool=True)\n" + body) is True
    assert _choice("from pynecore.lib import script as s\n@s.indicator(na_bool=True)\n" + body) is True
    assert _choice("from pynecore.lib import script\n@script.indicator(na_bool=False)\n" + body) is False


def __test_a_missing_keyword_and_a_missing_decorator_differ__():
    """No keyword is the two-state default, no script decorator is no choice at all"""
    assert _choice("from pynecore.lib import script\n@script.indicator('x')\ndef main():\n    pass\n") is False
    assert _choice("def main():\n    pass\n") is None
    assert _choice("@other.indicator(na_bool=True)\ndef main():\n    pass\n") is None


def __test_a_computed_choice_is_refused__():
    """The semantics are fixed before the module body runs, so only a literal works"""
    with pytest.raises(SyntaxError, match="literal True or False"):
        _choice("from pynecore.lib import script\nFLAG = True\n@script.indicator(na_bool=FLAG)\n"
                "def main():\n    pass\n")


def __test_the_script_decorator_belongs_on_main__():
    """A module has one entry point: the decorator on any other function is refused"""
    with pytest.raises(SyntaxError, match="belongs on 'main', not on 'entry'"):
        _choice("from pynecore.lib import script\n"
                "@script.library('l')\ndef entry():\n    pass\n"
                "@script.indicator(na_bool=True)\ndef main():\n    pass\n")


def __test_main_is_decorated_once__():
    """A redefined, decorated main is refused rather than silently picked"""
    with pytest.raises(SyntaxError, match="'main' is defined twice"):
        _choice("from pynecore.lib import script\n"
                "@script.indicator()\ndef main():\n    pass\n"
                "@script.indicator(na_bool=True)\ndef main():\n    pass\n")
