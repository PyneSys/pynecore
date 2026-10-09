"""
The runtime interface the emission calls into is part of the pipeline hash.

Cached script bytecode calls the slot-state helpers, the safe conversions and the
security protocol by name, position and keyword. A change to one of their
parameter lists must invalidate it; a change to a body must not, because a cached
script simply calls the current body.
"""
import shutil
from pathlib import Path

from pynecore.core import import_hook
from pynecore.core.import_hook import _RUNTIME_ABI, _runtime_abi


def __test_helper_copy(tmp_path: Path) -> Path:
    """A copy of the files the interface is read from."""
    package = Path(import_hook.__file__).parent.parent
    for relative, _ in _RUNTIME_ABI:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / relative, target)
    return tmp_path


def __test_every_listed_helper_is_defined__():
    """ A renamed or removed helper is noticed, not silently skipped """
    for relative, names in _runtime_abi().items():
        for name, interfaces in names.items():
            assert interfaces, f"{relative}: {name} is no longer defined"


def __test_the_cut_reads_what_a_whole_parse_reads__():
    """ Parsing only the definitions gives the same interfaces as parsing the files """
    assert _runtime_abi() == _runtime_abi(whole_files=True)


def __test_a_parameter_change_moves_the_interface__(tmp_path):
    """ An added parameter is a different calling interface """
    root = __test_helper_copy(tmp_path)
    before = _runtime_abi(root=root)
    path = root / 'core' / 'safe_convert.py'
    text = path.read_text()
    assert 'def safe_div(a: PyneFloat, b: PyneFloat):' in text
    path.write_text(text.replace('def safe_div(a: PyneFloat, b: PyneFloat):',
                                 'def safe_div(a: PyneFloat, b: PyneFloat, c=None):'))
    assert _runtime_abi(root=root) != before


def __test_a_body_change_keeps_the_interface__(tmp_path):
    """ The helper's behaviour is the current body's; cached callers stay valid """
    root = __test_helper_copy(tmp_path)
    before = _runtime_abi(root=root)
    path = root / 'core' / 'safe_convert.py'
    text = path.read_text()
    path.write_text(text.replace('def safe_div(a: PyneFloat, b: PyneFloat):',
                                 'def safe_div(a: PyneFloat, b: PyneFloat):\n    _unused = 0'))
    assert _runtime_abi(root=root) == before


def __test_a_new_class_field_moves_the_interface__(tmp_path):
    """ The emission constructs ``ScriptRequirements`` by keyword """
    root = __test_helper_copy(tmp_path)
    before = _runtime_abi(root=root)
    path = root / 'core' / 'broker' / 'models.py'
    text = path.read_text()
    marker = '    market_orders: bool = False\n'
    assert marker in text
    path.write_text(text.replace(marker, marker + '    probe_field: bool = False\n', 1))
    assert _runtime_abi(root=root) != before
