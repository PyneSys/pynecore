"""
@pyne
"""
import importlib.util
import sys
from pathlib import Path

import pynecore.core.import_hook as import_hook
from pynecore.core.import_hook import PyneLoader, _get_transform_pipeline_hash, compile_interface


def main():
    """Dummy main so this file is a valid Pyne script."""
    pass


def __test_helper_write_switch_module(path: Path, arms: int) -> None:
    """Write a module whose function is an ``elif`` chain ``arms`` levels deep.

    It is the shape PyneComp emits for a Pine ``switch`` with that many arms (a
    library mapping hundreds of event ids to their titles, for one).
    """
    lines = ['"""', '@pyne', '"""', '', '', 'def title(key: str) -> str:']
    for arm in range(arms):
        lines.append(f'    {"if" if arm == 0 else "elif"} key == "{arm}":')
        lines.append(f'        return "event {arm}"')
    lines += ['    return ""', '', '', 'def main():', '    pass', '']
    path.write_text('\n'.join(lines))


def __test_helper_load(path: Path):
    """Import ``path`` through the Pyne loader."""
    loader = PyneLoader(path.stem, str(path))
    spec = importlib.util.spec_from_file_location(path.stem, str(path), loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def __test_helper_count_depth_measures(monkeypatch) -> list[int]:
    """Record every nesting-depth measurement, i.e. every re-run of a stage."""
    measured: list[int] = []
    original = import_hook._nesting_depth

    def spy(tree):
        depth = original(tree)
        measured.append(depth)
        return depth

    monkeypatch.setattr(import_hook, '_nesting_depth', spy)
    return measured


def __test_deep_elif_chain_transforms__(tmp_path, monkeypatch):
    """A module nested deeper than the default recursion limit admits transforms in one run"""
    path = tmp_path / "deep_switch.py"
    __test_helper_write_switch_module(path, 600)
    measured = __test_helper_count_depth_measures(monkeypatch)
    limit = sys.getrecursionlimit()

    module = __test_helper_load(path)

    assert module.title("0") == "event 0"
    assert module.title("599") == "event 599"
    assert module.title("missing") == ""
    # The transform limit was enough: no stage had to be measured and re-run
    assert measured == []
    # The raised limit held for the transform only
    assert sys.getrecursionlimit() == limit


def __test_overflowing_stage_reruns_with_measured_headroom__(tmp_path, monkeypatch):
    """A module that overflows the transform limit is re-run under a limit sized to its depth"""
    path = tmp_path / "deeper_switch.py"
    __test_helper_write_switch_module(path, 600)
    measured = __test_helper_count_depth_measures(monkeypatch)
    # Stands in for a module nested beyond what the real transform limit covers
    monkeypatch.setattr(import_hook, '_TRANSFORM_RECURSION_LIMIT', 1000)
    limit = sys.getrecursionlimit()
    assert limit <= 1000, "the ambient limit would hide the overflow"

    module = __test_helper_load(path)

    assert module.title("599") == "event 599"
    assert measured and min(measured) >= 600
    assert sys.getrecursionlimit() == limit


def __test_deep_elif_chain_interface_is_derived__(tmp_path):
    """Transforming a deeply nested dependency for a lookup yields its interface instead of nothing"""
    path = tmp_path / "deep_switch_lib.py"
    __test_helper_write_switch_module(path, 600)
    limit = sys.getrecursionlimit()

    assert compile_interface(str(path.resolve()), _get_transform_pipeline_hash()) is not None
    assert sys.getrecursionlimit() == limit
