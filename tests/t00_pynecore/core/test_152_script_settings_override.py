"""
A script's ``[script]`` toml settings override what the decorator declares, and the
declaration stays known as the default: a value equal to it is no override. The
programmatic ``settings``/``inputs`` of ``import_script`` and ``ScriptRunner`` win
over the toml.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pynecore.core import script as script_mod
from pynecore.core.script_runner import ScriptRunner, import_script
from pynecore.lib import strategy

SCRIPT_SOURCE = '''"""
@pyne
"""
from pynecore.lib import script, strategy, input
{library_import}

@script.strategy("Settings", initial_capital=5000, commission_value=0.1,
                 default_qty_type=strategy.fixed, default_qty_value=2)
def main(length=input.int(10, "Length")):
    pass
'''

LIBRARY_SOURCE = '''"""
@pyne
"""
from pynecore.lib import script


@script.library("Settings Library")
def main():
    pass


def twice(x: float) -> float:
    return x * 2
'''


def __test_helper_script_path(tmp_path: Path, toml: str | None = None, *,
                              with_library: bool = False) -> Path:
    """Write the strategy (its sibling toml, and the library it imports)"""
    if with_library:
        (tmp_path / "settings_lib.py").write_text(LIBRARY_SOURCE, encoding="utf-8")
    path = tmp_path / "settings.py"
    library_import = "from settings_lib import twice\n" if with_library else ""
    path.write_text(SCRIPT_SOURCE.format(library_import=library_import), encoding="utf-8")
    if toml is not None:
        path.with_suffix(".toml").write_text(toml, encoding="utf-8")
    return path


def __test_helper_script(tmp_path: Path, toml: str | None = None, **import_kwargs):
    """Write the strategy (and its sibling toml) and import it"""
    return import_script(__test_helper_script_path(tmp_path, toml), **import_kwargs).main.script


def __test_helper_length(module) -> float:
    """The value the ``length`` input resolved to at import"""
    return inspect.signature(module.main).parameters["length"].default


# noinspection PyProtectedMember
def __test_toml_overrides_keep_the_declared_default__(tmp_path: Path):
    """ A toml value overrides the setting, the decorator argument remains the default """
    script = __test_helper_script(tmp_path, "[script]\ninitial_capital = 7000\ncommission_value = 0.1\n")
    assert script.initial_capital == 7000
    assert script.default("initial_capital") == 5000
    assert script.default("default_qty_value") == 2
    assert script._modified == {"initial_capital"}


# noinspection PyProtectedMember
def __test_a_value_equal_to_the_default_is_no_override__(tmp_path: Path):
    """ Setting a value back to the declaration un-marks it, so save() comments it out """
    script = __test_helper_script(tmp_path, "[script]\ninitial_capital = 7000\n")
    script.set_setting("initial_capital", 5000)
    assert "initial_capital" not in script._modified

    script.set_setting("slippage", 3)
    out = tmp_path / "out.toml"
    script.save(out)
    lines = out.read_text().splitlines()
    assert "#initial_capital = 5000" in lines
    assert "slippage = 3" in lines


# noinspection PyProtectedMember
def __test_programmatic_settings_win_over_the_toml__(tmp_path: Path):
    """ ``import_script(settings=...)`` applies after the toml and leaves nothing behind """
    script = __test_helper_script(tmp_path, "[script]\ninitial_capital = 7000\n",
                                  settings={"initial_capital": 9000, "pyramiding": 3})
    assert script.initial_capital == 9000
    assert script.pyramiding == 3
    assert script.default("initial_capital") == 5000
    assert script._modified == {"initial_capital", "pyramiding"}
    assert not script_mod._programmatic_settings


def __test_overrides_survive_an_imported_library__(tmp_path: Path):
    """ A library the script imports is decorated first, and must not consume the overrides """
    path = __test_helper_script_path(tmp_path, with_library=True)
    module = import_script(path, inputs={"length": 33}, settings={"initial_capital": 9000})
    assert module.main.script.initial_capital == 9000
    assert __test_helper_length(module) == 33


# noinspection PyProtectedMember
def __test_unknown_setting_raises_and_leaves_nothing_behind__(tmp_path: Path):
    """ A misspelled setting is an error, and the failed import does not leak overrides """
    path = __test_helper_script_path(tmp_path)
    with pytest.raises(ValueError, match="initial_capitol"):
        import_script(path, inputs={"length": 33}, settings={"initial_capitol": 9000})
    assert not script_mod._programmatic_settings
    assert not script_mod._programmatic_inputs

    module = import_script(path)
    assert module.main.script.initial_capital == 5000
    assert __test_helper_length(module) == 10


def __test_helper_import_in_child(path: Path, save_overrides: bool) -> list[str]:
    """Import with overrides where the decorator saves the toml (never under pytest)"""
    probe = (
        "import sys; "
        "from pathlib import Path; "
        "from pynecore.core.script_runner import import_script; "
        "import_script(Path(sys.argv[1]), inputs={'length': 33}, "
        "settings={'initial_capital': 9000}, save_overrides=sys.argv[2] == '1')"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYNE_SAVE_SCRIPT_TOML"}
    subprocess.run([sys.executable, "-c", probe, str(path), "1" if save_overrides else "0"],
                   cwd=path.parent, env=env, check=True, capture_output=True, text=True)
    return path.with_suffix(".toml").read_text(encoding="utf-8").splitlines()


def __test_overrides_stay_out_of_the_toml__(tmp_path: Path):
    """ The overrides configure the run only: the saved toml keeps what it held """
    path = __test_helper_script_path(tmp_path, "[script]\ninitial_capital = 7000\n")
    lines = __test_helper_import_in_child(path, save_overrides=False)
    assert "initial_capital = 7000" in lines
    assert "value = 33" not in lines
    assert "#value =" in lines


def __test_save_overrides_records_them_in_the_toml__(tmp_path: Path):
    """ ``save_overrides=True`` (the IDE's input form) writes the overrides into the toml """
    path = __test_helper_script_path(tmp_path, "[script]\ninitial_capital = 7000\n")
    lines = __test_helper_import_in_child(path, save_overrides=True)
    assert "initial_capital = 9000" in lines
    assert "value = 33" in lines


def __test_script_runner_settings__(tmp_path: Path, syminfo, dummy_ohlcv_iter):
    """ ``ScriptRunner(settings=...)`` configures the script it runs """
    path = __test_helper_script_path(tmp_path, "[script]\ninitial_capital = 7000\n")
    runner = ScriptRunner(path, dummy_ohlcv_iter, syminfo,
                          settings={"initial_capital": 9000, "commission_type": strategy.cash})
    assert runner.script.initial_capital == 9000
    assert runner.script.commission_type == strategy.commission.cash_per_order


def __test_toml_commission_type_cash_is_the_per_order_fee__(tmp_path: Path):
    """ The ``strategy.cash`` alias normalizes from the toml like from the decorator """
    script = __test_helper_script(tmp_path, '[script]\ncommission_type = "cash"\n')
    assert script.commission_type == strategy.commission.cash_per_order


# noinspection PyProtectedMember
def __test_pyramiding_zero_is_one_and_no_override__(tmp_path: Path):
    """ The decorator's pyramiding=0 means 1, so 1 (or 0) from the toml is no override """
    script = __test_helper_script(tmp_path, "[script]\npyramiding = 0\n")
    assert script.pyramiding == 1
    assert script.default("pyramiding") == 1
    assert "pyramiding" not in script._modified
