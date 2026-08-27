"""
Regression tests for :func:`pynecore.cli.commands.run._pin_timenow`.

The pin must reach BOTH the running process and the ``request.security``
subprocesses: those import ``pynecore.lib`` fresh under the ``spawn`` start
method and would otherwise read the system clock while the chart body reads the
anchored instant, so a script comparing ``timenow`` across the two would see
them disagree.
"""
import os

from pynecore import lib
from pynecore.cli.commands.run import _pin_timenow

_ENV = 'PYNE_TIMENOW_MS'


def _restore(previous_ms: int, previous_env: str | None) -> None:
    lib._timenow_ms = previous_ms
    if previous_env is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = previous_env


def __test_pin_sets_both_channels__():
    """The attribute drives this process, the env var the security children."""
    previous_ms, previous_env = lib._timenow_ms, os.environ.get(_ENV)
    try:
        _pin_timenow(1_735_689_600_000)
        assert lib._timenow_ms == 1_735_689_600_000
        assert os.environ[_ENV] == '1735689600000'
        # ``module_property`` only resolves as an attribute inside a @pyne script;
        # from plain Python the underlying function is called directly.
        assert lib.timenow() == 1_735_689_600_000
    finally:
        _restore(previous_ms, previous_env)


def __test_empty_window_leaves_the_clock_alone__():
    """``None`` (no bar in the window) must not pin anything."""
    previous_ms, previous_env = lib._timenow_ms, os.environ.get(_ENV)
    try:
        lib._timenow_ms = 0
        os.environ.pop(_ENV, None)
        _pin_timenow(None)
        assert lib._timenow_ms == 0
        assert _ENV not in os.environ
    finally:
        _restore(previous_ms, previous_env)
