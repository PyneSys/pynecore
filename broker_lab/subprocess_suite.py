"""Opt-in subprocess scenarios against the real ``pyne run`` lifecycle."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pynecore.testing.broker_lab import (
    ReferenceVenueProfile,
    Scenario,
    Step,
    run_subprocess,
    temporary_entry_point,
)

_FIXTURES = Path(__file__).with_name("fixtures")


class SubprocessProfile(ReferenceVenueProfile):
    """Run the installed CLI with disposable plugin metadata and workdirs."""

    def _run_pyne(self, *, fault_env: dict[str, str] | None = None):
        with (
            temporary_entry_point(
                group="pyne.plugin",
                name="offline-lab",
                target="offline_plugin:OfflineBrokerPlugin",
            ) as metadata_root,
            TemporaryDirectory(prefix="pyne-broker-lab-run-") as temp,
        ):
            workdir = Path(temp)
            (workdir / "workdir").mkdir()
            script = workdir / "offline_strategy.py"
            script.write_text(
                '"""@pyne"""\n'
                "from pynecore.lib import script\n\n"
                '@script.strategy("Offline lifecycle")\n'
                "def main():\n"
                "    pass\n",
                encoding="utf-8",
            )
            inherited = os.environ.get("PYTHONPATH", "")
            pythonpath = os.pathsep.join(part for part in (str(metadata_root), str(_FIXTURES), inherited) if part)
            env = {
                "PYTHONPATH": pythonpath,
                "PYTHONNOUSERSITE": "1",
                "PYNE_BROKER_LAB_OFFLINE": "1",
                "PYNE_NO_COLOR_LOG": "1",
            }
            if fault_env:
                env.update(fault_env)
            return run_subprocess(
                [
                    str(Path(sys.executable).with_name("pyne")),
                    "run",
                    str(script),
                    "offline-lab:LABUSD@1",
                    "--from",
                    "-2",
                    "--broker",
                    "--shutdown-timeout",
                    "0.1",
                    "--no-log-ohlcv",
                ],
                cwd=workdir,
                timeout=8.0,
                env=env,
            )

    def _run_two_shared(self):
        with (
            temporary_entry_point(
                group="pyne.plugin",
                name="offline-lab",
                target="offline_plugin:OfflineBrokerPlugin",
            ) as metadata_root,
            TemporaryDirectory(prefix="pyne-broker-lab-shared-") as temp,
        ):
            root = Path(temp)
            (root / "workdir").mkdir()
            script = root / "offline_strategy.py"
            script.write_text(
                '"""@pyne"""\n'
                "from pynecore.lib import script\n\n"
                '@script.strategy("Offline shared lifecycle")\n'
                "def main():\n"
                "    pass\n",
                encoding="utf-8",
            )
            inherited = os.environ.get("PYTHONPATH", "")
            env = {
                "PYTHONPATH": os.pathsep.join(part for part in (str(metadata_root), str(_FIXTURES), inherited) if part),
                "PYTHONNOUSERSITE": "1",
                "PYNE_BROKER_LAB_OFFLINE": "1",
                "PYNE_NO_COLOR_LOG": "1",
            }
            base = [
                str(Path(sys.executable).with_name("pyne")),
                "run",
                str(script),
                "offline-lab:LABUSD@1",
                "--from",
                "2026-07-20",
                "--to",
                "2026-07-21",
                "--broker",
                "--shutdown-timeout",
                "0.1",
                "--no-log-ohlcv",
            ]

            def run_label(label: str):
                return run_subprocess(
                    [*base, "--run-label", label],
                    cwd=root,
                    timeout=8.0,
                    env=env,
                )

            return [run_label("A"), run_label("B")]

    def handle_step(self, runner: Any, step: Step) -> bool:
        if step.kind == "pyne_run_clean":
            result = self._run_pyne()
            if result.returncode != 0:
                raise AssertionError(f"clean pyne run failed: {result.stderr}\n{result.stdout}")
            return True
        if step.kind == "pyne_run_transient_connect":
            result = self._run_pyne(fault_env={"PYNE_LAB_CONNECT_FAILURES": "1"})
            if result.returncode != 0:
                raise AssertionError(f"transient-connect pyne run failed: {result.stderr}\n{result.stdout}")
            return True
        if step.kind == "pyne_run_permanent_connect":
            result = self._run_pyne(fault_env={"PYNE_LAB_CONNECT_FAILURE": "permanent"})
            output = result.stdout + result.stderr
            if result.returncode == 0 or "permanent connect failure" not in output:
                raise AssertionError(f"permanent connect did not fail fast: {result!r}")
            return True
        if step.kind == "pyne_run_shutdown_timeout":
            result = self._run_pyne(fault_env={"PYNE_LAB_STUCK_SHUTDOWN": "1"})
            output = result.stdout + result.stderr
            if result.returncode != 0 or "shutdown" not in output.lower():
                raise AssertionError(f"shutdown timeout was not handled: {result!r}")
            return True
        if step.kind == "pyne_run_startup_exception":
            result = self._run_pyne(fault_env={"PYNE_LAB_BALANCE_FAILURE": "1"})
            output = result.stdout + result.stderr
            if result.returncode == 0 or "startup balance failure" not in output:
                raise AssertionError(f"startup exception was not surfaced: {result!r}")
            return True
        if step.kind == "pyne_run_two_shared":
            results = self._run_two_shared()
            failed = [result for result in results if result.returncode != 0]
            if failed:
                raise AssertionError(f"shared-workdir runs failed: {failed!r}")
            return True
        return super().handle_step(runner, step)


def build_suite(*, mode: str, seed: int) -> tuple[Scenario, ...]:
    del mode
    return (
        Scenario(
            name="real-pyne-run-clean-lifecycle",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_clean"),),
        ),
        Scenario(
            name="real-pyne-run-transient-connect-retry",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_transient_connect"),),
        ),
        Scenario(
            name="real-pyne-run-permanent-connect-fails-fast",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_permanent_connect"),),
        ),
        Scenario(
            name="real-pyne-run-shutdown-timeout",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_shutdown_timeout"),),
        ),
        Scenario(
            name="real-pyne-run-startup-exception-closes-store",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_startup_exception"),),
        ),
        Scenario(
            name="two-real-pyne-runs-share-workdir-and-ohlcv-safely",
            profile_factory=SubprocessProfile,
            seed=seed,
            steps=(Step("pyne_run_two_shared"),),
        ),
    )
