"""Opt-in self-tests for the broker-lab infrastructure."""

import importlib.metadata
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pynecore.testing.broker_lab import (
    DeterministicScheduler,
    pairwise_cases,
    run_subprocess,
    temporary_entry_point,
)


def main() -> int:
    axes = {"side": ("buy", "sell"), "mode": ("netting", "hedged"), "runs": (1, 2)}
    assert pairwise_cases(axes, seed=73) == pairwise_cases(axes, seed=73)

    observed: list[str] = []
    scheduler = DeterministicScheduler(100)
    scheduler.schedule(20, lambda: observed.append("second"))
    scheduler.schedule(10, lambda: observed.append("first"))
    scheduler.advance(20)
    assert observed == ["first", "second"]
    assert scheduler.now_ms == 120

    with temporary_entry_point(
        group="pyne.plugin",
        name="offline-lab",
        target="pynecore.testing.broker_lab:ScenarioRunner",
    ) as metadata_root:
        distributions = list(importlib.metadata.distributions(path=[str(metadata_root)]))
        entry_points = [ep for dist in distributions for ep in dist.entry_points]
        assert any(ep.name == "offline-lab" and ep.group == "pyne.plugin" for ep in entry_points)

    with TemporaryDirectory(prefix="pyne-broker-lab-child-") as temp:
        result = run_subprocess(
            [sys.executable, "-c", "print('offline-child-ok')"],
            cwd=Path(temp),
            timeout=2.0,
            env={"PYTHONNOUSERSITE": "1", "PYNE_BROKER_LAB_OFFLINE": "1"},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "offline-child-ok"
        assert not result.stderr

    print("PASS broker-lab-selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
