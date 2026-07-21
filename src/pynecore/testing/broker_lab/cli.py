"""Opt-in command-line runner for broker conformance suites."""

import argparse
from collections.abc import Sequence
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from .model import Scenario
from .runner import ScenarioRunner


def _load_suite(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"pyne_broker_lab_suite_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load broker-lab suite: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scenarios(module: ModuleType, mode: str, seed: int) -> Sequence[Scenario]:
    builder = getattr(module, "build_suite", None)
    if builder is not None:
        return tuple(builder(mode=mode, seed=seed))
    scenarios = tuple(getattr(module, "SCENARIOS", ()))
    if mode == "smoke":
        return tuple(s for s in scenarios if "smoke" in s.tags)
    return scenarios


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pynecore.testing.broker_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an opt-in broker-lab suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--mode", choices=("smoke", "extended"), default="smoke")
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--scenario")
    run.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    module = _load_suite(args.suite.resolve())
    scenarios = list(_scenarios(module, args.mode, args.seed))
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario.name == args.scenario]
        if not scenarios:
            parser.error(f"unknown scenario: {args.scenario}")
    results = [ScenarioRunner().run(scenario) for scenario in scenarios]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name} seed={result.seed}")
        if result.violation:
            print(f"  {result.violation}")
            print(
                "  reproduce: python -m pynecore.testing.broker_lab run "
                f"{args.suite} --mode {args.mode} {result.reproduction}"
            )
            for step in result.minimized_steps:
                print(f"    {step.kind} run={step.run} values={step.values!r}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                [
                    {
                        "name": result.name,
                        "passed": result.passed,
                        "seed": result.seed,
                        "violation": result.violation,
                        "reproduction": result.reproduction,
                        "minimized_steps": [
                            {"kind": s.kind, "run": s.run, "values": s.values} for s in result.minimized_steps
                        ],
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if all(result.passed for result in results) else 1
