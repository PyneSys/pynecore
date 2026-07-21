"""Isolated subprocess helpers for opt-in CLI lifecycle scenarios."""

import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class SubprocessResult:
    """Bounded child-process result captured by the broker lab."""

    returncode: int
    stdout: str
    stderr: str


def run_subprocess(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 10.0,
    env: Mapping[str, str] | None = None,
) -> SubprocessResult:
    """Run a child with a hard external timeout and captured output."""
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=child_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"broker-lab subprocess exceeded {timeout:.1f}s: {args!r}") from exc
    return SubprocessResult(completed.returncode, completed.stdout, completed.stderr)


@contextmanager
def temporary_entry_point(
    *,
    group: str,
    name: str,
    target: str,
) -> Iterator[Path]:
    """Inject disposable ``.dist-info`` metadata without production hooks."""
    with TemporaryDirectory(prefix="pyne-broker-lab-entrypoint-") as temp:
        root = Path(temp)
        dist_info = root / "offline_broker_lab-0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: offline-broker-lab\nVersion: 0\n",
            encoding="utf-8",
        )
        (dist_info / "entry_points.txt").write_text(
            f"[{group}]\n{name} = {target}\n",
            encoding="utf-8",
        )
        yield root


def python_executable() -> str:
    """Return the interpreter running the opt-in lab."""
    return sys.executable
