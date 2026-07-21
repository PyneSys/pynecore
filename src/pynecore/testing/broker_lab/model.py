"""Public scenario model for the offline broker conformance lab."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Step:
    """One deterministic broker scenario transition."""

    kind: str
    run: str = "main"
    values: dict[str, Any] = field(default_factory=dict)
    check_invariants: bool = True


class VenueProfile(Protocol):
    """Venue semantics consumed by :class:`ScenarioRunner`.

    Profiles may return a real plugin instance or a deliberately small broker
    implementation. Plugin suites normally return the real plugin with only its
    HTTP, WebSocket, or wire transport replaced.
    """

    plugin_name: str
    account_id: str
    symbol: str
    timeframe: str

    def create_broker(self, run_name: str, store_ctx: Any) -> Any:
        """Create the broker used by one logical run."""
        ...

    def handle_step(self, runner: Any, step: Step) -> bool:
        """Apply a profile-specific step and return whether it was handled."""
        ...

    def check_invariants(self, runner: Any) -> Sequence[str]:
        """Return invariant violations after the current transition."""
        ...

    def close(self) -> None:
        """Release profile-owned resources."""
        ...


ProfileFactory = Callable[[], VenueProfile]


@dataclass(frozen=True)
class Scenario:
    """A reproducible sequence executed against a fresh venue profile."""

    name: str
    profile_factory: ProfileFactory
    steps: tuple[Step, ...]
    runs: tuple[str, ...] = ("main",)
    seed: int = 0
    tags: frozenset[str] = frozenset({"smoke"})
    expected_violation: str | None = None


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome and reproduction data for one scenario."""

    name: str
    passed: bool
    seed: int
    executed_steps: tuple[Step, ...]
    violation: str | None = None
    minimized_steps: tuple[Step, ...] = ()
    artifact_dir: Path | None = None

    @property
    def reproduction(self) -> str:
        """Return the stable seed fragment used by CLI reproduction."""
        return f"--scenario {self.name} --seed {self.seed}"


class ScenarioInvariantError(AssertionError):
    """Raised when a step violates a broker-lab invariant."""
