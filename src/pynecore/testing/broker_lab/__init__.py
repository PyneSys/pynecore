"""Public API for deterministic offline broker conformance testing."""

from .generate import pairwise_cases
from .model import (
    Scenario,
    ScenarioInvariantError,
    ScenarioResult,
    Step,
    VenueProfile,
)
from .reference import (
    HedgedReferenceVenueProfile,
    ReferenceBroker,
    ReferenceVenueProfile,
    VenueOrder,
    VenueState,
)
from .runner import RunRuntime, ScenarioRunner
from .scheduler import DeterministicScheduler, ScheduledEvent
from .subprocess import SubprocessResult, run_subprocess, temporary_entry_point

__all__ = [
    "DeterministicScheduler",
    "HedgedReferenceVenueProfile",
    "ReferenceBroker",
    "ReferenceVenueProfile",
    "RunRuntime",
    "Scenario",
    "ScenarioInvariantError",
    "ScenarioResult",
    "ScenarioRunner",
    "ScheduledEvent",
    "Step",
    "SubprocessResult",
    "VenueProfile",
    "VenueOrder",
    "VenueState",
    "pairwise_cases",
    "run_subprocess",
    "temporary_entry_point",
]
