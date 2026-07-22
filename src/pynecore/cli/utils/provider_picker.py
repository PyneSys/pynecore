"""Interactive picker for installed data-provider plugins."""

from typing import NamedTuple

from .broker_picker import BrokerPicker


class ProviderChoice(NamedTuple):
    """Provider entry-point name and its optional human-readable name."""

    id: str
    name: str = ""


class ProviderPicker(BrokerPicker):
    """Single-pane provider picker sharing navigation with the broker picker."""

    def __init__(self, providers: list[ProviderChoice]):
        super().__init__(providers, provider_name="Data", item_name="providers")
