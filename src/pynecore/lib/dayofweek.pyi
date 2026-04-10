from ..core.callable_module import CallableModule
from ..types.datetime import DayOfWeek


class DayOfWeekModule(CallableModule):
    sunday: DayOfWeek
    monday: DayOfWeek
    tuesday: DayOfWeek
    wednesday: DayOfWeek
    thursday: DayOfWeek
    friday: DayOfWeek
    saturday: DayOfWeek

    def __call__(self, time: int | None = None, timezone: str | None = None) -> int: ...


dayofweek: DayOfWeekModule = DayOfWeekModule(__name__)
