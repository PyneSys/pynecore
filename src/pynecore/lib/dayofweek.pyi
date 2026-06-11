from ..types.datetime import DayOfWeek


# IDE-facing view of the function-and-namespace module: user code reads the
# constants and calls the bare name; the AST transformer resolves both at runtime.
class DayOfWeekModule:
    sunday: DayOfWeek
    monday: DayOfWeek
    tuesday: DayOfWeek
    wednesday: DayOfWeek
    thursday: DayOfWeek
    friday: DayOfWeek
    saturday: DayOfWeek

    def __call__(self, time: int | None = None, timezone: str | None = None) -> int: ...


dayofweek: DayOfWeekModule
