from datetime import datetime, timedelta

from rich.text import Text
from rich.progress import ProgressColumn


class DateColumn(ProgressColumn):
    """Custom progress column showing the current date being processed.

    The start time can come from the constructor (when the task spans a
    single fixed range) or from a per-task ``start_time`` field — useful
    for retry loops that re-anchor the start as they extend the request
    window without spawning a new spinner each time.
    """

    def __init__(self, start_time: datetime | None = None):
        super().__init__()
        self.start_time = start_time

    def render(self, task) -> Text:
        start_time = task.fields.get('start_time', self.start_time)
        if start_time is None:
            return Text("")
        current_date = start_time + timedelta(seconds=task.completed)
        return Text(current_date.strftime("%Y-%m-%d %H:%M"), style="magenta")
