from dataclasses import dataclass
from datetime import time
from typing import Set

from ..types.base import IntEnum


@dataclass(frozen=True)
class SessionInfo:
    """
    Session information containing time range, valid days, and timezone.
    
    This dataclass represents a trading session specification that can be used
    to determine if a given timestamp falls within the session bounds.
    """
    start_time: time
    end_time: time
    days: Set[int]  # 1=Sunday, 2=Monday, ..., 7=Saturday (TradingView format)
    timezone: str
    is_overnight: bool = False
    
    def __post_init__(self):
        """Validate session parameters after initialization."""
        # Check if this is an overnight session
        if self.start_time > self.end_time:
            object.__setattr__(self, 'is_overnight', True)
        
        # Validate days
        for day in self.days:
            if not 1 <= day <= 7:
                raise ValueError(f"Invalid day: {day}. Days must be between 1 (Sunday) and 7 (Saturday)")


class Session(IntEnum):
    __slots__ = ()
