from typing import Literal, NamedTuple, Self
from pathlib import Path
from dataclasses import dataclass
from datetime import time

SymInfoInterval = NamedTuple("SymInfoInterval", [('day', int), ('start', time), ('end', time)])
SymInfoSession = NamedTuple("SymInfoSession", [('day', int), ('time', time)])


@dataclass(kw_only=True, slots=True)
class SymInfo:
    """
    Symbol information dataclass

    It is stored in TOML format in the working directory. It is initially from the provider, but
    users can edit according to their needs to make it compatible with the TradingView platform.
    It is almost impossible to make providers fully compatible, this is why users may need to
    edit the symbol information in very specific cases.
    """
    prefix: str
    description: str
    ticker: str
    currency: str
    basecurrency: str | None = None
    period: str
    type: Literal[
        "stock", "fund", "dr", "right", "bond", "warrant", "structured", "index", "forex",
        "futures", "spread", "economic", "fundamental", "crypto", "spot", "swap", "option",
        "commodity", "other"
    ]
    volumetype: Literal["base", "quote", "tick", "n/a"] = 'base'
    mintick: float
    pricescale: int
    minmove: int = 1
    pointvalue: float
    opening_hours: list[SymInfoInterval]
    session_starts: list[SymInfoSession]
    session_ends: list[SymInfoSession]
    timezone: str = 'UTC'

    avg_spread: float | None = None
    taker_fee: float | None = None
    maker_fee: float | None = None

    # Analyst price target information (added 2025-07-08)
    target_price_average: float | None = None
    target_price_high: float | None = None
    target_price_low: float | None = None
    target_price_date: int | None = None  # UNIX timestamp

    @classmethod
    def load_toml(cls, path: Path) -> Self:
        """
        Load SymInfo object from TOML file.

        :param path: Path to the TOML file
        :return: SymInfo instance
        :raises ValueError: If required fields are missing or invalid
        """
        import tomllib

        with open(path, 'rb') as f:
            data = tomllib.load(f)

        if 'symbol' not in data:
            raise ValueError("Missing [symbol] section in TOML")

        symbol = data['symbol']

        # Parse time strings in arrays
        # noinspection PyShadowingNames
        def parse_time(time_str: str) -> time:
            """Parse time string in HH:MM:SS fmt"""
            h, m, s = map(int, time_str.split(':'))
            return time(h, m, s)

        # Convert opening hours
        opening_hours = []
        for oh in data.get('opening_hours', []):
            opening_hours.append(SymInfoInterval(
                day=oh['day'],
                start=parse_time(oh['start']),
                end=parse_time(oh['end'])
            ))

        # Convert session times
        session_starts = []
        for s in data.get('session_starts', []):
            session_starts.append(SymInfoSession(
                day=s['day'],
                time=parse_time(s['time'])
            ))

        session_ends = []
        for s in data.get('session_ends', []):
            session_ends.append(SymInfoSession(
                day=s['day'],
                time=parse_time(s['time'])
            ))

        # Create instance with all fields
        return cls(
            prefix=symbol['prefix'],
            description=symbol['description'],
            ticker=symbol['ticker'],
            currency=symbol['currency'],
            basecurrency=symbol['basecurrency'] if 'basecurrency' in symbol else None,
            period=symbol['period'],
            type=symbol['type'],
            mintick=symbol['mintick'],
            pricescale=symbol['pricescale'],
            minmove=symbol.get('minmove', 1),
            pointvalue=symbol['pointvalue'],
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            timezone=symbol.get('timezone', 'UTC'),
            volumetype=symbol.get('volumetype', 'base'),
            avg_spread=symbol.get('avg_spread'),
            taker_fee=symbol.get('taker_fee'),
            maker_fee=symbol.get('maker_fee'),
            target_price_average=symbol.get('target_price_average'),
            target_price_high=symbol.get('target_price_high'),
            target_price_low=symbol.get('target_price_low'),
            target_price_date=symbol.get('target_price_date')
        )

    def save_toml(self, path: Path):
        """
        Save SymInfo object to TOML-like fmt without dependencies.
        Organizes data under [symbol] section.
        None values are commented out with '#key ='

        :param path: Path to save the file
        """

        def time_to_str(t):
            """Convert time object to string"""
            return t.strftime("%H:%M:%S")

        # noinspection PyShadowingNames
        def format_field(key, value):
            """Format field to TOML string"""
            if value is None:
                return f"#{key} ="
            if isinstance(value, str):
                return f"{key} = \"{value}\""
            if isinstance(value, bool):
                return f"{key} = {str(value).lower()}"
            if isinstance(value, float):
                return f"{key} = {value:.8f}"
            return f"{key} = {value}"

        lines = ["[symbol]"]  # Root table/section

        # Basic fields
        for key in ['prefix', 'description', 'ticker', 'currency', 'basecurrency',
                    'period', 'type', 'mintick', 'pricescale', 'minmove', 'pointvalue',
                    'timezone', 'volumetype', 'avg_spread', 'taker_fee', 'maker_fee',
                    'target_price_average', 'target_price_high', 'target_price_low', 'target_price_date']:
            lines.append(format_field(key, getattr(self, key)))

        # Arrays of tables
        lines.append("\n# Opening hours")
        for oh in self.opening_hours:
            lines.append("[[opening_hours]]")
            lines.append(f"day = {oh.day}")
            lines.append(f'start = "{time_to_str(oh.start)}"')
            lines.append(f'end = "{time_to_str(oh.end)}"')
            lines.append("")

        lines.append("# Session starts")
        for s in self.session_starts:
            lines.append("[[session_starts]]")
            lines.append(f"day = {s.day}")
            lines.append(f'time = "{time_to_str(s.time)}"')
            lines.append("")

        lines.append("# Session ends")
        for s in self.session_ends:
            lines.append("[[session_ends]]")
            lines.append(f"day = {s.day}")
            lines.append(f'time = "{time_to_str(s.time)}"')
            lines.append("")

        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
