from typing import Literal, NamedTuple, Self
from pathlib import Path
from dataclasses import dataclass, field
from datetime import time, date, datetime

SymInfoInterval = NamedTuple("SymInfoInterval", [('day', int), ('start', time), ('end', time)])
SymInfoSession = NamedTuple("SymInfoSession", [('day', int), ('time', time)])
SymInfoScheduleVariant = NamedTuple("SymInfoScheduleVariant", [
    ('effective_from', date),
    ('opening_hours', list[SymInfoInterval]),
    ('session_starts', list[SymInfoSession]),
    ('session_ends', list[SymInfoSession]),
])
"""One era of a symbol's trading schedule, in effect from ``effective_from``
(an exchange-local trading-day date) until the next variant's date. See
:attr:`SymInfo.session_schedules`."""

# Commented TOML appended by ``save_toml`` when a symbol has no schedule history,
# so anyone opening the file sees how to add one. Valid TOML if uncommented.
_SESSION_SCHEDULE_EXAMPLE_COMMENT = """\
# Effective-dated session history (optional).
#
# Markets occasionally change their trading hours. A single static schedule then
# mis-confirms the part of a backtest that falls on the other side of the change.
# Add [[session_schedules]] blocks below -- oldest first, each self-contained --
# to describe that history. `effective_from` MUST be the first line of its block
# (an exchange-local trading-day date). A bar whose trading day is on or after a
# block's date uses that block; a date before the earliest block uses the
# earliest block. The flat opening_hours / session_starts / session_ends above
# are regenerated from the NEWEST variant on save, so edit the variants here, not
# the flat blocks.
#
# Example -- a market whose night session END moved 23:30 -> 23:00 on 2026-01-12:
#
# [[session_schedules]]
# effective_from = 2025-06-01
# [[session_schedules.opening_hours]]
# day = 0
# start = "10:00:00"
# end = "18:00:00"
# [[session_schedules.opening_hours]]
# day = 0
# start = "21:00:00"
# end = "23:30:00"
# [[session_schedules.session_starts]]
# day = 0
# time = "10:00:00"
# [[session_schedules.session_ends]]
# day = 0
# time = "23:30:00"
#
# [[session_schedules]]
# effective_from = 2026-01-12
# [[session_schedules.opening_hours]]
# day = 0
# start = "10:00:00"
# end = "18:00:00"
# [[session_schedules.opening_hours]]
# day = 0
# start = "21:00:00"
# end = "23:00:00"
# [[session_schedules.session_starts]]
# day = 0
# time = "10:00:00"
# [[session_schedules.session_ends]]
# day = 0
# time = "23:00:00\""""


def default_mincontract(sym_type: str, basecurrency: str | None = None) -> float:
    """
    Heuristic minimum order quantity step for symbols whose data source does
    not expose one.

    Crypto exchanges quote fractional lot steps (BTC pairs commonly 1e-5,
    other coins 1e-4); everything else trades in whole contracts. This is the
    last resort of the ``mincontract`` resolution chain (exchange value ->
    volume-data analysis -> heuristic), and it also fills the gap when
    loading symbol info saved before ``mincontract`` existed.

    :param sym_type: The symbol type (``SymInfo.type``).
    :param basecurrency: The symbol's base currency, if known.
    :return: The estimated minimum quantity step.
    """
    if sym_type == 'crypto':
        return 1e-05 if basecurrency == 'BTC' else 1e-04
    return 1.0


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
    mincontract: float
    """Minimum order quantity step (Pine ``syminfo.mincontract``); order sizes
    are truncated to this grid. Always positive: filled from the exchange when
    available, otherwise estimated from volume data or the
    :func:`default_mincontract` heuristic."""
    opening_hours: list[SymInfoInterval]
    session_starts: list[SymInfoSession]
    session_ends: list[SymInfoSession]
    session_schedules: list[SymInfoScheduleVariant] = field(default_factory=list)
    """Optional effective-dated session history (oldest first). When non-empty the
    flat ``opening_hours`` / ``session_starts`` / ``session_ends`` mirror the
    newest variant; consumers that need a schedule for a specific date resolve it
    via :meth:`schedule_for` / :meth:`schedule_index_for`. Empty for the common
    case of a symbol whose hours never changed."""
    timezone: str = 'UTC'

    avg_spread: float | None = None
    taker_fee: float | None = None
    maker_fee: float | None = None

    # Reference data (None when the data source does not expose it)
    country: str | None = None
    sector: str | None = None
    industry: str | None = None
    isin: str | None = None

    # Futures contract information
    expiration_date: int | None = None  # UNIX timestamp
    current_contract: str | None = None

    # Fundamentals (stocks only, None elsewhere)
    employees: int | None = None
    shareholders: int | None = None
    shares_outstanding_total: float | None = None
    shares_outstanding_float: float | None = None

    # Analyst recommendation counts
    recommendations_buy: int | None = None
    recommendations_buy_strong: int | None = None
    recommendations_date: int | None = None  # UNIX timestamp
    recommendations_hold: int | None = None
    recommendations_sell: int | None = None
    recommendations_sell_strong: int | None = None
    recommendations_total: int | None = None

    # Analyst price target information (added 2025-07-08)
    target_price_average: float | None = None
    target_price_high: float | None = None
    target_price_low: float | None = None
    target_price_median: float | None = None
    target_price_date: int | None = None  # UNIX timestamp
    target_price_estimates: int | None = None

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

        # Effective-dated session history (optional). Each variant is a
        # self-contained schedule taking effect on its ``effective_from``
        # exchange-local trading day. The list is sorted ascending and, when
        # present, overwrites the flat fields above with the NEWEST variant so
        # setup-time classification and the live "now" both see today's calendar.
        # noinspection PyShadowingNames
        def parse_effective_from(value: object) -> date:
            """Normalize a TOML local-date / local-datetime / ISO string to a date."""
            if isinstance(value, datetime):  # datetime first: it subclasses date
                return value.date()
            if isinstance(value, date):
                return value
            if isinstance(value, str):
                return date.fromisoformat(value.replace(' ', 'T').split('T')[0])
            raise ValueError("Invalid session_schedules effective_from type "
                             f"{type(value).__name__}: expected a date or YYYY-MM-DD string")

        session_schedules: list[SymInfoScheduleVariant] = []
        seen_effective: set[date] = set()
        for sched in data.get('session_schedules', []):
            if 'effective_from' not in sched:
                raise ValueError("session_schedules entry is missing 'effective_from'")
            eff = parse_effective_from(sched['effective_from'])
            if eff in seen_effective:
                raise ValueError(f"Duplicate session_schedules effective_from: {eff}")
            seen_effective.add(eff)
            session_schedules.append(SymInfoScheduleVariant(
                effective_from=eff,
                opening_hours=[SymInfoInterval(day=oh['day'], start=parse_time(oh['start']),
                                               end=parse_time(oh['end']))
                              for oh in sched.get('opening_hours', [])],
                session_starts=[SymInfoSession(day=s['day'], time=parse_time(s['time']))
                                for s in sched.get('session_starts', [])],
                session_ends=[SymInfoSession(day=s['day'], time=parse_time(s['time']))
                             for s in sched.get('session_ends', [])],
            ))
        session_schedules.sort(key=lambda v: v.effective_from)
        if session_schedules:
            # History is the source of truth: the flat blocks mirror the newest.
            newest = session_schedules[-1]
            opening_hours = newest.opening_hours
            session_starts = newest.session_starts
            session_ends = newest.session_ends

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
            # Files saved before mincontract existed fall back to the heuristic
            mincontract=(float(symbol.get('mincontract', 0.0))
                         or default_mincontract(symbol['type'], symbol.get('basecurrency'))),
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            session_schedules=session_schedules,
            timezone=symbol.get('timezone', 'UTC'),
            volumetype=symbol.get('volumetype', 'base'),
            avg_spread=symbol.get('avg_spread'),
            taker_fee=symbol.get('taker_fee'),
            maker_fee=symbol.get('maker_fee'),
            country=symbol.get('country'),
            sector=symbol.get('sector'),
            industry=symbol.get('industry'),
            isin=symbol.get('isin'),
            expiration_date=symbol.get('expiration_date'),
            current_contract=symbol.get('current_contract'),
            employees=symbol.get('employees'),
            shareholders=symbol.get('shareholders'),
            shares_outstanding_total=symbol.get('shares_outstanding_total'),
            shares_outstanding_float=symbol.get('shares_outstanding_float'),
            recommendations_buy=symbol.get('recommendations_buy'),
            recommendations_buy_strong=symbol.get('recommendations_buy_strong'),
            recommendations_date=symbol.get('recommendations_date'),
            recommendations_hold=symbol.get('recommendations_hold'),
            recommendations_sell=symbol.get('recommendations_sell'),
            recommendations_sell_strong=symbol.get('recommendations_sell_strong'),
            recommendations_total=symbol.get('recommendations_total'),
            target_price_average=symbol.get('target_price_average'),
            target_price_high=symbol.get('target_price_high'),
            target_price_low=symbol.get('target_price_low'),
            target_price_median=symbol.get('target_price_median'),
            target_price_date=symbol.get('target_price_date'),
            target_price_estimates=symbol.get('target_price_estimates')
        )

    def save_toml(self, path: Path):
        """
        Save SymInfo object to TOML-like fmt without dependencies.
        Organizes data under [symbol] section.
        None values are commented out with '#key ='

        An existing [download] section (written by `pyne data download`, see
        core.download_info) is preserved verbatim across the rewrite.

        :param path: Path to save the file
        """
        from .download_info import extract_download_section

        preserved_download = None
        if path.exists():
            try:
                preserved_download = extract_download_section(path.read_text(encoding='utf-8'))
            except OSError:
                pass

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
                    'mincontract', 'timezone', 'volumetype', 'avg_spread', 'taker_fee', 'maker_fee',
                    'country', 'sector', 'industry', 'isin',
                    'expiration_date', 'current_contract',
                    'employees', 'shareholders',
                    'shares_outstanding_total', 'shares_outstanding_float',
                    'recommendations_buy', 'recommendations_buy_strong', 'recommendations_date',
                    'recommendations_hold', 'recommendations_sell', 'recommendations_sell_strong',
                    'recommendations_total',
                    'target_price_average', 'target_price_high', 'target_price_low',
                    'target_price_median', 'target_price_date', 'target_price_estimates']:
            lines.append(format_field(key, getattr(self, key)))

        # Arrays of tables. With an effective-dated history the flat blocks mirror
        # the NEWEST variant (the load-time invariant); regenerate them from it on
        # save so a programmatic edit that left ``self.*`` stale cannot leak the old
        # schedule into the flat block -- this is what the comment below promises.
        if self.session_schedules:
            newest = self.session_schedules[-1]
            flat_opening_hours = newest.opening_hours
            flat_session_starts = newest.session_starts
            flat_session_ends = newest.session_ends
        else:
            flat_opening_hours = self.opening_hours
            flat_session_starts = self.session_starts
            flat_session_ends = self.session_ends

        lines.append("\n# Opening hours")
        for oh in flat_opening_hours:
            lines.append("[[opening_hours]]")
            lines.append(f"day = {oh.day}")
            lines.append(f'start = "{time_to_str(oh.start)}"')
            lines.append(f'end = "{time_to_str(oh.end)}"')
            lines.append("")

        lines.append("# Session starts")
        for s in flat_session_starts:
            lines.append("[[session_starts]]")
            lines.append(f"day = {s.day}")
            lines.append(f'time = "{time_to_str(s.time)}"')
            lines.append("")

        lines.append("# Session ends")
        for s in flat_session_ends:
            lines.append("[[session_ends]]")
            lines.append(f"day = {s.day}")
            lines.append(f'time = "{time_to_str(s.time)}"')
            lines.append("")

        # Effective-dated session history, or a commented example when absent.
        if self.session_schedules:
            lines.append("# Effective-dated session history (oldest first). The flat")
            lines.append("# opening_hours / session_starts / session_ends above are regenerated")
            lines.append("# from the newest variant on save -- edit the variants below.")
            for variant in self.session_schedules:
                lines.append("[[session_schedules]]")
                lines.append(f"effective_from = {variant.effective_from.isoformat()}")
                for oh in variant.opening_hours:
                    lines.append("[[session_schedules.opening_hours]]")
                    lines.append(f"day = {oh.day}")
                    lines.append(f'start = "{time_to_str(oh.start)}"')
                    lines.append(f'end = "{time_to_str(oh.end)}"')
                for s in variant.session_starts:
                    lines.append("[[session_schedules.session_starts]]")
                    lines.append(f"day = {s.day}")
                    lines.append(f'time = "{time_to_str(s.time)}"')
                for s in variant.session_ends:
                    lines.append("[[session_schedules.session_ends]]")
                    lines.append(f"day = {s.day}")
                    lines.append(f'time = "{time_to_str(s.time)}"')
                lines.append("")
        else:
            lines.append(_SESSION_SCHEDULE_EXAMPLE_COMMENT)
            lines.append("")

        if preserved_download:
            lines.append(preserved_download)
            lines.append("")

        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    @property
    def has_schedule_history(self) -> bool:
        """``True`` when an effective-dated session history is present."""
        return bool(self.session_schedules)

    def schedule_index_for(self, d: date) -> int:
        """
        Index into :attr:`session_schedules` of the variant effective on ``d``.

        The chain is sorted ascending by ``effective_from`` at load time, so this
        is the last variant with ``effective_from <= d``. A date before the first
        variant clamps to index ``0`` (the oldest variant), never to the flat
        fields -- those mirror the NEWEST schedule and would reintroduce the stale
        divergence. Requires a non-empty :attr:`session_schedules`.

        :param d: Exchange-local trading-day date.
        :return: Index into :attr:`session_schedules`.
        """
        idx = 0
        for i, variant in enumerate(self.session_schedules):
            if variant.effective_from <= d:
                idx = i
            else:
                break
        return idx

    def schedule_for(self, d: date) -> tuple[
            list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """
        Resolve the session schedule effective on date ``d``.

        Without history this returns the flat :attr:`opening_hours` /
        :attr:`session_starts` / :attr:`session_ends`, so a migrated caller is
        bit-identical to the pre-history behaviour. With history it returns the
        variant selected by :meth:`schedule_index_for`.

        :param d: Exchange-local trading-day date.
        :return: ``(opening_hours, session_starts, session_ends)`` effective on ``d``.
        """
        if not self.session_schedules:
            return self.opening_hours, self.session_starts, self.session_ends
        v = self.session_schedules[self.schedule_index_for(d)]
        return v.opening_hours, v.session_starts, v.session_ends


def mintick_decimals(mintick: float) -> int:
    """
    Number of decimal places implied by a symbol's ``mintick``.

    Derived from ``str(mintick)`` (Python's shortest round-trip repr), so
    ``0.05`` yields ``2`` and ``0.025`` yields ``3`` without exposing float
    dust (``f"{0.05:.20f}"`` would be ``"0.05000000000000000278"``). This
    mirrors the ``format.mintick`` logic in :mod:`pynecore.lib.string` and is
    correct for fractional tick grids, where ``pricescale`` may be
    ``round(1 / mintick)`` (e.g. ``20`` for ``0.05``) rather than a power of
    ten.

    :param mintick: The symbol's minimum tick size.
    :return: Decimal place count, ``0`` for non-positive or integer ticks.
    """
    if not mintick or mintick <= 0:
        return 0
    tick_str = str(mintick)
    if 'e' in tick_str or 'E' in tick_str:
        # Scientific notation (very small ticks): expand without float dust.
        tick_str = f"{mintick:.20f}".rstrip('0')
    if '.' not in tick_str:
        return 0
    return len(tick_str.rstrip('0').split('.')[1])
