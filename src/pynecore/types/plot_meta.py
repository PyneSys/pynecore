from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .color import Color


@dataclass(slots=True)
class PlotMeta:
    """
    Static (registered-once) metadata for a plot-family output.

    Covers every plot family (``plot``, ``plotshape``, ``plotchar``, ``plotarrow``,
    ``plotcandle``, ``plotbar``, ``bgcolor``, ``barcolor``, ``hline``, ``fill``); the
    serializer drops ``None`` fields and applies kind-specific defaults. Per-bar values
    (the series data and dynamic color channels) live elsewhere; this object only holds
    what is fixed for the whole run.
    """
    id: str
    kind: str  # 'plot'|'shape'|'char'|'arrow'|'candle'|'bar'|'bgcolor'|'barcolor'|'hline'|'fill'

    title: str | None = None
    color: Color | None = None
    linewidth: int = 1
    style: Any = None
    trackprice: bool = False
    histbase: float = 0.0
    offset: int = 0
    join: bool = False
    editable: bool = True
    show_last: int | None = None
    display: Any = None
    format: str | None = None
    precision: int | None = None
    force_overlay: bool = False

    char: str | None = None
    location: Any = None
    size: Any = None
    text: str | None = None
    textcolor: Color | None = None

    colorup: Color | None = None
    colordown: Color | None = None
    minheight: int | None = None
    maxheight: int | None = None

    wickcolor: Color | None = None
    bordercolor: Color | None = None

    price: float | None = None
    linestyle: Any = None

    plot1: str | None = None
    plot2: str | None = None
    hline1: str | None = None
    hline2: str | None = None

    fillgaps: bool = False

    dynamic: bool = False

    _on_change: Callable[[str], None] | None = field(default=None, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        notify = getattr(self, '_on_change', None)
        old = getattr(self, name, None) if notify is not None else None
        object.__setattr__(self, name, value)
        if notify is None or name == '_on_change' or old is value:
            return
        # Equal color objects can still have different mutation lifetimes.
        if not isinstance(old, Color) and not isinstance(value, Color) \
                and type(old) is type(value) and old == value:
            return
        notify(name)


_COLOR_FIELDS = ('color', 'textcolor', 'colorup', 'colordown', 'wickcolor', 'bordercolor')
_MISSING = object()


class PlotMetaRegistry(dict[str, PlotMeta]):
    """Per-run metadata with independent reader cursors and a shared wire cache.

    Scalar assignments notify the registry. Mutable colors are sampled by their
    integer value, once per distinct referenced object, without changing Color's
    allocation or assignment paths. Readers inspect the metadata registry only
    when a revision changes; a fast path never serializes unchanged records.
    """

    def __init__(self) -> None:
        super().__init__()
        self._revision = 0
        self._changed: dict[str, int] = {}
        self._colors: dict[int, tuple[Color, int, set[str]]] = {}
        self._color_ids: dict[str, set[int]] = {}
        # key -> (last checked revision, last wire change, serialized record)
        self._wire: dict[str, tuple[int, int, dict[str, Any]]] = {}

    def __setitem__(self, key: str, meta: PlotMeta) -> None:
        if key in self:
            self._detach(key)
        super().__setitem__(key, meta)
        self._watch_colors(key, meta)

        def changed(name: str) -> None:
            if name in _COLOR_FIELDS:
                self._unwatch_colors(key)
                self._watch_colors(key, meta)
            self._touch(key)

        meta._on_change = changed
        self._wire.pop(key, None)
        self._touch(key)

    def __delitem__(self, key: str) -> None:
        self._detach(key)
        super().__delitem__(key)
        self._changed.pop(key, None)
        self._wire.pop(key, None)
        self._revision += 1

    def clear(self) -> None:
        for meta in self.values():
            meta._on_change = None
        super().clear()
        self._changed.clear()
        self._colors.clear()
        self._color_ids.clear()
        self._wire.clear()
        # Never reuse revisions: an existing reader can survive a run reset.
        self._revision += 1

    def update(self, other: Mapping[str, PlotMeta] | Iterable[tuple[str, PlotMeta]] = (),
               **kwargs: PlotMeta) -> None:
        for key, meta in dict(other, **kwargs).items():
            self[key] = meta

    def setdefault(self, key: str, default: PlotMeta | None = None) -> PlotMeta:
        if key not in self:
            if default is None:
                raise TypeError('Plot metadata must be a PlotMeta instance')
            self[key] = default
        return self[key]

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        meta = self[key]
        del self[key]
        return meta

    def popitem(self) -> tuple[str, PlotMeta]:
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def __ior__(self, other: Mapping[str, PlotMeta]) -> 'PlotMetaRegistry':
        self.update(other)
        return self

    def collect_changes(self, cursor: int, serialize: Callable[[PlotMeta], dict[str, Any]]) \
            -> tuple[int, list[dict[str, Any]]]:
        """Read latest changed records without consuming another reader's updates.

        :param cursor: Revision returned by the reader's previous call; 0 for a new reader
        :param serialize: The metadata serializer; evaluated once per modified record
        :return: Current revision and detached records whose wire value changed
        """
        self._poll_colors()
        if cursor == self._revision:
            return cursor, []
        records = []
        for key, revision in self._changed.items():
            if revision <= cursor:
                continue
            cached = self._wire.get(key)
            if cached is None or cached[0] != revision:
                record = serialize(self[key])
                wire_revision = revision
                if cached is not None and cached[2] == record:
                    wire_revision, record = cached[1], cached[2]
                cached = self._wire[key] = (revision, wire_revision, record)
            if cached[1] > cursor:
                records.append(cached[2].copy())
        return self._revision, records

    def _touch(self, key: str) -> None:
        self._revision += 1
        self._changed[key] = self._revision

    def _watch_colors(self, key: str, meta: PlotMeta) -> None:
        ids = self._color_ids[key] = set()
        for name in _COLOR_FIELDS:
            color = getattr(meta, name)
            if not isinstance(color, Color):
                continue
            cid = id(color)
            ids.add(cid)
            entry = self._colors.get(cid)
            if entry is None:
                self._colors[cid] = (color, color.value, {key})
            else:
                entry[2].add(key)

    def _unwatch_colors(self, key: str) -> None:
        for cid in self._color_ids.pop(key, ()):
            keys = self._colors[cid][2]
            keys.discard(key)
            if not keys:
                del self._colors[cid]

    def _detach(self, key: str) -> None:
        self[key]._on_change = None
        self._unwatch_colors(key)

    def _poll_colors(self) -> None:
        for cid, (color, value, keys) in self._colors.items():
            current = color.value
            if current != value:
                self._colors[cid] = (color, current, keys)
                for key in keys:
                    self._touch(key)
