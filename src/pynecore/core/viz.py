"""
Plot-style and drawing visual-data serialization for PyneCore.

Turns the per-run plot-family metadata (``lib._plot_meta``), per-bar plot values
and dynamic color channels (``lib._plot_data`` / ``lib._viz_dyn``) and the live
drawing registries (line/label/box/table/polyline/linefill) into a compact
NDJSON stream. One JSON object per line; record kinds are tagged by ``"t"``:
``hdr`` (header), ``meta`` (a plot-family definition), ``bar`` (per-bar values +
changed colors), ``ev`` (a drawing create/update/delete event when journaling),
``drawings`` (a full end-of-run snapshot) and ``end``.

Enum reverse maps are built once, at import time, by introspecting the ``lib.*``
constant modules, so there is no hand-maintained name table to drift.
"""
import json
import math
from pathlib import Path
from typing import Any

from pynecore import lib
from pynecore.lib import (
    plot as _plot_mod, display as _display_mod, shape as _shape_mod,
    location as _location_mod, size as _size_mod, xloc as _xloc_mod,
    yloc as _yloc_mod, extend as _extend_mod, position as _position_mod,
    line as _line_mod, label as _label_mod, hline as _hline_mod,
    text as _text_mod, font as _font_mod, format as _format_mod,
    box as _box_mod, table as _table_mod, polyline as _polyline_mod,
    linefill as _linefill_mod,
)
from pynecore.types import script_type as _script_type_mod
from pynecore.types.base import reset_vid_counter
from pynecore.types.color import Color
from pynecore.types.na import NA
from pynecore.types.plot import PlotEnum
from pynecore.types.display import Display
from pynecore.types.script_type import ScriptType
from pynecore.types.shape import Shape
from pynecore.types.location import Location
from pynecore.types.size import Size
from pynecore.types.xloc import XLoc
from pynecore.types.yloc import YLoc
from pynecore.types.extend import Extend
from pynecore.types.position import Position
from pynecore.types.line import LineEnum
from pynecore.types.label import LabelStyleEnum
from pynecore.types.hline import HLineEnum
from pynecore.types.text import AlignEnum, FormatEnum, WrapEnum
from pynecore.types.font import FontFamilyEnum
from pynecore.types.format import Format


#
# Enum name maps — built once by module introspection
#

def _names(mod, cls, strip: str = '') -> dict:
    """Reverse-map every ``cls`` constant in ``mod`` to its attribute name.

    :param mod: The module holding the constants (e.g. ``lib.shape``)
    :param cls: The constant type to collect (e.g. ``Shape``)
    :param strip: Optional attribute-name prefix to drop (e.g. ``'style_'``)
    :return: ``{constant_value: attr_name}``. StrLiteral values compare equal to
             their raw string, so a single entry serves both object and str lookups.
    """
    return {v: (k[len(strip):] if strip and k.startswith(strip) else k)
            for k, v in vars(mod).items() if isinstance(v, cls)}


PLOT_STYLE = _names(_plot_mod, PlotEnum, strip='style_')
PLOT_LINESTYLE = _names(_plot_mod, PlotEnum, strip='linestyle_')
DISPLAY = _names(_display_mod, Display)
SCRIPT_TYPE = _names(_script_type_mod, ScriptType)
SHAPE = _names(_shape_mod, Shape)
LOCATION = _names(_location_mod, Location)
SIZE = _names(_size_mod, Size)
XLOC = _names(_xloc_mod, XLoc)
YLOC = _names(_yloc_mod, YLoc)
EXTEND = _names(_extend_mod, Extend)
POSITION = _names(_position_mod, Position)
LINE_STYLE = _names(_line_mod, LineEnum, strip='style_')
LABEL_STYLE = _names(_label_mod, LabelStyleEnum, strip='style_')
HLINE_LINESTYLE = _names(_hline_mod, HLineEnum, strip='style_')
ALIGN = _names(_text_mod, AlignEnum, strip='align_')
FORMAT = _names(_text_mod, FormatEnum, strip='format_')
WRAP = _names(_text_mod, WrapEnum, strip='wrap_')
FONT = _names(_font_mod, FontFamilyEnum, strip='family_')
NUM_FORMAT = _names(_format_mod, Format)


def enum_name(value: Any, table: dict, default: Any = None) -> Any:
    """Look up ``value`` in an enum reverse map.

    :param value: The enum constant (or its raw str/int)
    :param table: A reverse map built by :func:`_names`
    :param default: Returned when ``value`` is ``None`` / unknown int enum
    :return: The attribute name, or the raw string for an unknown StrLiteral,
             or ``default``
    """
    if value is None:
        return default
    try:
        if value in table:
            return table[value]
    except TypeError:
        pass
    if isinstance(value, str):
        return str(value)
    return default


#
# Value serialization helpers
#

def color_str(c: Any) -> str | None:
    """Serialize a :class:`Color` to ``#RRGGBBAA``; ``None`` / NA -> ``None``."""
    if c is None or isinstance(c, NA):
        return None
    return f'#{c.value:08X}'


def _coord(x: Any) -> Any:
    """Drawing coordinate: NA -> ``None``, everything else unchanged."""
    return None if isinstance(x, NA) else x


def _size_val(v: Any) -> Any:
    """Cell/text size: a ``Size`` constant -> its name, an int -> the int."""
    if v is None:
        return None
    if isinstance(v, str):
        return enum_name(v, SIZE, str(v))
    return v


def json_default(o: Any) -> Any:
    """``json.dumps`` fallback: NA -> ``None``, Color -> hex, else raise."""
    if isinstance(o, NA):
        return None
    if isinstance(o, Color):
        return f'#{o.value:08X}'
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _scrub(o: Any) -> Any:
    """Replace non-finite floats with ``None`` recursively (json ``allow_nan=False``)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    return o


def _clean(d: dict) -> dict:
    """Drop keys whose value is ``None``."""
    return {k: v for k, v in d.items() if v is not None}


#
# Plot-family metadata
#

def serialize_meta(meta: Any) -> dict:
    """Serialize a :class:`PlotMeta` to a ``meta`` record.

    Enum fields are mapped to names, colors to hex, kind-specific defaults are
    applied when the source field is ``None``, and remaining ``None`` fields are
    dropped.

    :param meta: The registered plot-family metadata object
    :return: A JSON-ready ``{"t":"meta", ...}`` dict
    """
    k = meta.kind
    d: dict[str, Any] = {"t": "meta", "id": meta.id, "kind": k}

    if meta.title is not None:
        d["title"] = meta.title
    d["display"] = enum_name(meta.display, DISPLAY, "all")
    if meta.force_overlay:
        d["force_overlay"] = True
    if not meta.editable:
        d["editable"] = False
    if meta.offset:
        d["offset"] = meta.offset
    if meta.show_last is not None:
        d["show_last"] = meta.show_last
    if meta.format is not None:
        d["format"] = meta.format
    if meta.precision is not None:
        d["precision"] = meta.precision

    if k == 'plot':
        d["style"] = enum_name(meta.style, PLOT_STYLE, "line")
        d["color"] = color_str(meta.color)
        d["linewidth"] = meta.linewidth
        if meta.trackprice:
            d["trackprice"] = True
        if meta.histbase:
            d["histbase"] = meta.histbase
        if meta.join:
            d["join"] = True
    elif k == 'shape':
        d["style"] = enum_name(meta.style, SHAPE, "xcross")
        d["location"] = enum_name(meta.location, LOCATION, "abovebar")
        d["size"] = enum_name(meta.size, SIZE, "auto")
        d["color"] = color_str(meta.color)
        if meta.text is not None:
            d["text"] = meta.text
        d["textcolor"] = color_str(meta.textcolor)
    elif k == 'char':
        d["char"] = meta.char if meta.char is not None else "◆"
        d["location"] = enum_name(meta.location, LOCATION, "abovebar")
        d["size"] = enum_name(meta.size, SIZE, "auto")
        d["color"] = color_str(meta.color)
        if meta.text is not None:
            d["text"] = meta.text
        d["textcolor"] = color_str(meta.textcolor)
    elif k == 'arrow':
        d["colorup"] = color_str(meta.colorup)
        d["colordown"] = color_str(meta.colordown)
        if meta.minheight is not None:
            d["minheight"] = meta.minheight
        if meta.maxheight is not None:
            d["maxheight"] = meta.maxheight
    elif k == 'candle':
        d["color"] = color_str(meta.color)
        d["wickcolor"] = color_str(meta.wickcolor)
        d["bordercolor"] = color_str(meta.bordercolor)
    elif k == 'bar':
        d["color"] = color_str(meta.color)
    elif k == 'hline':
        d["price"] = meta.price
        d["color"] = color_str(meta.color)
        d["linewidth"] = meta.linewidth
        d["linestyle"] = enum_name(meta.linestyle, HLINE_LINESTYLE, "solid")
    elif k == 'fill':
        d["color"] = color_str(meta.color)
        if meta.plot1 is not None:
            d["plot1"] = meta.plot1
        if meta.plot2 is not None:
            d["plot2"] = meta.plot2
        if meta.hline1 is not None:
            d["hline1"] = meta.hline1
        if meta.hline2 is not None:
            d["hline2"] = meta.hline2
        if meta.fillgaps:
            d["fillgaps"] = True

    if meta.dynamic:
        d["dynamic"] = True

    return _clean(d)


#
# Drawing serialization
#

def _line_dict(l: Any) -> dict:
    return {
        "id": l.vid,
        "x1": _coord(l.x1), "y1": _coord(l.y1),
        "x2": _coord(l.x2), "y2": _coord(l.y2),
        "xloc": enum_name(l.xloc, XLOC),
        "extend": enum_name(l.extend, EXTEND),
        "color": color_str(l.color),
        "style": enum_name(l.style, LINE_STYLE),
        "width": l.width,
        "force_overlay": l.force_overlay,
    }


def _label_dict(la: Any) -> dict:
    return {
        "id": la.vid,
        "x": _coord(la.x), "y": _coord(la.y),
        "text": la.text,
        "xloc": enum_name(la.xloc, XLOC),
        "yloc": enum_name(la.yloc, YLOC),
        "color": color_str(la.color),
        "style": enum_name(la.style, LABEL_STYLE),
        "textcolor": color_str(la.textcolor),
        "size": enum_name(la.size, SIZE),
        "textalign": enum_name(la.textalign, ALIGN),
        "tooltip": la.tooltip,
        "text_font_family": enum_name(la.text_font_family, FONT),
        "force_overlay": la.force_overlay,
        "text_formatting": enum_name(la.text_formatting, FORMAT),
    }


def _box_dict(b: Any) -> dict:
    return {
        "id": b.vid,
        "left": _coord(b.left), "top": _coord(b.top),
        "right": _coord(b.right), "bottom": _coord(b.bottom),
        "border_color": color_str(b.border_color),
        "border_width": b.border_width,
        "border_style": enum_name(b.border_style, LINE_STYLE),
        "extend": enum_name(b.extend, EXTEND),
        "xloc": enum_name(b.xloc, XLOC),
        "bgcolor": color_str(b.bgcolor),
        "text": b.text,
        "text_size": _size_val(b.text_size),
        "text_color": color_str(b.text_color),
        "text_halign": enum_name(b.text_halign, ALIGN),
        "text_valign": enum_name(b.text_valign, ALIGN),
        "text_wrap": enum_name(b.text_wrap, WRAP),
        "text_font_family": enum_name(b.text_font_family, FONT),
        "text_formatting": enum_name(b.text_formatting, FORMAT),
        "force_overlay": b.force_overlay,
    }


def _cell_dict(col: int, row: int, cell: Any) -> dict:
    d = {
        "col": col, "row": row,
        "text": cell.text,
        "width": cell.width,
        "height": cell.height,
        "text_color": color_str(cell.text_color),
        "text_halign": enum_name(cell.text_halign, ALIGN),
        "text_valign": enum_name(cell.text_valign, ALIGN),
        "text_size": _size_val(cell.text_size),
        "bgcolor": color_str(cell.bgcolor),
        "tooltip": cell.tooltip,
        "text_font_family": enum_name(cell.text_font_family, FONT),
        "text_formatting": enum_name(cell.text_formatting, FORMAT),
    }
    if cell.is_merged:
        # [start_col, start_row, end_col, end_row] of the merged range; the range's
        # top-left cell renders, the others are hidden by it.
        d["merge"] = [cell.merge_start_col, cell.merge_start_row,
                      cell.merge_end_col, cell.merge_end_row]
    return d


def _table_dict(t: Any) -> dict:
    return {
        "id": t.vid,
        "position": enum_name(t.position, POSITION),
        "columns": t.columns,
        "rows": t.rows,
        "bgcolor": color_str(t.bgcolor),
        "frame_color": color_str(t.frame_color),
        "frame_width": t.frame_width,
        "border_color": color_str(t.border_color),
        "border_width": t.border_width,
        "force_overlay": t.force_overlay,
        "cells": [_cell_dict(col, row, cell) for (col, row), cell in t.cells.items()],
    }


def _point_dict(p: Any) -> dict:
    return {"index": _coord(p.index), "time": _coord(p.time), "price": _coord(p.price)}


def _polyline_dict(p: Any) -> dict:
    return {
        "id": p.vid,
        "points": [_point_dict(pt) for pt in p.points],
        "curved": p.curved,
        "closed": p.closed,
        "xloc": enum_name(p.xloc, XLOC),
        "line_color": color_str(p.line_color),
        "fill_color": color_str(p.fill_color),
        "line_style": enum_name(p.line_style, LINE_STYLE),
        "line_width": p.line_width,
        "force_overlay": p.force_overlay,
    }


def _linefill_dict(lf: Any) -> dict:
    return {
        "id": lf.vid,
        "line1": lf.line1.vid,
        "line2": lf.line2.vid,
        "color": color_str(lf.color),
        "line1_state": _line_dict(lf.line1),
        "line2_state": _line_dict(lf.line2),
    }


def serialize_drawing(obj: Any) -> dict:
    """Serialize a single drawing object by its type."""
    from pynecore.types.line import Line
    from pynecore.types.label import Label
    from pynecore.types.box import Box
    from pynecore.types.table import Table
    from pynecore.types.polyline import Polyline
    from pynecore.types.linefill import LineFill
    if isinstance(obj, Line):
        return _line_dict(obj)
    if isinstance(obj, Label):
        return _label_dict(obj)
    if isinstance(obj, Box):
        return _box_dict(obj)
    if isinstance(obj, Table):
        return _table_dict(obj)
    if isinstance(obj, Polyline):
        return _polyline_dict(obj)
    if isinstance(obj, LineFill):
        return _linefill_dict(obj)
    raise TypeError(f"Not a drawing object: {type(obj).__name__}")


def _all_drawings() -> dict:
    """``{(family, vid): serialized_state}`` for every live drawing."""
    out: dict[tuple[str, int], dict] = {}
    for l in list(_line_mod._registry):
        out[("line", l.vid)] = _line_dict(l)
    for la in list(_label_mod._registry):
        out[("label", la.vid)] = _label_dict(la)
    for b in list(_box_mod._registry):
        out[("box", b.vid)] = _box_dict(b)
    for t in list(_table_mod._registry):
        out[("table", t.vid)] = _table_dict(t)
    for p in list(_polyline_mod._registry):
        out[("polyline", p.vid)] = _polyline_dict(p)
    for lf in list(_linefill_mod._registry):
        out[("linefill", lf.vid)] = _linefill_dict(lf)
    return out


def drawings_snapshot() -> dict:
    """Full snapshot of every live drawing, grouped by family."""
    snap: dict[str, Any] = {
        "t": "drawings", "lines": [], "labels": [], "boxes": [],
        "tables": [], "polylines": [], "linefills": [],
    }
    for l in list(_line_mod._registry):
        snap["lines"].append(_line_dict(l))
    for la in list(_label_mod._registry):
        snap["labels"].append(_label_dict(la))
    for b in list(_box_mod._registry):
        snap["boxes"].append(_box_dict(b))
    for t in list(_table_mod._registry):
        snap["tables"].append(_table_dict(t))
    for p in list(_polyline_mod._registry):
        snap["polylines"].append(_polyline_dict(p))
    for lf in list(_linefill_mod._registry):
        snap["linefills"].append(_linefill_dict(lf))
    return snap


def journal_diff(shadow: dict, bar_index: int) -> list[dict]:
    """Diff the live drawings against ``shadow`` and update it in place.

    :param shadow: ``{(family, vid): state}`` from the previous bar; mutated here
    :param bar_index: The current bar index, recorded on every event
    :return: A list of ``create`` / ``update`` / ``delete`` event records
    """
    current = _all_drawings()
    events: list[dict] = []
    for key, state in current.items():
        old = shadow.get(key)
        if old is None:
            events.append({"t": "ev", "i": bar_index, "op": "create",
                           "obj": key[0], "id": key[1], "s": state})
        elif old != state:
            events.append({"t": "ev", "i": bar_index, "op": "update",
                           "obj": key[0], "id": key[1], "s": state})
    for key in shadow:
        if key not in current:
            events.append({"t": "ev", "i": bar_index, "op": "delete",
                           "obj": key[0], "id": key[1]})
    shadow.clear()
    shadow.update(current)
    return events


def reset_state() -> None:
    """Clear all plot-family and drawing state at the start of a run."""
    lib._plot_meta.clear()
    lib._plot_meta_new.clear()
    lib._viz_dyn.clear()
    lib._viz_seq.clear()
    _line_mod._registry.clear()
    _label_mod._registry.clear()
    _box_mod._registry.clear()
    _table_mod._registry.clear()
    _polyline_mod._registry.clear()
    _linefill_mod._registry.clear()
    reset_vid_counter()


def _encode_color_channel(v: Any) -> Any:
    """Encode a dynamic color channel: Color -> hex, tuple -> list, NA -> None."""
    if v is None or isinstance(v, NA):
        return None
    if isinstance(v, Color):
        return color_str(v)
    if isinstance(v, tuple):
        return [_encode_color_channel(x) for x in v]
    return v


class VizWriter:
    """Buffered main-thread NDJSON writer for plot / drawing visual data."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f: Any = None
        self._last_colors: dict[str, Any] = {}
        self.bars = 0

    def open(self) -> None:
        self._last_colors.clear()
        self.bars = 0
        self._f = open(self.path, 'w', buffering=1 << 16)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def is_open(self) -> bool:
        return self._f is not None

    def _emit(self, obj: dict) -> None:
        self._f.write(json.dumps(_scrub(obj), separators=(',', ':'),
                                 default=json_default, allow_nan=False))
        self._f.write('\n')

    def write_header(self, script: Any, syminfo: Any, journal: bool) -> None:
        """Write the ``hdr`` record with script + symbol context."""
        script_rec = _clean({
            "type": enum_name(script.script_type, SCRIPT_TYPE),
            "title": script.title,
            "shorttitle": script.shorttitle,
            "overlay": script.overlay,
            "format": enum_name(script.format, NUM_FORMAT),
            "precision": script.precision,
            "explicit_plot_zorder": script.explicit_plot_zorder,
            "max_lines_count": script.max_lines_count,
            "max_labels_count": script.max_labels_count,
            "max_boxes_count": script.max_boxes_count,
            "max_polylines_count": script.max_polylines_count,
        })
        syminfo_rec = _clean({
            "tickerid": getattr(syminfo, 'tickerid', None) or None,
            "prefix": getattr(syminfo, 'prefix', None) or None,
            "ticker": getattr(syminfo, 'ticker', None) or None,
            "timeframe": getattr(syminfo, 'period', None) or None,
            "mintick": getattr(syminfo, 'mintick', None) or None,
            "timezone": getattr(syminfo, 'timezone', None) or None,
            "type": getattr(syminfo, 'type', None) or None,
        })
        self._emit({"t": "hdr", "v": 1, "script": script_rec,
                    "syminfo": syminfo_rec, "journal": journal})

    def write_bar(self, bar_index: int, time_ms: int, values: dict, dyn: dict) -> None:
        """Drain pending metas, then write one ``bar`` record.

        ``"v"`` holds the per-bar plot values, ``"c"`` the color channels that
        changed since the last emitted value. Both keys are omitted when empty.
        """
        if lib._plot_meta_new:
            for meta in lib._plot_meta_new:
                self._emit(serialize_meta(meta))
            lib._plot_meta_new.clear()

        rec: dict[str, Any] = {"t": "bar", "i": bar_index, "time": time_ms}
        if values:
            rec["v"] = dict(values)
        c: dict[str, Any] = {}
        for cid, val in dyn.items():
            enc = _encode_color_channel(val)
            if self._last_colors.get(cid, _MISSING) != enc:
                c[cid] = enc
                self._last_colors[cid] = enc
        if c:
            rec["c"] = c
        self._emit(rec)
        self.bars += 1

    def write_events(self, events: list[dict]) -> None:
        for ev in events:
            self._emit(ev)

    def write_drawings_snapshot(self) -> None:
        self._emit(drawings_snapshot())

    def write_end(self, bars: int) -> None:
        self._emit({"t": "end", "bars": bars})


_MISSING = object()
