<!--
---
weight: 804
title: "Visual Output (Viz)"
description: "Plot styles and drawing data output — the NDJSON viz stream, CLI flags and programmatic accessors"
icon: "brush"
date: "2026-07-19"
lastmod: "2026-07-19"
draft: false
toc: true
categories: ["Programmatic", "API"]
tags: ["viz", "plot", "drawings", "ndjson", "plot-styles"]
---
-->

# Visual Output (Viz)

The plot CSV (`--plot` / `plot_path`) only stores the numeric value of every `plot()` call, one
column per plot. It carries no style information (colors, line widths, shapes) and nothing about the
drawing objects (lines, labels, boxes, tables, polylines, linefills) a script creates.

The **viz** output fills that gap. It is an opt-in NDJSON stream that captures, per bar:

- the **style metadata** of every plot-family call (`plot`, `plotshape`, `plotchar`, `plotarrow`,
  `plotcandle`, `plotbar`, `bgcolor`, `barcolor`, `hline`, `fill`),
- the **per-bar values** those calls produce,
- the **per-bar dynamic colors** (only when they change), and
- a full **snapshot** — or an optional per-bar **journal** — of the drawing objects.

> The plots CSV is unchanged and keeps working exactly as before. Viz output is entirely opt-in;
> enabling it never alters the CSV.

## Enabling It

### CLI

| Flag                | Effect                                                                      |
|---------------------|-----------------------------------------------------------------------------|
| `--viz`, `-vz`      | Write the viz NDJSON to `<output_dir>/<script>_viz.ndjson`                   |
| `--viz-path PATH`   | Write the viz NDJSON to `PATH` (implies `--viz`)                             |
| `--viz-journal`     | Also record per-bar drawing create/update/delete events (implies `--viz`)   |

```bash
pyne run my_script.py BINANCE_BTCUSD_60 --viz
pyne run my_script.py BINANCE_BTCUSD_60 --viz-path out/viz.ndjson --viz-journal
```

### Programmatically

`ScriptRunner` takes two viz parameters:

| Parameter      | Type           | Description                                                                    |
|----------------|----------------|--------------------------------------------------------------------------------|
| `viz_path`     | `Path \| None` | Write the viz NDJSON here. `None` disables file output.                         |
| `viz_journal`  | `bool`         | Diff the live drawings every bar and emit create/update/delete events.         |

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner

runner = ScriptRunner(
    script_path=Path("my_script.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    viz_path=Path("out/viz.ndjson"),
    viz_journal=True,
)

for candle, plot_data in runner.run_iter():
    ...  # the NDJSON is written as bars are processed
```

Journaling can also run **without** a file: set `viz_journal=True`, leave `viz_path=None`, and attach
a `viz_events` callback (see [Programmatic Accessors](#programmatic-accessors)) to receive the events
in memory.

## NDJSON Format

The stream is [NDJSON](https://ndjson.org/): one JSON object per line. Every record is tagged by a
`"t"` field. The records appear in this order:

1. `hdr` — always the first line.
2. Interleaved `meta` and `bar` records (and `ev` records when journaling). A plot's `meta` is
   emitted lazily, on the first bar the plot actually fires, immediately before that bar's `bar`
   record.
3. `drawings` — a full end-of-run snapshot, second to last.
4. `end` — always the last line.

### `hdr` — header

| Field      | Type   | Description                                                        |
|------------|--------|--------------------------------------------------------------------|
| `t`        | str    | `"hdr"`                                                            |
| `v`        | int    | Format version (`1`)                                              |
| `script`   | object | Script context: `type`, `title`, `shorttitle`, `overlay`, `format`, `precision`, `explicit_plot_zorder`, `max_lines_count`, `max_labels_count`, `max_boxes_count`, `max_polylines_count` (null fields dropped) |
| `syminfo`  | object | Symbol context: `tickerid`, `prefix`, `ticker`, `timeframe`, `mintick`, `timezone`, `type` (empty fields dropped) |
| `journal`  | bool   | Whether per-bar `ev` records are present                          |

### `meta` — plot-family definition

The registered-once style of one plot-family call. Common fields are shared; kind-specific fields
follow. `display` defaults to `"all"`; other defaults are applied per kind and null fields are
dropped, so a record only lists what deviates from the default.

A `meta` record is emitted before the first `bar` record of the plot. When a plot only turns out
to be dynamic on a later bar (its first differing color arrives after the record went out), the
record is re-emitted with `dynamic: true` before that bar's color delta — a consumer must treat a
repeated `id` as an update of the earlier record.

| Field     | Type | Applies to | Notes                                                            |
|-----------|------|------------|------------------------------------------------------------------|
| `t`       | str  | all        | `"meta"`                                                        |
| `id`      | str  | all        | The plot id (see [Ordinal Ids](#ordinal-id-semantics))          |
| `kind`    | str  | all        | `plot`/`shape`/`char`/`arrow`/`candle`/`bar`/`bgcolor`/`barcolor`/`hline`/`fill` |
| `title`   | str  | all        | Present when the call had a title                               |
| `display` | str  | all        | `all`/`none`/`pane`/`data_window`/`price_scale`/`status_line`   |
| `dynamic` | bool | all        | `true` when the call has a per-bar color channel                |
| `style`   | str  | plot       | `line`/`stepline`/`histogram`/`columns`/`area`/… (default `line`) |
| `color`   | str  | most       | `#RRGGBBAA` hex; the static color                               |
| `linewidth` | int | plot, hline | Line width in pixels                                          |
| `trackprice`, `histbase`, `join` | — | plot | Present only when non-default                     |
| `style`, `location`, `size` | str | shape | Defaults `xcross` / `abovebar` / `auto`               |
| `char`    | str  | char       | The character (default `◆`)                                    |
| `location`, `size` | str | char | Defaults `abovebar` / `auto`                            |
| `text`, `textcolor` | str | shape, char | Marker text and its color                             |
| `colorup`, `colordown` | str | arrow | Up / down arrow colors                                |
| `minheight`, `maxheight` | int | arrow | Present when set                                    |
| `color`, `wickcolor`, `bordercolor` | str | candle | Candle body / wick / border colors        |
| `color`   | str  | bar        | Bar color                                                      |
| `price`, `linestyle` | — | hline | Level and `solid`/`dashed`/`dotted` (default `solid`)  |
| `plot1`, `plot2` | str | fill | Ids of the two filled plots                             |
| `hline1`, `hline2` | str | fill | Ids of the two filled hlines                          |
| `fillgaps` | bool | fill      | Present when `true`                                            |

### `bar` — per-bar values

| Field  | Type   | Description                                                                     |
|--------|--------|---------------------------------------------------------------------------------|
| `t`    | str    | `"bar"`                                                                        |
| `i`    | int    | Bar index (0-based)                                                            |
| `time` | int    | Bar open time in **milliseconds**                                             |
| `v`    | object | `{plot_id: value}` for every plot that fired this bar. Omitted when empty.     |
| `c`    | object | `{plot_id: color}` — only for color channels that **changed** this bar. Omitted when empty. |

`plotcandle` / `plotbar` expand to four keys per call: `"<title> (open)"`, `"<title> (high)"`,
`"<title> (low)"`, `"<title> (close)"`. `plotshape` stores `0`/`1` (or `null` when its series is
na); `plotchar` and `plotarrow` store the raw series value. Non-finite floats and na are encoded as
`null`.

### `ev` — drawing event (journal mode only)

| Field | Type   | Description                                                        |
|-------|--------|--------------------------------------------------------------------|
| `t`   | str    | `"ev"`                                                            |
| `i`   | int    | Bar index the change was detected on                              |
| `op`  | str    | `"create"` / `"update"` / `"delete"`                             |
| `obj` | str    | `line`/`label`/`box`/`table`/`polyline`/`linefill`               |
| `id`  | int    | The drawing's `vid` (see [Drawing vids](#drawing-vids))          |
| `s`   | object | The full serialized drawing state (omitted for `delete`)         |

### `drawings` — end-of-run snapshot

| Field       | Type  | Description                              |
|-------------|-------|------------------------------------------|
| `t`         | str   | `"drawings"`                            |
| `lines`     | array | Serialized `line` objects still live    |
| `labels`    | array | Serialized `label` objects              |
| `boxes`     | array | Serialized `box` objects                |
| `tables`    | array | Serialized `table` objects (with cells; a merged cell carries `merge: [start_col, start_row, end_col, end_row]` — the range's top-left cell renders, the rest are hidden) |
| `polylines` | array | Serialized `polyline` objects           |
| `linefills` | array | Serialized `linefill` objects (each embeds `line1_state` / `line2_state`) |

### `end` — terminator

| Field  | Type | Description                      |
|--------|------|----------------------------------|
| `t`    | str  | `"end"`                         |
| `bars` | int  | Total number of bars written     |

### Full Example

A three-bar indicator with a dynamic-colored `plot`, a static `plot`, a `fill`, a `plotshape` and an
`hline`:

```json
{"t":"hdr","v":1,"script":{"type":"indicator","title":"Demo","shorttitle":"demo","overlay":true,"format":"inherit","explicit_plot_zorder":false,"max_lines_count":50,"max_labels_count":50,"max_boxes_count":50,"max_polylines_count":50},"syminfo":{"tickerid":"BINANCE:BTCUSD","prefix":"BINANCE","ticker":"BINANCE:BTCUSD","timeframe":"60","mintick":0.01,"timezone":"UTC","type":"crypto"},"journal":false}
{"t":"meta","id":"close","kind":"plot","title":"close","display":"all","style":"line","color":"#F23645FF","linewidth":2}
{"t":"meta","id":"open","kind":"plot","title":"open","display":"all","style":"line","color":"#2962FFFF","linewidth":1}
{"t":"meta","id":"fill#0","kind":"fill","title":"band","display":"all","color":"#787B8619","plot1":"close","plot2":"open"}
{"t":"meta","id":"up","kind":"shape","title":"up","display":"all","style":"triangleup","location":"abovebar","size":"auto","color":"#4CAF50FF"}
{"t":"meta","id":"hline#0","kind":"hline","title":"level","display":"all","price":100.0,"color":"#787B86FF","linewidth":1,"linestyle":"solid"}
{"t":"bar","i":0,"time":1704067200000,"v":{"close":100.5,"open":100.0,"up":1}}
{"t":"bar","i":1,"time":1704070800000,"v":{"close":100.2,"open":100.5,"up":0},"c":{"close":"#00E676FF"}}
{"t":"bar","i":2,"time":1704074400000,"v":{"close":101.5,"open":100.2,"up":1}}
{"t":"drawings","lines":[],"labels":[],"boxes":[],"tables":[],"polylines":[],"linefills":[]}
{"t":"end","bars":3}
```

## Only-On-Change Color Encoding

A plot's color is registered once, on the bar the plot first fires, as the `meta.color` static
color. On every later bar the runtime compares the call's color to that registered color by object
**identity** — a `color.*` constant or the exact same object is treated as static and never repeated.

Only a color that differs (for example the branches of a `bar_index % 2 == 0 ? color.red :
color.lime` expression, or a fresh `color.new(...)` each bar) is a *dynamic channel*. Once a plot has
become dynamic it records its color on **every** later bar, and each color is written to the `bar`
record's `c` object **only when the encoded value actually changes from the last one emitted**. In
the example above, `close` alternates red/lime: bar 0 registers red as static (no `c`), bar 1
switches to lime (emitted), bar 2 switches back to red (emitted again, because the last emitted value
was lime), and so on. A reader reconstructs the color for any bar by carrying the last emitted value
forward, falling back to `meta.color` before the first emitted change.

`bgcolor` and `barcolor` likewise record every bar; an `na` (unpainted) bar is encoded as `null`, so
a paint → unpaint → paint sequence round-trips exactly.

## Ordinal-Id Semantics

`plot`, `plotshape`, `plotchar`, `plotarrow`, `plotcandle` and `plotbar` are identified by their
**title** (`plot(close, "RSI")` → id `"RSI"`; duplicate titles get a numeric suffix). `plotcandle` /
`plotbar` additionally expose their four OHLC value keys as `"<title> (open)"` etc.

`bgcolor`, `barcolor`, `fill` and `hline` have no natural title key, so they are identified by a
per-family **ordinal** assigned in call order: `bgcolor#0`, `bgcolor#1`, `fill#0`, `fill#1`,
`hline#0`, … The counters reset every bar, so as long as the script issues these calls in a stable
order (the normal case) the ids are stable across bars. A `fill` records the ids of the plots or
hlines it spans (`plot1`/`plot2` or `hline1`/`hline2`).

## Drawing vids

Every drawing object (`line`, `label`, `box`, `table`, `polyline`, `linefill`) is assigned a
monotonically increasing integer `vid` at creation. `line.copy()` / `label.copy()` / `box.copy()`
produce a clone with a **fresh** vid. Both the `drawings` snapshot and the journal `ev` records key
objects by vid, and a `linefill` embeds the current state of its two lines under `line1_state` /
`line2_state`.

## Journal Mode

With `viz_journal=True`, the runtime diffs the live drawing registries against the previous bar and
emits an `ev` record for every object that was created, changed, or deleted. A drawing that is
created once and mutated each bar (e.g. a trend line whose endpoint is moved with `line.set_xy2`)
produces exactly one `create`, one `update` per changed bar, and one `delete`. Without journaling the
stream carries only the final `drawings` snapshot.

## Programmatic Accessors

After (or during) a run, the runner exposes the same state the writer serializes:

| Accessor              | Kind         | Description                                                              |
|-----------------------|--------------|--------------------------------------------------------------------------|
| `runner.plot_meta`    | property     | `{id: PlotMeta}` — the registered plot-family metadata                   |
| `runner.drawings()`   | method       | A full drawings snapshot (same shape as the `drawings` NDJSON record)    |
| `runner.viz_events`   | attribute    | Optional `Callable[[list[dict]], None]`; receives each bar's journal events |

Plot-family and drawing state is reset only at **run start**, so `plot_meta` / `drawings()` remain
valid after the iterator is exhausted:

```python
runner = ScriptRunner(script_path=Path("my_script.py"), ohlcv_iter=candles, syminfo=syminfo)
list(runner.run_iter())  # exhaust the iterator

for pid, meta in runner.plot_meta.items():
    print(pid, meta.kind, meta.color)

snap = runner.drawings()
print(len(snap["lines"]), "lines survived")
```

To capture per-bar journal events in memory, enable journaling and attach a callback before
iterating:

```python
runner = ScriptRunner(script_path=Path("my_script.py"), ohlcv_iter=candles,
                      syminfo=syminfo, viz_journal=True)
batches: list[list[dict]] = []
runner.viz_events = batches.append
list(runner.run_iter())
```

### Reading the Live Per-Bar State

For the freshest per-bar view, read the library-level state **inside** the `run_iter()` loop —
`run_iter()` yields before it clears the per-bar viz state:

```python
from pynecore import lib

for candle, plot_data in runner.run_iter():
    # lib._plot_meta: {id: PlotMeta} registered so far
    # lib._viz_dyn:   {id: color} dynamic color channels active on THIS bar
    dyn = dict(lib._viz_dyn)  # copy — it is cleared after the yield
```

`lib._viz_dyn` holds the current-bar color of every channel that has become dynamic (a purely static
plot, whose color always equals its registered `meta.color`, is absent). The NDJSON writer applies
its [only-on-change](#only-on-change-color-encoding) filter on top of this; an in-process reader sees
the full per-bar value directly.
