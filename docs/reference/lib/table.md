<!--
---
weight: 436
title: "table"
description: "Data tables — create and populate on-chart tables"
icon: "table_chart"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["table", "library", "reference"]
---
-->

# table

On-chart data tables — create, populate, and style tables to display information directly on your chart. Tables are useful for showing strategy statistics, price levels, or any other data you want to visualize alongside your price action.

## Quick Example

```python
from pynecore.lib import script, table, position, color, close, bar_index

@script.indicator(title="Price Table", overlay=True)
def main():
    # Create a table at the top-left corner
    tbl: table.Table = table.new(
        position.top_left,
        columns=2,
        rows=3,
        bgcolor=color.gray,
        border_width=1
    )
    
    # Add header row
    table.cell(tbl, 0, 0, "Metric", text_color=color.white)
    table.cell(tbl, 1, 0, "Value", text_color=color.white)
    
    # Add data rows
    table.cell(tbl, 0, 1, "Close", text_color=color.white)
    table.cell(tbl, 1, 1, str(close), text_color=color.white)
    
    table.cell(tbl, 0, 2, "Bar", text_color=color.white)
    table.cell(tbl, 1, 2, str(bar_index), text_color=color.white)
```

## Table Creation and Deletion

### table.new()

Creates a new table on the chart.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `position` | `position.Position` | Where to place the table: `position.top_left`, `position.top_center`, `position.top_right`, `position.middle_left`, `position.middle_center`, `position.middle_right`, `position.bottom_left`, `position.bottom_center`, `position.bottom_right` | (required) |
| `columns` | `int` | Number of columns | (required) |
| `rows` | `int` | Number of rows | (required) |
| `bgcolor` | `color.Color` | Background color of the entire table | `None` |
| `frame_color` | `color.Color` | Color of the outer frame | `None` |
| `frame_width` | `int` | Width of the outer frame in pixels | `0` |
| `border_color` | `color.Color` | Color of cell borders (excluding outer frame) | `None` |
| `border_width` | `int` | Width of cell borders in pixels | `0` |
| `force_overlay` | `bool` | If `True`, display on main chart even if indicator is in a separate pane | `False` |

**Returns:** `table.Table` — a table object to pass to other `table.*()` functions

Example:
```python
tbl: table.Table = table.new(position.top_left, 3, 4)
```

### table.delete()

Deletes a table.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to delete |

**Returns:** `None`

## Table Styling

### table.set_position()

Changes the position of a table on the chart.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `position` | `position.Position` | New position |

**Returns:** `None`

### table.set_bgcolor()

Sets the background color of the entire table.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `bgcolor` | `color.Color` | Background color |

**Returns:** `None`

### table.set_frame_color()

Sets the color of the table's outer frame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `frame_color` | `color.Color` | Frame color |

**Returns:** `None`

### table.set_frame_width()

Sets the width of the table's outer frame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `frame_width` | `int` | Frame width in pixels |

**Returns:** `None`

### table.set_border_color()

Sets the color of cell borders (excluding the outer frame).

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `border_color` | `color.Color` | Border color |

**Returns:** `None`

### table.set_border_width()

Sets the width of cell borders (excluding the outer frame).

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `border_width` | `int` | Border width in pixels |

**Returns:** `None`

## Cell Operations

### table.cell()

Creates or updates a cell with text and formatting.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `table_id` | `table.Table` | The table containing the cell | (required) |
| `column` | `int` | Column index (0-based) | (required) |
| `row` | `int` | Row index (0-based) | (required) |
| `text` | `str` | Cell text content | `""` |
| `width` | `int \| float` | Cell width as percentage of visual space (0 = auto) | `0` |
| `height` | `int \| float` | Cell height as percentage of visual space (0 = auto) | `0` |
| `text_color` | `color.Color` | Text color | `color.black` |
| `text_halign` | `text.Align` | Horizontal alignment: `text.align_left`, `text.align_center`, `text.align_right` | `text.align_center` |
| `text_valign` | `text.Align` | Vertical alignment: `text.align_top`, `text.align_center`, `text.align_bottom` | `text.align_center` |
| `text_size` | `int \| str` | Text size: positive integer or `size.tiny`, `size.small`, `size.normal`, `size.large`, `size.huge` | `size.normal` |
| `bgcolor` | `color.Color` | Background color | `None` |
| `tooltip` | `str` | Tooltip text on hover | `""` |
| `text_font_family` | `font.FontFamily` | Font: `font.family_default`, `font.family_monospace` | `font.family_default` |
| `text_formatting` | `text.Format` | Formatting: `text.format_none`, `text.format_bold`, `text.format_italic`, `text.format_strikeout` | `text.format_none` |

**Returns:** `None`

Example:
```python
table.cell(tbl, 0, 0, "Price", text_color=color.white, bgcolor=color.blue)
```

### table.cell_set_text()

Updates the text in a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text` | `str` | New text content |

**Returns:** `None`

Example:
```python
table.cell_set_text(tbl, 1, 1, str(close))
```

### table.cell_set_bgcolor()

Sets the background color of a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `bgcolor` | `color.Color` | Background color |

**Returns:** `None`

### table.cell_set_text_color()

Sets the text color of a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_color` | `color.Color` | Text color |

**Returns:** `None`

### table.cell_set_text_size()

Sets the text size of a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_size` | `int \| str` | Text size (integer or size constant) |

**Returns:** `None`

### table.cell_set_text_halign()

Sets the horizontal alignment of a cell's text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_halign` | `text.Align` | Alignment: `text.align_left`, `text.align_center`, `text.align_right` |

**Returns:** `None`

### table.cell_set_text_valign()

Sets the vertical alignment of a cell's text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_valign` | `text.Align` | Alignment: `text.align_top`, `text.align_center`, `text.align_bottom` |

**Returns:** `None`

### table.cell_set_width()

Sets the width of a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `width` | `int \| float` | Width as percentage (0 = auto) |

**Returns:** `None`

### table.cell_set_height()

Sets the height of a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `height` | `int \| float` | Height as percentage (0 = auto) |

**Returns:** `None`

### table.cell_set_tooltip()

Sets a tooltip that appears when hovering over a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `tooltip` | `str` | Tooltip text |

**Returns:** `None`

### table.cell_set_text_font_family()

Sets the font family of a cell's text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_font_family` | `font.FontFamily` | Font: `font.family_default` or `font.family_monospace` |

**Returns:** `None`

### table.cell_set_text_formatting()

Applies text formatting (bold, italic, strikethrough) to a cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table containing the cell |
| `column` | `int` | Column index |
| `row` | `int` | Row index |
| `text_formatting` | `text.Format` | Format: `text.format_none`, `text.format_bold`, `text.format_italic`, `text.format_strikeout` |

**Returns:** `None`

## Table Utilities

### table.clear()

Removes cells from a rectangular range within a table.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `start_column` | `int` | Starting column index |
| `start_row` | `int` | Starting row index |
| `end_column` | `int` | Ending column index (defaults to start_column) |
| `end_row` | `int` | Ending row index (defaults to start_row) |

**Returns:** `None`

Example:
```python
table.clear(tbl, 0, 1, 1, 2)  # Clear cells from (0,1) to (1,2)
```

### table.merge_cells()

Merges a rectangular range of cells into a single cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `table_id` | `table.Table` | The table to modify |
| `start_column` | `int` | Starting column index |
| `start_row` | `int` | Starting row index |
| `end_column` | `int` | Ending column index |
| `end_row` | `int` | Ending row index |

**Returns:** `None`

Example:
```python
table.merge_cells(tbl, 0, 0, 1, 0)  # Merge top two cells
```

## Module Properties

### table.all

Returns an array of all tables currently drawn by the script.

**Type:** `list[table.Table]`

Example:
```python
all_tables: list[table.Table] = table.all
```

## Compatibility Notes

All functions in the `table` namespace are fully supported in PyneCore. The namespace provides complete table creation and manipulation capabilities matching TradingView's Pine Script v6 `table` API.