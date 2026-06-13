from ..core.module_property import module_property
from ..types.table import Table
from ..types.na import NA
from ..lib import (color as _color, position as _position, size as _size, text as _text, font as _font)

_registry: list[Table] = []


def new(position: _position.Position, columns: int, rows: int, bgcolor: _color.Color = None,
        frame_color: _color.Color = None, frame_width: int = 0, border_color: _color.Color = None,
        border_width: int = 0, force_overlay: bool = False) -> Table:
    """
    Creates a new table object.

    :param position: Position of the table. Possible values are: position.top_left, position.top_center,
                     position.top_right, position.middle_left, position.middle_center, position.middle_right,
                     position.bottom_left, position.bottom_center, position.bottom_right
    :param columns: The number of columns in the table
    :param rows: The number of rows in the table
    :param bgcolor: The background color of the table. Optional. The default is no color
    :param frame_color: The color of the outer frame of the table. Optional. The default is no color
    :param frame_width: The width of the outer frame of the table. Optional. The default is 0
    :param border_color: The color of the borders of the cells (excluding the outer frame). Optional.
                         The default is no color
    :param border_width: The width of the borders of the cells (excluding the outer frame). Optional. The default is 0
    :param force_overlay: If true, the drawing will display on the main chart pane, even when the
                          script occupies a separate pane. Optional. The default is false
    :return: A table object
    """
    table = Table(
        position=position,
        columns=columns,
        rows=rows,
        bgcolor=bgcolor,
        frame_color=frame_color,
        frame_width=frame_width,
        border_color=border_color,
        border_width=border_width,
        force_overlay=force_overlay
    )
    _registry.append(table)
    return table


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Table]:
    """
    Returns an array filled with all the current tables drawn by the script.

    :return: Array of all table objects
    """
    return _registry


# noinspection PyShadowingBuiltins
def delete(table_id: Table) -> None:
    """
    Deletes a table object.

    :param table_id: A table object
    """
    if isinstance(table_id, NA):
        return
    try:
        _registry.remove(table_id)
    except ValueError:
        pass


def cell(table_id: Table, column: int, row: int, text: str = "", width: int | float = 0,
         height: int | float = 0, text_color: _color.Color = _color.black,
         text_halign: _text.AlignEnum = _text.align_center, text_valign: _text.AlignEnum = _text.align_center,
         text_size: int | str = _size.normal, bgcolor: _color.Color = None, tooltip: str = "",
         text_font_family: _font.FontFamilyEnum = _font.family_default,
         text_formatting: _text.FormatEnum = _text.format_none) -> None:
    """
    Defines a cell in the table and sets its attributes.

    :param table_id: A table object
    :param column: The index of the cell's column. Numbering starts at 0
    :param row: The index of the cell's row. Numbering starts at 0
    :param text: The text to be displayed inside the cell. Optional. The default is empty string
    :param width: The width of the cell as a % of the indicator's visual space. Optional. By default,
                  auto-adjusts the width based on the text inside the cell. Value 0 has the same effect
    :param height: The height of the cell as a % of the indicator's visual space. Optional. By default,
                   auto-adjusts the height based on the text inside of the cell. Value 0 has the same effect
    :param text_color: The color of the text. Optional. The default is color.black
    :param text_halign: The horizontal alignment of the cell's text. Optional. The default value is text.align_center
    :param text_valign: The vertical alignment of the cell's text. Optional. The default value is text.align_center
    :param text_size: Size of the object. The size can be any positive integer, or one of the size.*
                      built-in constant strings. The default value is size.normal or 14
    :param bgcolor: The background color of the text. Optional. The default is no color
    :param tooltip: The tooltip to be displayed inside the cell. Optional
    :param text_font_family: The font family of the text. Optional. The default value is font.family_default
    :param text_formatting: The formatting of the displayed text. Optional. The default is text.format_none
    """
    if isinstance(table_id, NA):
        return

    # Create or get the cell
    cell_obj = table_id.get_cell(column, row)

    # Set all the cell properties
    cell_obj.text = text
    cell_obj.width = width
    cell_obj.height = height
    cell_obj.text_color = text_color
    cell_obj.text_halign = text_halign
    cell_obj.text_valign = text_valign
    cell_obj.text_size = text_size
    cell_obj.bgcolor = bgcolor
    cell_obj.tooltip = tooltip
    cell_obj.text_font_family = text_font_family
    cell_obj.text_formatting = text_formatting


def clear(table_id: Table, start_column: int, start_row: int, end_column: int = None,
          end_row: int = None) -> None:
    """
    Removes a cell or a sequence of cells from the table.

    :param table_id: A table object
    :param start_column: The index of the column of the first cell to delete. Numbering starts at 0
    :param start_row: The index of the row of the first cell to delete. Numbering starts at 0
    :param end_column: The index of the column of the last cell to delete. Optional. The default is
                       the argument used for start_column
    :param end_row: The index of the row of the last cell to delete. Optional. The default is the
                    argument used for start_row
    """
    if isinstance(table_id, NA):
        return
    if end_column is None:
        end_column = start_column
    if end_row is None:
        end_row = start_row

    # Clear the specified range of cells
    table_id.clear_cells(start_column, start_row, end_column, end_row)


def merge_cells(table_id: Table, start_column: int, start_row: int, end_column: int, end_row: int) -> None:
    """
    Merges a sequence of cells in the table into one cell.

    :param table_id: A table object
    :param start_column: The index of the column of the first cell to merge. Numbering starts at 0
    :param start_row: The index of the row of the first cell to merge. Numbering starts at 0
    :param end_column: The index of the column of the last cell to merge. Numbering starts at 0
    :param end_row: The index of the row of the last cell to merge. Numbering starts at 0
    """
    if isinstance(table_id, NA):
        return

    # Merge the specified range of cells
    table_id.merge_cells(start_column, start_row, end_column, end_row)


# Cell setter functions
def cell_set_bgcolor(table_id: Table, column: int, row: int, bgcolor: _color.Color) -> None:
    """Sets the background color of the cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.bgcolor = bgcolor


def cell_set_height(table_id: Table, column: int, row: int, height: int | float) -> None:
    """Sets the height of cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.height = height


def cell_set_text(table_id: Table, column: int, row: int, text: str) -> None:
    """Sets the text in the specified cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text = text


def cell_set_text_color(table_id: Table, column: int, row: int, text_color: _color.Color) -> None:
    """Sets the color of the text inside the cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_color = text_color


def cell_set_text_font_family(table_id: Table, column: int, row: int, text_font_family: _font.FontFamilyEnum) -> None:
    """Sets the font family of the text inside the cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_font_family = text_font_family


def cell_set_text_formatting(table_id: Table, column: int, row: int, text_formatting: _text.FormatEnum) -> None:
    """Sets the formatting attributes the drawing applies to displayed text."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_formatting = text_formatting


def cell_set_text_halign(table_id: Table, column: int, row: int, text_halign: _text.AlignEnum) -> None:
    """Sets the horizontal alignment of the cell's text."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_halign = text_halign


def cell_set_text_size(table_id: Table, column: int, row: int, text_size: int | str) -> None:
    """Sets the size of the cell's text."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_size = text_size


def cell_set_text_valign(table_id: Table, column: int, row: int, text_valign: _text.AlignEnum) -> None:
    """Sets the vertical alignment of a cell's text."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.text_valign = text_valign


def cell_set_tooltip(table_id: Table, column: int, row: int, tooltip: str) -> None:
    """Sets the tooltip in the specified cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.tooltip = tooltip


def cell_set_width(table_id: Table, column: int, row: int, width: int | float) -> None:
    """Sets the width of the cell."""
    if isinstance(table_id, NA):
        return
    cell_obj = table_id.get_cell(column, row)
    cell_obj.width = width


# Table setter functions
def set_bgcolor(table_id: Table, bgcolor: _color.Color) -> None:
    """Sets the background color of a table."""
    if isinstance(table_id, NA):
        return
    table_id.bgcolor = bgcolor


def set_border_color(table_id: Table, border_color: _color.Color) -> None:
    """Sets the color of the borders (excluding the outer frame) of the table's cells."""
    if isinstance(table_id, NA):
        return
    table_id.border_color = border_color


def set_border_width(table_id: Table, border_width: int) -> None:
    """Sets the width of the borders (excluding the outer frame) of the table's cells."""
    if isinstance(table_id, NA):
        return
    table_id.border_width = border_width


def set_frame_color(table_id: Table, frame_color: _color.Color) -> None:
    """Sets the color of the outer frame of a table."""
    if isinstance(table_id, NA):
        return
    table_id.frame_color = frame_color


def set_frame_width(table_id: Table, frame_width: int) -> None:
    """Sets the width of the outer frame of a table."""
    if isinstance(table_id, NA):
        return
    table_id.frame_width = frame_width


def set_position(table_id: Table, position: _position.Position) -> None:
    """Sets the position of a table."""
    if isinstance(table_id, NA):
        return
    table_id.position = position
