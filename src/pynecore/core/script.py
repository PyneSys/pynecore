from typing import cast, Any, Callable, TypeVar
import os
import sys

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pynecore.lib.format as _format
import pynecore.lib.scale as _scale
import pynecore.lib.strategy as _strategy
import pynecore.lib.strategy.commission
import pynecore.lib.currency as _currency
import pynecore.lib.display as _display

from pynecore.types import script_type as _script_type
from pynecore.types.series import Series
from pynecore.types.color import Color
from pynecore.types.na import NA

__all__ = ['script', 'input']

from pynecore.types.source import Source
from . import safe_convert

# Global registry for library main functions
_registered_libraries: list[tuple[str, Callable]] = []

# TypeVar for enum type preservation
TEnum = TypeVar('TEnum', bound=StrEnum)


@dataclass(kw_only=True)
class InputData:
    """
    Input dataclass
    """
    id: str | None = None
    input_type: str | None = None
    defval: int | bool | Color | float | str | None = None
    title: str | None = None
    minval: int | float | None = None
    maxval: int | float | None = None
    step: int | float | None = None
    tooltip: str | None = None
    inline: bool | None = False
    group: str | None = None
    confirm: bool | None = False
    options: tuple[int | float | str, ...] | None = None
    display: _display.Display | None = None


old_input_values: dict[str, Any] = {}
inputs: dict[str | None, InputData] = {}


# noinspection PyShadowingBuiltins,PyShadowingNames
@dataclass(kw_only=True)
class Script:
    """
    Script parameters dataclass
    """
    # These fields will be skipped when saving to toml
    _SKIP_FIELDS = {'script_type', 'inputs', 'title', 'shorttitle', 'position'}

    script_type: _script_type.ScriptType | None = None
    inputs: dict[str, InputData] = field(default_factory=dict)

    title: str | None = None
    shorttitle: str | None = None

    overlay: bool = False
    format: _format.Format = _format.inherit
    precision: int | None = None
    scale: _scale.Scale | None = None
    pyramiding: int = 0
    calc_on_order_fills: bool = False
    calc_on_every_tick: bool = False
    max_bars_back: int = 0
    timeframe: str | None = None
    timeframe_gaps: bool = True
    explicit_plot_zorder: bool = False
    max_lines_count: int = 50
    max_labels_count: int = 50
    max_boxes_count: int = 50
    calc_bars_count: int = 0
    max_polylines_count: int = 50
    dynamic_requests: bool = False
    behind_chart: bool = True

    backtest_fill_limits_assumption: int = 0
    default_qty_type: _strategy.QtyType = _strategy.cash
    default_qty_value: float = 1
    initial_capital: float | int = 1000000
    currency: _currency.Currency = _currency.NONE
    slippage: int = 0
    commission_type: _strategy.commission.Commission = _strategy.commission.percent  # type: ignore
    commission_value: int | float = 0.0
    process_orders_on_close: bool = False
    close_entries_rule: str = 'FIFO'
    margin_long: int | float = 100.0  # Defaulted to 100.0 in Pine Script v6
    margin_short: int | float = 100.0  # Defaulted to 100.0 in Pine Script v6
    risk_free_rate: float = 2.0
    use_bar_magnifier: bool = False
    fill_orders_on_standard_ohlc: bool = False

    position: _strategy.Position | None = None

    _modified: set[str] = field(default_factory=set)

    def save(self, path: Path):
        """
        Save script settings to TOML-like format.
        Non-settable fields are commented out with '#'.
        None values are also commented out.
        Input values are saved with their metadata as comments.

        :param path: Path to save the file
        """

        def _format_value(value) -> str:
            """Format value according to its type"""
            if isinstance(value, bool):
                return str(value).lower()
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, Color):
                value = str(value)
            if isinstance(value, str):
                return f'"{value}"'
            return str(value)

        lines = [
            "# Indicator / Strategy / Library Settings",
            "",
            "[script]"
        ]

        # Save general settings
        for key, value in self.__dict__.items():
            if key.startswith('_') or key in self._SKIP_FIELDS:
                continue
            if value is None:
                line = f"#{key} ="
            else:
                line = "#" if key not in self._modified else ""
                line += f"{key} = {_format_value(value)}"
            lines.append(line)

        # Add an empty line before inputs
        lines.append("")
        lines.append("# Input Settings")

        # Save inputs
        for arg_name, input_data in self.inputs.items():
            lines.append(f"\n[inputs.{arg_name.removesuffix('__global__')}]")
            lines.append("# Input metadata, cannot be modified")

            # Add all metadata as comments
            for key, value in input_data.__dict__.items():
                if key == 'id':
                    continue
                if value is not None:
                    # We use ':` to not confuse with real values
                    lines.append(f"# {key.rjust(10)}: {_format_value(value)}")

            lines.append("# Change here to modify the input value")

            # Add the actual value
            if input_data.defval is not None and arg_name in old_input_values:
                lines.append(f"value = {_format_value(old_input_values[arg_name])}")
            else:
                lines.append("#value =")

        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def load(self, path: str | Path) -> None:
        """
        Load script settings from TOML file and update this script instance.
        Only loads settable fields and input values, preserving the original script structure.

        :param path: Path to load the file from
        """
        import tomllib

        with open(path, 'rb') as f:
            data = tomllib.load(f)

        if 'script' not in data:
            raise ValueError("Invalid TOML: missing [script] section!")

        script_data = data['script']

        for key, value in script_data.items():
            if key not in self._SKIP_FIELDS and hasattr(self, key):
                if value != getattr(self, key):
                    setattr(self, key, value)
                    # We just save the modified fields
                    self._modified.add(key)

        if 'inputs' not in data:
            return

        # Fill old_input_values
        for arg_name, arg_data in data['inputs'].items():
            if 'value' not in arg_data:
                continue
            old_input_values[arg_name] = arg_data['value']
            old_input_values[arg_name + '__global__'] = arg_data['value']  # For strict mode

    #
    # decorators
    #

    def _decorate(self):
        # Get the script path from the caller frame
        script_path = Path(sys._getframe(2).f_globals['__file__']).resolve()  # noqa F821
        toml_path = script_path.with_suffix('.toml')

        # Load settings from toml file if exists
        if toml_path.exists():
            self.load(toml_path)

        def decorator(func):
            # Save inputs to script instance then clear inputs (for next script)
            self.inputs = inputs.copy()  # type: ignore
            inputs.clear()

            # Set script attribute to the main function to be able to access script properties
            setattr(func, 'script', self)

            if self.script_type in (_script_type.indicator, _script_type.strategy):
                # Save toml file if not in pytest and not disabled by env var PYNE_SAVE_SCRIPT_TOML = 0
                if os.environ.get('PYNE_SAVE_SCRIPT_TOML', '1') == '1' and 'pytest' not in sys.modules:
                    self.save(toml_path)

            old_input_values.clear()
            return func

        return decorator

    @classmethod
    def indicator(
            cls,
            title='', shorttitle='',
            overlay=False,
            format: _format.Format = _format.inherit,
            precision: int | None = None,
            scale: _scale.Scale | None = None,
            max_bars_back=0,
            timeframe: str | None = None,
            timeframe_gaps=True,
            explicit_plot_zorder=False,
            max_lines_count=50,
            max_labels_count=50,
            max_boxes_count=50,
            calc_bars_count=0,
            max_polylines_count=50,
            dynamic_requests=False,
            behind_chart=True,
            *_, **__
    ) -> Callable[..., Any]:
        """
        Decorator for indicator script. You should deocrate `main` function with this decorator if
        your script is an indicator script.

        :param title: The title of the script
        :param shorttitle: The script's display name
        :param overlay: If True, the script will be displayed on the price chart as an overlay,
                        otherwise it will be displayed in a separate pane
        :param format: Specifies the formatting of the script's displayed values
        :param precision: Specifies the number of digits after the floating point of the script's displayed values
        :param scale: The price scale used
        :param max_bars_back: The length of the historical buffer the script keeps for every
                              Series variables which determines how many past values can be
                              referenced by Series objects
        :param timeframe: Adds multi-timeframe functionality to simple scripts
        :param timeframe_gaps: Specifies how the indicator's values are displayed on chart bars
                               when the `timeframe` is higher than the chart's.
        :param explicit_plot_zorder: Specifies the order in which the script's plots, fills, and hlines are rendered
        :param max_lines_count: The number of last line drawings displayed on the chart
        :param max_labels_count: The number of last label drawings displayed
        :param max_boxes_count: The number of last box drawings displayed
        :param calc_bars_count: Limits the initial calculation of a script to the last number of bars specified
        :param max_polylines_count: The number of last polyline drawings displayed
        :param dynamic_requests: Specifies whether the script can dynamically call functions from
                                 the `request.*()` namespace
        :param behind_chart: Controls whether the script's plots and drawings in the main chart pane
                             appear behind the chart display
        """
        script = cls()
        script.script_type = _script_type.indicator
        script.title = title
        script.shorttitle = shorttitle

        script.overlay = overlay
        script.format = format
        script.precision = precision
        script.scale = scale
        script.max_bars_back = max_bars_back
        script.timeframe = timeframe
        script.timeframe_gaps = timeframe_gaps
        script.explicit_plot_zorder = explicit_plot_zorder
        script.max_lines_count = max_lines_count
        script.max_labels_count = max_labels_count
        script.max_boxes_count = max_boxes_count
        script.calc_bars_count = calc_bars_count
        script.max_polylines_count = max_polylines_count
        script.dynamic_requests = dynamic_requests
        script.behind_chart = behind_chart

        return script._decorate()

    @classmethod
    def strategy(
            cls,
            title='', shorttitle='',
            overlay=False,
            format: _format.Format = _format.inherit,
            precision: int | None = None,
            scale: _scale.Scale | None = None,

            pyramiding: int = 0,
            calc_on_order_fills=False,
            calc_on_every_tick=False,

            max_bars_back=0,

            backtest_fill_limits_assumption=0,
            default_qty_type: _strategy.QtyType = _strategy.fixed,
            default_qty_value: float = 1,
            initial_capital: float | int = 1000000,
            currency: _currency.Currency = _currency.NONE,
            slippage: int = 0,
            commission_type: _strategy.commission.Commission = _strategy.commission.percent,  # type: ignore
            commission_value: int | float = 0.0,
            process_orders_on_close=False,
            close_entries_rule='FIFO',
            margin_long: int | float = 0.0,
            margin_short: int | float = 0.0,

            explicit_plot_zorder=False,
            max_lines_count=50,
            max_labels_count=50,
            max_boxes_count=50,
            calc_bars_count=0,

            risk_free_rate=2.0,
            use_bar_magnifier=False,
            fill_orders_on_standard_ohlc=False,

            max_polylines_count=50,
            dynamic_requests=False,

            behind_chart=True,

            *_, **__
    ) -> Callable[..., Any]:
        """
        Decorator for strategy script. You should deocrate `main` function with this decorator if
        your script is a strategy script.

        :param title: The title of the script
        :param shorttitle: The script's display name
        :param overlay: If True, the script will be displayed on the price chart as an overlay,
        :param format: Specifies the formatting of the script's displayed values
        :param precision: Specifies the number of digits after the floating point of the script's displayed values
        :param scale: The price scale used
        :param pyramiding: The maximum number of entries allowed in the same direction
        :param calc_on_order_fills: Specifies whether the strategy should be recalculated after an order is filled
        :param calc_on_every_tick: Specifies whether the strategy should be recalculated on each realtime tick
        :param max_bars_back: The length of the historical buffer the script keeps for every
        :param backtest_fill_limits_assumption: Limit order execution threshold in ticks
        :param default_qty_type: Specifies the units used for `default_qty_value`
        :param default_qty_value: The default quantity to trade, in units determined by the argument
                                 used with the `default_qty_type` parameter
        :param initial_capital: The amount of funds initially available for the strategy to trade,
                                in units of `currency`
        :param currency: Currency used by the strategy in currency-related calculations
        :param slippage: Slippage expressed in ticks
        :param commission_type: Determines what the number passed to the `commission_value`
        :param commission_value: Commission applied to the strategy's orders in units determined by
                                 the argument passed to the `commission_type` parameter
        :param process_orders_on_close: When set to true, generates an additional attempt to execute
                                        orders after a bar closes and strategy calculations are completed.
        :param close_entries_rule: Determines the order in which trades are closed
        :param margin_long: Margin long is the percentage of the purchase price of a security that
                            must be covered by cash or collateral for long positions
        :param margin_short: Margin short is the percentage of the purchase price of a security that
                             must be covered by cash or collateral for short positions
        :param explicit_plot_zorder: Specifies the order in which the script's plots, fills, and hlines are rendered
        :param max_lines_count: The number of last line drawings displayed on the chart
        :param max_labels_count: The number of last label drawings displayed
        :param max_boxes_count: The number of last box drawings displayed
        :param calc_bars_count: Limits the initial calculation of a script to the last number of bars specified
        :param risk_free_rate: The risk-free rate of return is the annual percentage change in the
                               value of an investment with minimal or zero risk
        :param use_bar_magnifier: When true, the Broker Emulator uses lower timeframe data during
                                  history backtesting to achieve more realistic results
        :param fill_orders_on_standard_ohlc: When true, forces strategies running on Heikin Ashi
                                             charts to fill orders using actual OHLC prices, for more
                                             realistic results.
        :param max_polylines_count: The number of last polyline drawings displayed
        :param dynamic_requests: Specifies whether the script can dynamically call functions from
        :param behind_chart: Controls whether the script's plots and drawings in the main chart pane
        """
        script = cls()
        script.script_type = _script_type.strategy
        script.title = title
        script.shorttitle = shorttitle

        script.overlay = overlay
        script.format = format
        script.precision = precision
        script.scale = scale

        script.pyramiding = pyramiding
        script.calc_on_order_fills = calc_on_order_fills
        script.calc_on_every_tick = calc_on_every_tick

        script.max_bars_back = max_bars_back

        script.backtest_fill_limits_assumption = backtest_fill_limits_assumption
        script.default_qty_type = default_qty_type
        script.default_qty_value = default_qty_value
        script.initial_capital = initial_capital
        script.currency = currency
        script.slippage = slippage
        script.commission_type = commission_type
        script.commission_value = commission_value
        script.process_orders_on_close = process_orders_on_close
        script.close_entries_rule = close_entries_rule
        script.margin_long = margin_long
        script.margin_short = margin_short
        script.explicit_plot_zorder = explicit_plot_zorder

        script.max_lines_count = max_lines_count
        script.max_labels_count = max_labels_count
        script.max_boxes_count = max_boxes_count
        script.calc_bars_count = calc_bars_count

        script.risk_free_rate = risk_free_rate
        script.use_bar_magnifier = use_bar_magnifier
        script.fill_orders_on_standard_ohlc = fill_orders_on_standard_ohlc

        script.max_polylines_count = max_polylines_count
        script.dynamic_requests = dynamic_requests
        script.behind_chart = behind_chart

        script.position = _strategy.Position()

        return script._decorate()

    @classmethod
    def library(
            cls,
            title='',
            overlay=False,
            dynamic_requests=False,
            *_, **__
    ) -> Callable[..., Any]:
        """
        Decorator for library script. You should deocrate `main` function with this decorator if
        your script is a library script.

        :param title: The title of the script
        :param overlay: If True, the script will be displayed on the price chart as an overlay,
        :param dynamic_requests: Specifies whether the script can dynamically call functions from
        """
        script = cls()
        script.script_type = _script_type.library
        script.title = title
        script.shorttitle = title

        script.overlay = overlay
        script.dynamic_requests = dynamic_requests

        def decorator(func):
            # Register library main function if not already registered
            lib_entry = (script.title or 'Untitled Library', func)
            if lib_entry not in _registered_libraries:
                _registered_libraries.append(lib_entry)
            return script._decorate()(func)

        return decorator


script = Script


class _Input:
    """
    Input functions
    """

    def __call__(self, defval: Any, title: str | None = None, *,
                 tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
                 display: _display.Display | None = None, _id: str | None = None, **__) -> Any:
        """
        Adds an input to your script's settings, which allows you to provide configuration options

        :param defval: The default value of the input
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        input_type = type(defval).__name__.lower()
        if input_type == 'source':
            defval = str(defval)
        inputs[_id] = InputData(
            id=_id,
            input_type=input_type,
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            display=display,
        )
        return defval if _id not in old_input_values else old_input_values[_id]

    @classmethod
    def _int(cls, defval: int, title: str | None = None, *,
             minval: int | None = None, maxval: int | None = None, step: int | None = None,
             tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
             confirm: bool | None = False, options: tuple[int] | None = None,
             display: _display.Display | None = None, _id: str | None = None, **__) -> int | NA[int]:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for an integer input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param minval: The minimum value of the input
        :param maxval: The maximum value of the input
        :param step: The step value of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param options: A tuple of integers that the user can select from
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='int',
            defval=defval,
            title=title,
            minval=minval,
            maxval=maxval,
            step=step,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            options=options,
            display=display,
        )
        return defval if _id not in old_input_values else safe_convert.safe_int(old_input_values[_id])

    @classmethod
    def _bool(cls, defval: bool, title: str | None = None, *,
              tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
              confirm: bool | None = False, display: _display.Display | None = None,
              _id: str | None = None, **__) -> bool:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a boolean input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='bool',
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            display=display,
        )
        return defval if _id not in old_input_values else bool(old_input_values[_id])

    @classmethod
    def _float(cls, defval: float, title: str | None = None, *,
               tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
               confirm: bool | None = False, options: tuple[int] | None = None,
               minval: float | None = None, maxval: float | None = None, step: float | None = None,
               display: _display.Display | None = None, _id: str | None = None, **__) -> float | NA[float]:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a float input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param minval: The minimum value of the input
        :param maxval: The maximum value of the input
        :param step: The step value of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param options: A tuple of integers that the user can select from
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='float',
            defval=defval,
            title=title,
            minval=minval,
            maxval=maxval,
            step=step,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            options=options,
            display=display,
        )
        return defval if _id not in old_input_values else safe_convert.safe_float(old_input_values[_id])

    @classmethod
    def string(cls, defval: str, title: str | None = None, *,
               tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
               confirm: bool | None = False,
               options: tuple[str] | None = None,
               display: _display.Display | None = None,
               _id: str | None = None, **__) -> str:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a string input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param display: Controls where the script will display the input's information
        :param options: A tuple of strings that the user can select from
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='string',
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            display=display,
            options=options,
        )
        return defval if _id not in old_input_values else str(old_input_values[_id])

    @classmethod
    def color(cls, defval: Color, title: str | None = None, *,
              tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
              confirm: bool | None = False, display: _display.Display | None = None,
              _id: str | None = None, **__) -> Color:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a color input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='color',
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            display=display,
        )
        return defval if _id not in old_input_values else Color(old_input_values[_id])

    @classmethod
    def source(cls, defval: str | Source, title: str | None = None, *,
               tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
               confirm: bool | None = False, display: _display.Display | None = None,
               _id: str | None = None, **__) -> Series[float]:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a source input to the script's inputs.

        :param defval: The name of the "source" registered in lib module,
                       like "open", "high", "low", "close", "hl2", "hlc3", "ohlc4"
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param display: Controls where the script will display the input's information
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        defval = str(defval)
        inputs[_id] = InputData(
            id=_id,
            input_type='source',
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            display=display,
        )
        # We actually return a string here, but the InputTransformer will add a `getattr()` call to get the
        return cast(Series[float], defval if _id not in old_input_values else old_input_values[_id])

    @classmethod
    def enum(cls, defval: TEnum, title: str | None = None, *,
             tooltip: str | None = None, inline: bool | None = False, group: str | None = None,
             confirm: bool | None = False,
             options: tuple[str] | None = None,
             display: _display.Display | None = None,
             _id: str | None = None, **__) -> TEnum:
        """
        Adds an input to your script's settings, which allows you to provide configuration options
        to script users. This function adds a field for a enum input to the script's inputs.

        :param defval: The default value of the input
        :param title: The title of the input
        :param tooltip: The tooltip of the input
        :param inline: If True, the input will be displayed inline
        :param group: The group of the input
        :param confirm: If True, the user will be asked to confirm the input
        :param display: Controls where the script will display the input's information
        :param options: A tuple of strings that the user can select from
        :param _id: The unique identifier of the input, it is filled by the InputTransformer
        :return: The input value from toml file or the default
        """
        inputs[_id] = InputData(
            id=_id,
            input_type='enum',
            defval=defval,
            title=title,
            tooltip=tooltip,
            inline=inline,
            group=group,
            confirm=confirm,
            display=display,
            options=options,
        )
        if _id not in old_input_values:
            return defval
        else:
            # Convert string value back to the specific enum type
            value = old_input_values[_id]
            if isinstance(value, str):
                try:
                    return defval.__class__(value)
                except ValueError:
                    return defval
            return defval

    int = _int
    bool = _bool
    float = _float

    # We don't have interactive inputs, so it is the same as float
    price = _float

    # These are incomplete, but good workaround
    session = string
    symbol = string
    timeframe = string
    textarea = string

    # time() returns UNIX timestamp in milliseconds (int)
    time = _int


# noinspection PyShadowingBuiltins
input = _Input()
