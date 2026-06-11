from ..types.hline import HLineEnum, HLine
from . import color as _color, display as _display


# IDE-facing view of the function-and-namespace module: user code reads the
# constants and calls the bare name; the AST transformer resolves both at runtime.
class HLineModule:
    style_solid: HLineEnum
    style_dotted: HLineEnum
    style_dashed: HLineEnum

    def __call__(
            self,
            price: float,
            title: str = ...,
            color: _color.Color = ...,
            linestyle: HLineEnum = ...,
            linewidth: int = ...,
            editable: bool = ...,
            display: _display.Display = ...
    ) -> HLine: ...


hline: HLineModule
