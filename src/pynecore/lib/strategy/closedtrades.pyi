from ...core.callable_module import CallableModule
from ...types.na import NA


class ClosedTradesModule(CallableModule, int):

    def __call__(self) -> int: ...

    def new(self): ...

    #
    # Functions
    #

    def commission(self, trade_num: int) -> float | NA[float]: ...

    def entry_bar_index(self, trade_num: int) -> int | NA[int]: ...

    def entry_comment(self, trade_num: int) -> str | NA[str]: ...

    def entry_id(self, trade_num: int) -> str | NA[str]: ...

    def entry_price(self, trade_num: int) -> float | NA[float]: ...

    def entry_time(self, trade_num: int) -> int | NA[int]: ...

    def exit_bar_index(self, trade_num: int) -> int | NA[int]: ...

    def exit_comment(self, trade_num: int) -> str | NA[str]: ...

    def exit_id(self, trade_num: int) -> str | NA[str]: ...

    def exit_price(self, trade_num: int) -> float | NA[float]: ...

    def exit_time(self, trade_num: int) -> int | NA[int]: ...

    def max_drawdown(self, trade_num: int) -> float | NA[float]: ...

    def max_drawdown_percent(self, trade_num: int) -> float | NA[float]: ...

    def max_runup(self, trade_num: int) -> float | NA[float]: ...

    def max_runup_percent(self, trade_num: int) -> float | NA[float]: ...

    def profit(self, trade_num: int) -> float | NA[float]: ...

    def profit_percent(self, trade_num: int) -> float | NA[float]: ...

    def size(self, trade_num: int) -> float: ...


closedtrades: ClosedTradesModule = ClosedTradesModule(__name__)
