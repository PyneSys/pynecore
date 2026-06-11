from ..types.alert import AlertEnum


# IDE-facing view of the function-and-namespace module: user code reads the
# constants and calls the bare name; the AST transformer resolves both at runtime.
class AlertModule:
    freq_all: AlertEnum
    freq_once_per_bar: AlertEnum
    freq_once_per_bar_close: AlertEnum

    def __call__(self, message: str, freq: AlertEnum = ...) -> None: ...


alert: AlertModule
