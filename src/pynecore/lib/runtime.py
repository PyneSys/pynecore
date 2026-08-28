from .. import lib


def error(message: str):
    """
    Stop running script with an error message

    A ``request.security`` child re-runs the whole script at the context's
    timeframe, but TradingView evaluates only the REQUESTED EXPRESSION in that
    context — a chart-level resolution guard never reaches it. Raising in the
    child would kill the worker and take the chart run down with it, so the call
    is a no-op there.
    """
    # noinspection PyProtectedMember
    if lib._in_security:
        return
    raise RuntimeError(message)
