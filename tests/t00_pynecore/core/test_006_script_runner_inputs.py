"""
@pyne
"""
from pynecore.lib import script, input, close, plot


@script.indicator(title="Input Override Test")
def main(
    length=input.int(10, "Length"),
    multiplier=input.float(1.5, "Multiplier"),
):
    plot(length, "length")
    plot(multiplier, "multiplier")
    plot(close * multiplier, "result")


def __test_inputs_default__(csv_reader, runner):
    """ ScriptRunner uses default input values when no inputs provided """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for candle, _plot in runner(cr).run_iter():
            assert _plot["length"] == 10
            assert _plot["multiplier"] == 1.5
            break


def __test_inputs_override__(script_path, module_key, syminfo, csv_reader):
    """ ScriptRunner inputs parameter overrides default values """
    import sys
    from pynecore.core.script_runner import ScriptRunner

    # Remove module to force re-import with new inputs
    # import_script uses script_path.stem as the module name
    stem = script_path.stem
    for key in [module_key, stem]:
        sys.modules.pop(key, None)

    with csv_reader('series_if_for.csv', subdir="data") as cr:
        r = ScriptRunner(script_path, cr, syminfo, inputs={"length": 20, "multiplier": 3.0})
        for candle, _plot in r.run_iter():
            assert _plot["length"] == 20
            assert _plot["multiplier"] == 3.0
            break


def __test_inputs_partial_override__(script_path, module_key, syminfo, csv_reader):
    """ ScriptRunner inputs parameter can override only some values """
    import sys
    from pynecore.core.script_runner import ScriptRunner

    stem = script_path.stem
    for key in [module_key, stem]:
        sys.modules.pop(key, None)

    with csv_reader('series_if_for.csv', subdir="data") as cr:
        r = ScriptRunner(script_path, cr, syminfo, inputs={"length": 50})
        for candle, _plot in r.run_iter():
            assert _plot["length"] == 50
            assert _plot["multiplier"] == 1.5
            break
