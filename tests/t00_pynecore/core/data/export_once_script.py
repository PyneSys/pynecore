"""
@pyne

Importing script of ``test_097``: it calls both exports of ``export_once_lib``
across the module boundary.
"""
from pynecore.lib import close, script

from export_once_lib import scaled, smoothed


@script.indicator("Export Once Script")
def main():
    return {"scaled": scaled(close), "smoothed": smoothed(close, 3)}
