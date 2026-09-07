"""
@pyne

Importing script of ``test_097``: it reaches the exported method the way a
compiled Pine script does, through ``method_call``, which resolves the name on
the library module at call time.
"""
from pynecore.core.pine_method import method_call
from pynecore.lib import close, script

import export_once_shadow_lib as shadow


@script.indicator("Export Once Shadow Script")
def main():
    return {"picked": method_call('picked', shadow.Holder.new(close))}
