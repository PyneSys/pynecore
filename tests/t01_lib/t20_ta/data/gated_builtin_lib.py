"""
@pyne

Library module for ``test_368``: a module-level helper reading a stateful
builtin variable (the hand-written export surface, per-call-site gated), plus a
Pine-convention library ``main`` whose demo body pins the study-mode laws
(m577/m578) when the module is run directly as the script: global-scope and
non-exported-function reads are engine-global, an ``@export`` function keeps
its own call-gated machine.
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import script, ta, bar_index
from pynecore.types import Persistent


def lib_nvi() -> float:
    return ta.nvi


exported_nvi = Exported()


@script.library("Gated Builtin Lib")
def main():
    @export
    def exported_nvi() -> float:
        return ta.nvi

    def local_nvi() -> float:
        return ta.nvi

    runs: Persistent[int] = 0
    runs += 1
    every_nvi = ta.nvi
    gated_direct = -1.0
    gated_local = -1.0
    gated_export = -1.0
    if bar_index % 2 == 0:
        gated_direct = ta.nvi
        gated_local = local_nvi()
        gated_export = exported_nvi()
    return {"every_nvi": every_nvi, "gated_direct": gated_direct,
            "gated_local": gated_local, "gated_export": gated_export,
            "runs": runs}
