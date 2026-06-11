"""
@pyne

Regression script: a non-literal ``+=`` on a varip float allocates a Kahan
compensation slot, and the companion must carry the varip flag too — a var
rollback restoring the compensation while the sum survives would
desynchronize the pair.
"""
from pynecore import Persistent
from pynecore.types import IBPersistent

some_value = 0.1


def main():
    plain: Persistent[float] = 0.0
    varip_total: IBPersistent[float] = 0.0
    plain += some_value
    varip_total += some_value
