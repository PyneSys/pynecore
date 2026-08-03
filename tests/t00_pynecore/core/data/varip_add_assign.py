"""
@pyne

Regression script: a ``+=`` accumulator declared varip must keep the varip
flag on its slot, so a var rollback leaves it alone while the plain one is
restored.
"""
from pynecore import Persistent
from pynecore.types import IBPersistent

some_value = 0.1


def main():
    plain: Persistent[float] = 0.0
    varip_total: IBPersistent[float] = 0.0
    plain += some_value
    varip_total += some_value
