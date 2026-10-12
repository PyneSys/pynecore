"""@pyne
SPDX-License-Identifier: Apache-2.0
Synthetic example: consume a feature aligned with the current input bar.
"""
from pynecore.lib import extra_fields, input, plot, script


@script.indicator("Synthetic cached feature")
def main(offset=input.float(0)):
    plot(extra_fields["feature"] + offset, "value")
