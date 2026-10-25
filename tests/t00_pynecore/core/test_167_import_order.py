"""
Any PyneCore module can be the first one a program imports.

``pynecore.core.script`` imports submodules of ``pynecore.lib``, so it loads the lib
package; ``pynecore.lib`` re-exports ``script`` and ``input`` from it. Importing
``core.script`` first once failed on that cycle, and so did every module importing it
at its top (the profiler among them). Each case runs in a fresh interpreter: in this
one the modules are long imported.
"""
import subprocess
import sys

import pytest

_PROBE = '''
import importlib, sys
importlib.import_module(sys.argv[1])
from pynecore.lib import script, input
import pynecore.core.script as core_script
assert script is core_script.script and input is core_script.input
'''


@pytest.mark.parametrize('module', [
    'pynecore.core.script',
    'pynecore.core.profiler',
    'pynecore.lib',
    'pynecore.core.script_runner',
])
def __test_module_imports_first__(module, tmp_path):
    """ The module imports in a fresh interpreter, and the lib still hands out ``script`` """
    # Run away from the repository: from its root, ``pynecore`` resolves as a namespace
    # package and skips the package init that installs the import hook
    completed = subprocess.run([sys.executable, '-c', _PROBE, module], cwd=tmp_path,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
