"""Regression coverage for numeric-array normalization in log comparisons."""

from pathlib import Path
import subprocess
import sys

import pytest

from pynecore.lib import log


@pytest.mark.parametrize(('expected', 'actual'), [
    ('[1.23, 0.0, -2.5]', '[1.234, -0.00001, -2.5]'),
    ('[100.0, -0.25, 3.0]', '[1e+2, -25e-2, 3.]'),
    ('values: [1.0, 2.0] / [3.0]', 'values: [1,\t 2] / [3]'),
    ('[[1.23], [2.35]]', '[[1.234], [2.346]]'),
    ('[1.0, 2.0]', '[١, ٢]'),
    ('[ 1] [+1] [.5] [1E2] [1,] []', '[ 1] [+1] [.5] [1E2] [1,] []'),
])
def __test_numeric_array_log_comparison__(log_comparator, expected, actual):
    """Normalize supported numbers while preserving unsupported array forms."""
    with log_comparator(f'[2026-01-01T00:00:00]: {expected}', float_precision=2):
        log.logger.warning(actual)


def __test_malformed_numeric_arrays_complete_without_backtracking__():
    """Bound execution time for missing delimiters, long digits and separators."""
    conftest = Path(__file__).with_name('conftest.py')
    code = '''
import inspect
import runpy
import sys

fixture = runpy.run_path(sys.argv[1])['log_comparator'].__wrapped__
comparator = fixture(None)
normalize = inspect.getclosurevars(comparator.__wrapped__).nonlocals['round_numbers_in_array']
for text in (
    '[' + '99,' * 100_000,
    '[' + '9' * 100_000 + 'x]',
    '[1e' + '9' * 100_000 + 'x]',
    '[1,' + ' ' * 100_000 + 'x]',
    '[' * 100_000 + 'x]',
):
    assert normalize(text) == text
assert normalize('[1.234, -0.00001]', 2) == '[1.23, 0.0]'
'''
    subprocess.run(
        [sys.executable, '-c', code, str(conftest)],
        check=True, timeout=10, capture_output=True, text=True,
    )
