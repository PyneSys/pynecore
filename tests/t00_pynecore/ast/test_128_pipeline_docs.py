"""
The step tables of ``docs/advanced/ast-transformations.md`` against ``pipeline.py``.

The page lists every step of the two pipeline halves in a table between HTML
comment markers (``<!-- pipeline:analysis -->`` ... ``<!-- /pipeline:analysis -->``
and the same for ``lowering``). The first column of a row is the step name in
backticks. A rule of a fused phase is written ``Phase/Rule``, directly under the
row of the phase. The ``User code only`` column holds ``yes`` or ``no`` for a
step or a rule and ``-`` for a phase, which has no such flag of its own.

These tests fail when a step is added, removed, renamed or moved in
``ANALYSIS`` / ``LOWERING`` without the page following.
"""
import re
from collections.abc import Sequence
from pathlib import Path

from pynecore.transformers.pipeline import ANALYSIS, LOWERING, Phase, Step

__test_helper_DOC = Path(__file__).resolve().parents[3] / 'docs' / 'advanced' / 'ast-transformations.md'

#: The columns the tables have, in order
__test_helper_HEADER = ('Step', 'Purpose', 'User code only', 'Ordering constraint')


def __test_helper_declared(steps: Sequence[Step | Phase]) -> list[tuple[str, str]]:
    """ The (name, user code only) rows the table of a pipeline half has to have """
    rows: list[tuple[str, str]] = []
    for step in steps:
        if isinstance(step, Phase):
            rows.append((step.name, '-'))
            rows.extend((f'{step.name}/{rule.name}', 'yes' if rule.user_code_only else 'no')
                        for rule in step.rules)
        else:
            rows.append((step.name, 'yes' if step.user_code_only else 'no'))
    return rows


def __test_helper_documented(marker: str) -> list[tuple[str, str]]:
    """ The (name, user code only) rows of the table between the markers of ``marker`` """
    text = __test_helper_DOC.read_text(encoding='utf-8')
    found = re.search(rf'<!-- pipeline:{marker} -->\n(.*?)\n<!-- /pipeline:{marker} -->',
                      text, re.DOTALL)
    assert found is not None, f"markers of the {marker} table are missing from {__test_helper_DOC.name}"
    lines = found.group(1).strip().splitlines()
    assert len(lines) > 2, f"the {marker} table has no rows"
    assert all(line.startswith('|') and line.endswith('|') for line in lines), \
        f"the {marker} table block holds something other than table lines"

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip('|').split('|')]

    assert tuple(cells(lines[0])) == __test_helper_HEADER, f"unexpected header: {lines[0]}"
    assert set(lines[1].replace('|', '').replace(' ', '')) == {'-'}, \
        "the line under the header is no separator"

    rows: list[tuple[str, str]] = []
    for line in lines[2:]:
        row = cells(line)
        assert len(row) == len(__test_helper_HEADER), f"wrong column count: {line}"
        name = re.fullmatch(r'`([A-Za-z0-9_/]+)`', row[0])
        assert name is not None, f"the step name must be one code span: {row[0]!r}"
        rows.append((name.group(1), row[2]))
    return rows


def __test_analysis_table_matches_pipeline__():
    """ The analysis table lists the steps of ANALYSIS, in order, with their flags """
    assert __test_helper_documented('analysis') == __test_helper_declared(ANALYSIS)


def __test_lowering_table_matches_pipeline__():
    """ The lowering table lists the steps and the phase rules of LOWERING, in order """
    assert __test_helper_documented('lowering') == __test_helper_declared(LOWERING)


def __test_step_names_are_unique__():
    """ A name identifies a step in the tables and in the timing report """
    names = [name for name, _ in __test_helper_declared(ANALYSIS) + __test_helper_declared(LOWERING)]
    assert len(names) == len(set(names))
