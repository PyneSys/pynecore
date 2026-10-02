"""Regression tests for shared-workdir startup error logging."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from pynecore.cli.utils.error_hook import setup_global_error_logging


@pytest.fixture
def __test_helper_original_hook__(monkeypatch):
    """Restore the process's exception hook after each test."""
    original_hook = Mock()
    monkeypatch.setattr(sys, 'excepthook', original_hook)
    return original_hook


@pytest.mark.parametrize('existing_log', [False, True])
def __test_setup_cleans_log_and_preserves_exception_reporting__(
        tmp_path, existing_log, __test_helper_original_hook__):
    """Startup clears an old log and reports new exceptions through both hooks."""
    log_path = tmp_path / 'output' / 'logs' / 'error.log'
    if existing_log:
        log_path.parent.mkdir(parents=True)
        log_path.write_text('Earlier failure\n', encoding='utf-8')

    setup_global_error_logging(log_path)

    assert log_path.parent.is_dir()
    assert not log_path.exists()
    try:
        raise ValueError('Startup test failure')
    except ValueError as error:
        sys.excepthook(type(error), error, error.__traceback__)
        __test_helper_original_hook__.assert_called_once_with(
            ValueError, error, error.__traceback__)

    log_text = log_path.read_text(encoding='utf-8')
    assert 'Traceback (most recent call last):' in log_text
    assert 'ValueError: Startup test failure' in log_text
    assert 'Earlier failure' not in log_text


def __test_setup_survives_competing_log_deletion__(
        tmp_path, monkeypatch, __test_helper_original_hook__):
    """Startup succeeds when another process deletes the shared log first."""
    log_path = tmp_path / 'output' / 'logs' / 'error.log'
    log_path.parent.mkdir(parents=True)
    log_path.write_text('Earlier failure\n', encoding='utf-8')
    original_unlink = Path.unlink
    competing_deletions = []

    def unlink_after_competing_delete(path, *args, **kwargs):
        if path == log_path:
            original_unlink(path)
            competing_deletions.append(path)
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink_after_competing_delete)
        setup_global_error_logging(log_path)

    assert competing_deletions == [log_path]
    assert not log_path.exists()
    assert sys.excepthook is not __test_helper_original_hook__
