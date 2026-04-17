###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for primus_turbo.common.logger.PrimusTurboLogger.

All tests are CPU-only and do not require a GPU or distributed setup.
"""

import logging

import pytest

from primus_turbo.common.logger import PrimusTurboLogger


class _CapturingHandler(logging.Handler):
    """Minimal in-memory handler for capturing log records."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture()
def fresh_logger():
    """Yield a PrimusTurboLogger with a clean once-cache and a capturing handler."""
    inst = PrimusTurboLogger()
    inst._log_once_cache.clear()

    handler = _CapturingHandler()
    handler.setLevel(logging.DEBUG)
    original_level = inst._logger.level
    inst._logger.setLevel(logging.DEBUG)
    inst._logger.addHandler(handler)

    yield inst, handler

    inst._logger.removeHandler(handler)
    inst._logger.setLevel(original_level)
    inst._log_once_cache.clear()


class TestPrimusTurboLoggerSingleton:
    def test_singleton_identity(self):
        a = PrimusTurboLogger()
        b = PrimusTurboLogger()
        assert a is b

    def test_singleton_across_imports(self):
        from primus_turbo.common.logger import logger as logger_ref

        assert logger_ref is PrimusTurboLogger()


class TestLogOnce:
    def test_log_once_emits_first_call(self, fresh_logger):
        inst, handler = fresh_logger
        inst.debug("unique-once-msg", once=True)
        assert "unique-once-msg" in handler.messages()

    def test_log_once_suppresses_second_call(self, fresh_logger):
        inst, handler = fresh_logger
        inst.debug("repeat-msg", once=True)
        inst.debug("repeat-msg", once=True)
        assert handler.messages().count("repeat-msg") == 1

    def test_log_multiple_times_without_once(self, fresh_logger):
        inst, handler = fresh_logger
        inst.debug("multi-msg")
        inst.debug("multi-msg")
        assert handler.messages().count("multi-msg") == 2

    def test_different_levels_are_independent_in_once_cache(self, fresh_logger):
        inst, handler = fresh_logger
        inst.debug("level-msg", once=True)
        inst.info("level-msg", once=True)
        assert handler.messages().count("level-msg") == 2


class TestRankFiltering:
    def test_rank_none_always_logs(self, fresh_logger):
        inst, handler = fresh_logger
        inst.debug("rank-none-msg", rank=None)
        assert "rank-none-msg" in handler.messages()

    def test_rank_0_logs_when_rank_is_0(self, fresh_logger, monkeypatch):
        inst, handler = fresh_logger
        monkeypatch.setattr(inst, "_get_rank", lambda: 0)
        inst.debug("rank0-msg", rank=0)
        assert "rank0-msg" in handler.messages()

    def test_rank_1_suppressed_when_rank_is_0(self, fresh_logger, monkeypatch):
        inst, handler = fresh_logger
        monkeypatch.setattr(inst, "_get_rank", lambda: 0)
        inst.debug("rank1-msg", rank=1)
        assert "rank1-msg" not in handler.messages()


class TestLogLevelEnv:
    def test_set_level_takes_effect(self, fresh_logger):
        inst, handler = fresh_logger
        inst.set_level(logging.DEBUG)
        inst.debug("should-appear")
        assert "should-appear" in handler.messages()

    def test_resolve_log_level_unknown_string_falls_back_to_warning(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_LOG_LEVEL", "NOTAVALIDLEVEL")
        level = PrimusTurboLogger._resolve_log_level()
        assert level == logging.WARNING

    def test_resolve_log_level_debug(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_LOG_LEVEL", "DEBUG")
        level = PrimusTurboLogger._resolve_log_level()
        assert level == logging.DEBUG

    def test_resolve_log_level_error(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_LOG_LEVEL", "ERROR")
        level = PrimusTurboLogger._resolve_log_level()
        assert level == logging.ERROR
