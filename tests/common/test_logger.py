###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for ``primus_turbo.common.logger``.

The logger is a process-wide singleton with behavior that is easy to break
during refactors:

* ``_resolve_log_level`` reads the ``PRIMUS_TURBO_LOG_LEVEL`` env var through
  the ``ENV_LOG_LEVEL`` constant (renamed in PR #305). A typo in either place
  would silently fall back to ``WARNING`` and hide debug output in production.
* ``log(..., once=True)`` must dedupe on ``(level, msg)``.
* ``log(..., rank=N)`` must suppress output on non-matching ranks.

These tests avoid touching the singleton's handlers so they stay hermetic.
"""

import logging

import pytest

from primus_turbo.common import logger as logger_module
from primus_turbo.common.constants import ENV_LOG_LEVEL
from primus_turbo.common.logger import LogLevelEnum, PrimusTurboLogger


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)
    return monkeypatch


class TestResolveLogLevel:
    """Covers PrimusTurboLogger._resolve_log_level env parsing."""

    def test_default_is_warning_when_unset(self, clean_env):
        assert PrimusTurboLogger._resolve_log_level() == logging.WARNING

    @pytest.mark.parametrize(
        "level_str,expected",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_all_documented_levels(self, clean_env, level_str, expected):
        clean_env.setenv(ENV_LOG_LEVEL, level_str)
        assert PrimusTurboLogger._resolve_log_level() == expected

    def test_case_insensitive(self, clean_env):
        clean_env.setenv(ENV_LOG_LEVEL, "debug")
        assert PrimusTurboLogger._resolve_log_level() == logging.DEBUG

    def test_invalid_level_falls_back_to_warning(self, clean_env):
        """Unknown/garbage values must not crash; they fall back to WARNING."""
        clean_env.setenv(ENV_LOG_LEVEL, "NOT_A_LEVEL")
        assert PrimusTurboLogger._resolve_log_level() == logging.WARNING

    def test_non_logging_attribute_not_accepted(self, clean_env):
        """`logging` exposes non-integer attributes (e.g. ``root``). Those must
        not be accepted as a level — only real integer level constants."""
        clean_env.setenv(ENV_LOG_LEVEL, "root")
        assert PrimusTurboLogger._resolve_log_level() == logging.WARNING

    def test_uses_env_log_level_constant(self, clean_env):
        """Regression: logger must read the env var named by ENV_LOG_LEVEL,
        not a hardcoded string."""
        clean_env.setenv(ENV_LOG_LEVEL, "ERROR")
        assert PrimusTurboLogger._resolve_log_level() == logging.ERROR
        clean_env.delenv(ENV_LOG_LEVEL, raising=False)
        assert PrimusTurboLogger._resolve_log_level() == logging.WARNING


class TestLogLevelEnum:
    def test_str_returns_name(self):
        assert str(LogLevelEnum.DEBUG) == "DEBUG"
        assert str(LogLevelEnum.WARNING) == "WARNING"

    def test_values_match_names(self):
        for member in LogLevelEnum:
            assert member.value == member.name


class TestLoggerSingleton:
    def test_singleton_identity(self):
        a = PrimusTurboLogger()
        b = PrimusTurboLogger()
        assert a is b
        assert a is logger_module.logger


class _RecordingHandler(logging.Handler):
    """Minimal handler that captures emitted records for inspection."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_logger(monkeypatch):
    """Attach a recording handler to the singleton and yield (logger, records).

    Cleans up the handler and the ``once`` dedup cache afterwards so tests
    don't leak state into each other or the rest of the suite.
    """
    lg = logger_module.logger
    handler = _RecordingHandler()
    prior_level = lg._logger.level
    lg._logger.addHandler(handler)
    lg.set_level(logging.DEBUG)
    lg._log_once_cache.clear()
    try:
        yield lg, handler.records
    finally:
        lg._logger.removeHandler(handler)
        lg._logger.setLevel(prior_level)
        lg._log_once_cache.clear()


class TestLoggerBehavior:
    def test_info_emits(self, captured_logger):
        lg, records = captured_logger
        lg.info("hello")
        assert len(records) == 1
        assert records[0].getMessage() == "hello"
        assert records[0].levelno == logging.INFO

    def test_once_dedups_same_message(self, captured_logger):
        lg, records = captured_logger
        for _ in range(5):
            lg.warning("flaky-config", once=True)
        assert len(records) == 1

    def test_once_per_level_distinct_messages(self, captured_logger):
        """``once`` keys by (level, msg); different messages still emit."""
        lg, records = captured_logger
        lg.warning("m1", once=True)
        lg.warning("m2", once=True)
        lg.warning("m1", once=True)
        assert [r.getMessage() for r in records] == ["m1", "m2"]

    def test_once_distinct_across_levels(self, captured_logger):
        """Same string at different levels should each log once."""
        lg, records = captured_logger
        lg.info("shared", once=True)
        lg.warning("shared", once=True)
        lg.info("shared", once=True)
        assert len(records) == 2
        assert {r.levelno for r in records} == {logging.INFO, logging.WARNING}

    def test_rank_matches_default_zero(self, captured_logger):
        """Without torch.distributed initialized, _get_rank()==0."""
        lg, records = captured_logger
        lg.info("on-zero", rank=0)
        assert len(records) == 1

    def test_rank_mismatch_suppresses(self, captured_logger, monkeypatch):
        """rank=1 filter must drop the message on rank 0."""
        lg, records = captured_logger
        monkeypatch.setattr(PrimusTurboLogger, "_get_rank", staticmethod(lambda: 0))
        lg.info("only-on-rank-1", rank=1)
        assert records == []

    def test_rank_filter_before_once_cache(self, captured_logger, monkeypatch):
        """If a rank-filtered call is dropped, it must NOT poison the ``once``
        cache — otherwise the allowed rank could never log the message."""
        lg, records = captured_logger
        monkeypatch.setattr(PrimusTurboLogger, "_get_rank", staticmethod(lambda: 0))
        lg.info("privileged", once=True, rank=1)
        assert records == []
        monkeypatch.setattr(PrimusTurboLogger, "_get_rank", staticmethod(lambda: 1))
        lg.info("privileged", once=True, rank=1)
        assert len(records) == 1

    def test_error_debug_warning_paths(self, captured_logger):
        lg, records = captured_logger
        lg.debug("d")
        lg.info("i")
        lg.warning("w")
        lg.error("e")
        levels = [r.levelno for r in records]
        assert levels == [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]

    def test_set_level_filters_below(self, captured_logger):
        lg, records = captured_logger
        lg.set_level(logging.ERROR)
        lg.info("below")
        lg.warning("below")
        lg.error("above")
        assert [r.getMessage() for r in records] == ["above"]

    def test_get_rank_returns_int_without_distributed(self):
        """Exercises the exception path when torch.distributed is not init'd."""
        assert PrimusTurboLogger._get_rank() == 0
