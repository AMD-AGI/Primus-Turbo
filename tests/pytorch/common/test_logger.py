###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import logging
from unittest.mock import Mock

import pytest
import torch.distributed as dist

from primus_turbo.common.logger import LogLevelEnum, PrimusTurboLogger


@pytest.mark.parametrize(
    "value, level",
    [
        ("info", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("bad", logging.WARNING),
        ("getLogger", logging.WARNING),
    ],
)
def test_environment_log_level(monkeypatch, value, level):
    monkeypatch.setenv("PRIMUS_TURBO_LOG_LEVEL", value)
    assert PrimusTurboLogger._resolve_log_level() == level


@pytest.fixture
def logger(monkeypatch):
    # Isolate the singleton and underlying logging state from the library logger.
    monkeypatch.setattr(PrimusTurboLogger, "_instance", None)
    sink = Mock(spec=logging.Logger)
    monkeypatch.setattr(PrimusTurboLogger, "_setup_logger", lambda self: sink)
    obj = PrimusTurboLogger("coverage-test")
    assert PrimusTurboLogger() is obj
    return obj, sink


@pytest.mark.parametrize(
    "method, level",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
    ],
)
def test_logging_forwards_format_arguments(logger, method, level):
    obj, sink = logger
    getattr(obj, method)("value=%s", 17, exc_info=True)
    sink.log.assert_called_once_with(level, "value=%s", 17, exc_info=True)


def test_rank_filter_does_not_consume_once(logger, monkeypatch):
    obj, sink = logger
    monkeypatch.setattr(obj, "_get_rank", lambda: 1)
    obj.info("message", once=True, rank=0)
    sink.log.assert_not_called()
    monkeypatch.setattr(obj, "_get_rank", lambda: 0)
    obj.info("message", once=True, rank=0)
    obj.info("message", once=True, rank=0)
    assert sink.log.call_count == 1
    obj.warning("message", once=True)
    assert sink.log.call_count == 2


def test_set_level(logger):
    obj, sink = logger
    obj.set_level(logging.DEBUG)
    sink.setLevel.assert_called_once_with(logging.DEBUG)
    assert str(LogLevelEnum.DEBUG) == "DEBUG"


@pytest.mark.parametrize("initialized, expected", [(False, 0), (True, 3)])
def test_distributed_rank(monkeypatch, initialized, expected):
    monkeypatch.setattr(dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    assert PrimusTurboLogger._get_rank() == expected


def test_distributed_rank_failure_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", Mock(side_effect=RuntimeError("not ready")))
    assert PrimusTurboLogger._get_rank() == 0
