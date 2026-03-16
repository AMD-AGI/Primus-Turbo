###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import os
import sys
from enum import Enum

# NOTE: global log level environment variable to control the log level of the entire primus_turbo library.
_LOG_LEVEL_ENV = "PRIMUS_TURBO_LOG_LEVEL"
_ROOT_LOGGER_NAME = "primus_turbo"

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_logger_initialized = False


class LogLevelEnum(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.name


def _resolve_log_level() -> int:
    """Resolve log level from environment variable, defaulting to WARNING."""
    level_str = os.environ.get(_LOG_LEVEL_ENV, LogLevelEnum.WARNING.value).upper()
    numeric = getattr(logging, level_str, None)
    if not isinstance(numeric, int):
        numeric = logging.WARNING
    return numeric


def _ensure_root_logger() -> logging.Logger:
    """Initialise the library root logger exactly once."""
    global _root_logger_initialized
    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _root_logger_initialized:
        return logger

    logger.setLevel(_resolve_log_level())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
        logger.addHandler(handler)

    _root_logger_initialized = True
    return logger


def _get_logger() -> logging.Logger:
    """
    Get a logger under the ``primus_turbo`` namespace.
    """
    _ensure_root_logger()
    return logging.getLogger(_ROOT_LOGGER_NAME)


def _get_rank() -> int:
    """Get the current distributed rank via ``torch.distributed`` if available, else 0."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
        # TODO(ruibin): support jax distributed.
    except Exception:
        pass
    return 0


def set_log_level(level: int) -> None:
    """
    Override the log level for the entire ``primus_turbo`` logger hierarchy.
    """
    _ensure_root_logger()
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(level)


_log_once_cache: set = set()


def log_once(level: int, msg: str, *args, **kwargs) -> None:
    """
    Log *msg* at most once per process for the given *(logger, level, msg)* triple.

    Useful inside hot loops to avoid flooding the log output.
    """
    logger = _get_logger()
    key = (logger.name, level, msg)
    if key not in _log_once_cache:
        _log_once_cache.add(key)
        logger.log(level, msg, *args, **kwargs)


def log_debug(msg: str, *args, **kwargs) -> None:
    _get_logger().debug(msg, *args, **kwargs)


def log_info(msg: str, *args, **kwargs) -> None:
    _get_logger().info(msg, *args, **kwargs)


def log_warning(msg: str, *args, **kwargs) -> None:
    _get_logger().warning(msg, *args, **kwargs)


def log_error(msg: str, *args, **kwargs) -> None:
    _get_logger().error(msg, *args, **kwargs)


def log_debug_once(msg: str, *args, **kwargs) -> None:
    log_once(logging.DEBUG, msg, *args, **kwargs)


def log_info_once(msg: str, *args, **kwargs) -> None:
    log_once(logging.INFO, msg, *args, **kwargs)


def log_warning_once(msg: str, *args, **kwargs) -> None:
    log_once(logging.WARNING, msg, *args, **kwargs)


def log_error_once(msg: str, *args, **kwargs) -> None:
    log_once(logging.ERROR, msg, *args, **kwargs)


def _log_rank0(level: int, msg: str, *args, **kwargs) -> None:
    if _get_rank() == 0:
        _get_logger().log(level, msg, *args, **kwargs)


def _log_once_rank0(level: int, msg: str, *args, **kwargs) -> None:
    if _get_rank() == 0:
        log_once(level, msg, *args, **kwargs)


def log_debug_rank0(msg: str, *args, **kwargs) -> None:
    _log_rank0(logging.DEBUG, msg, *args, **kwargs)


def log_info_rank0(msg: str, *args, **kwargs) -> None:
    _log_rank0(logging.INFO, msg, *args, **kwargs)


def log_warning_rank0(msg: str, *args, **kwargs) -> None:
    _log_rank0(logging.WARNING, msg, *args, **kwargs)


def log_error_rank0(msg: str, *args, **kwargs) -> None:
    _log_rank0(logging.ERROR, msg, *args, **kwargs)


def log_debug_once_rank0(msg: str, *args, **kwargs) -> None:
    _log_once_rank0(logging.DEBUG, msg, *args, **kwargs)


def log_info_once_rank0(msg: str, *args, **kwargs) -> None:
    _log_once_rank0(logging.INFO, msg, *args, **kwargs)


def log_warning_once_rank0(msg: str, *args, **kwargs) -> None:
    _log_once_rank0(logging.WARNING, msg, *args, **kwargs)


def log_error_once_rank0(msg: str, *args, **kwargs) -> None:
    _log_once_rank0(logging.ERROR, msg, *args, **kwargs)
