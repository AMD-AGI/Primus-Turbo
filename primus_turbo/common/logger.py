###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import os
import sys
from enum import Enum

from primus_turbo.common.constants import ENV_LOG_LEVEL

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogLevelEnum(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.name


class PrimusTurboLogger:
    """Singleton logger for the primus_turbo library.

    Parameters ``once`` and ``rank`` can be combined freely on every call:

    * ``logger.info("msg")``                – log on all ranks, every time
    * ``logger.info("msg", once=True)``     – log at most once per process
    * ``logger.info("msg", rank=0)``        – log only on rank 0
    * ``logger.info("msg", once=True, rank=0)`` – once, rank 0 only
    """

    _instance: "PrimusTurboLogger | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "primus_turbo"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._name = name
        self._log_once_cache: set = set()
        self._logger = self._setup_logger()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_log_level() -> int:
        level_str = os.environ.get(ENV_LOG_LEVEL, LogLevelEnum.WARNING.value).upper()
        numeric = getattr(logging, level_str, None)
        if not isinstance(numeric, int):
            numeric = logging.WARNING
        return numeric

    def _setup_logger(self) -> logging.Logger:
        lg = logging.getLogger(self._name)
        lg.setLevel(self._resolve_log_level())
        lg.propagate = False
        if not lg.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
            lg.addHandler(handler)
        return lg

    @staticmethod
    def _get_rank() -> int:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_level(self, level: int) -> None:
        """Override the log level for the entire primus_turbo logger hierarchy."""
        self._logger.setLevel(level)

    def log(self, level: int, msg: str, *args, once: bool = False, rank: int | None = None, **kwargs) -> None:
        """Core logging method.

        Args:
            level: Standard :mod:`logging` level (e.g. ``logging.INFO``).
            msg:   Log message (may contain ``%``-style placeholders).
            once:  If ``True``, this *(level, msg)* pair is emitted at most
                   once per process.
            rank:  If set, only the process whose distributed rank matches
                   this value will emit the message.  ``None`` means all
                   ranks log.
        """
        if rank is not None and self._get_rank() != rank:
            return
        if once:
            key = (level, msg)
            if key in self._log_once_cache:
                return
            self._log_once_cache.add(key)
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, once: bool = False, rank: int | None = None, **kwargs) -> None:
        self.log(logging.DEBUG, msg, *args, once=once, rank=rank, **kwargs)

    def info(self, msg: str, *args, once: bool = False, rank: int | None = None, **kwargs) -> None:
        self.log(logging.INFO, msg, *args, once=once, rank=rank, **kwargs)

    def warning(self, msg: str, *args, once: bool = False, rank: int | None = None, **kwargs) -> None:
        self.log(logging.WARNING, msg, *args, once=once, rank=rank, **kwargs)

    def error(self, msg: str, *args, once: bool = False, rank: int | None = None, **kwargs) -> None:
        self.log(logging.ERROR, msg, *args, once=once, rank=rank, **kwargs)


logger = PrimusTurboLogger()
