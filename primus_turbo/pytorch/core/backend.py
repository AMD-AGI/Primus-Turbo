###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

__all__ = ["BackendType", "BackendConfig", "KernelBackend", "KernelDispatcher"]


class BackendType(Enum):
    CK = "CK"
    HIPBLASLT = "HIPBLASLT"
    AITER = "AITER"
    TRITON = "TRITON"
    AUTO = "AUTO"


class BackendConfig:
    """
    Backend configuration management.

    Priority (high to low):
    1. Code settings - set_gemm_backend(), etc.
    2. Environment variables - PRIMUS_TURBO_GEMM_BACKEND, etc.
    3. Auto-tune - PRIMUS_TURBO_AUTO_TUNE=1
    4. Code defaults
    """

    _gemm_backend: Optional[BackendType] = None
    _grouped_gemm_backend: Optional[BackendType] = None
    _auto_tune: Optional[bool] = None

    @classmethod
    def set_gemm_backend(cls, backend: Optional[BackendType]) -> None:
        """Set the GEMM backend in code."""
        cls._gemm_backend = backend

    @classmethod
    def set_grouped_gemm_backend(cls, backend: Optional[BackendType]) -> None:
        """Set the Grouped GEMM backend in code."""
        cls._grouped_gemm_backend = backend

    @classmethod
    def set_auto_tune(cls, enabled: Optional[bool]) -> None:
        """Set whether auto-tune is enabled in code."""
        cls._auto_tune = enabled

    @classmethod
    def get_gemm_backend(cls) -> Optional[BackendType]:
        """Get the GEMM backend configuration. Returns None if not set."""
        if cls._gemm_backend is not None:
            return cls._gemm_backend
        backend = os.environ.get("PRIMUS_TURBO_GEMM_BACKEND")
        if backend:
            return BackendType(backend.upper())
        return None

    @classmethod
    def get_grouped_gemm_backend(cls) -> Optional[BackendType]:
        """Get the Grouped GEMM backend configuration. Returns None if not set."""
        if cls._grouped_gemm_backend is not None:
            return cls._grouped_gemm_backend
        backend = os.environ.get("PRIMUS_TURBO_GROUPED_GEMM_BACKEND")
        if backend:
            return BackendType(backend.upper())
        return None

    @classmethod
    def auto_tune_enabled(cls) -> bool:
        """Check if auto-tune is enabled."""
        if cls._auto_tune is not None:
            return cls._auto_tune
        return os.environ.get("PRIMUS_TURBO_AUTO_TUNE", "0") == "1"


# Kernel Backend
class KernelBackend(ABC):
    @staticmethod
    @abstractmethod
    def can_handle(**kwargs) -> bool:
        raise NotImplementedError("can_handle is not implemented")

    @staticmethod
    @abstractmethod
    def execute(**kwargs):
        raise NotImplementedError("execute is not implemented")


# Responsible for auto-tune scheduling
class KernelDispatcher(ABC):
    pass
