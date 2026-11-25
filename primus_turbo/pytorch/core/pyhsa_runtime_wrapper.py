###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ctypes binding of the HSA runtime calls that drive SDMA copies directly.

HIP routes ``hipMemcpyAsync`` to memory imported through a fabric handle down a slow path, and
offers no way to pick the copy engine or chain a remote flag write after a copy; the HSA calls do.
``hsa_agent_t`` and ``hsa_signal_t`` are one-``uint64_t`` structs, passed here as plain ``uint64``
(the same registers under the SysV x86-64 ABI).
"""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import torch

hsa_status_t = ctypes.c_int
hsa_agent_t = ctypes.c_uint64
hsa_signal_t = ctypes.c_uint64
hsa_signal_value_t = ctypes.c_int64


class hsaEnum:
    HSA_STATUS_SUCCESS = 0x0
    HSA_STATUS_INFO_BREAK = 0x1
    HSA_DEVICE_TYPE_CPU = 0
    HSA_DEVICE_TYPE_GPU = 1
    HSA_AGENT_INFO_DEVICE = 17
    HSA_AMD_AGENT_INFO_BDFID = 0xA006
    HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F
    HSA_SIGNAL_CONDITION_EQ = 0
    HSA_SIGNAL_CONDITION_LT = 2
    HSA_WAIT_STATE_BLOCKED = 0
    HSA_WAIT_STATE_ACTIVE = 1
    HSA_FENCE_SCOPE_NONE = 0
    HSA_FENCE_SCOPE_AGENT = 1
    HSA_FENCE_SCOPE_SYSTEM = 2
    HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1


class hsaCopyEnum:
    HSA_AMD_MEMORY_COPY_OP_VERSION = 1
    HSA_AMD_MEMORY_COPY_OP_LINEAR = 0
    HSA_AMD_MEMORY_COPY_WAIT_ALWAYS = 0
    HSA_AMD_MEMORY_COPY_WAIT_GE = 5
    HSA_AMD_MEMORY_COPY_SIGNAL_NONE = 0
    HSA_AMD_MEMORY_COPY_SIGNAL_WRITE = 1


class hsa_amd_memory_copy_op_t(ctypes.Structure):
    """``hsa_amd_memory_copy_op_t`` (ROCr batch copy), unions flattened to their 8-byte scalar members."""

    _fields_ = [
        ("version", ctypes.c_uint16),
        ("type", ctypes.c_uint16),
        ("num_entries", ctypes.c_uint16),
        ("traffic_class", ctypes.c_uint16),
        ("completion_signal", hsa_signal_t),
        ("src", ctypes.c_void_p),
        ("src_agent", hsa_agent_t),
        ("dst_agent", hsa_agent_t),
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("unused_size", ctypes.c_size_t),
        ("wait_function", ctypes.c_uint16),
        ("wait_scope", ctypes.c_uint16),
        ("wait_reserved", ctypes.c_uint32),
        ("wait_addr", ctypes.c_void_p),
        ("wait_value", ctypes.c_uint64),
        ("wait_mask", ctypes.c_uint64),
        ("signal_operation", ctypes.c_uint16),
        ("signal_scope", ctypes.c_uint16),
        ("signal_reserved", ctypes.c_uint32),
        ("signal_addr", ctypes.c_void_p),
        ("signal_data", ctypes.c_uint64),
        ("reserved1", ctypes.c_uint64),
    ]


assert ctypes.sizeof(hsa_amd_memory_copy_op_t) == 128
assert hsa_amd_memory_copy_op_t.wait_function.offset == 64
assert hsa_amd_memory_copy_op_t.signal_operation.offset == 96

_AGENT_CALLBACK = ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.c_void_p)


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


def find_hsa_runtime_library() -> str:
    """The libhsa-runtime64 HIP already loaded (wheels ship it outside the loader path), else its soname."""
    if torch.version.hip is None:
        raise ValueError("HSA runtime wrapper only supports ROCm backend.")
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1]
                if "/libhsa-runtime64.so" in path:
                    return path
    except OSError:
        pass
    return "libhsa-runtime64.so.1"


class HSARuntimeLibrary:
    exported_functions = [
        Function("hsa_init", hsa_status_t, []),
        Function("hsa_status_string", hsa_status_t, [hsa_status_t, ctypes.POINTER(ctypes.c_char_p)]),
        Function("hsa_iterate_agents", hsa_status_t, [_AGENT_CALLBACK, ctypes.c_void_p]),
        Function("hsa_agent_get_info", hsa_status_t, [hsa_agent_t, ctypes.c_int, ctypes.c_void_p]),
        # hsa_amd_signal_create(value, num_consumers, consumers, attributes, signal*)
        Function(
            "hsa_amd_signal_create",
            hsa_status_t,
            [
                hsa_signal_value_t,
                ctypes.c_uint32,
                ctypes.POINTER(hsa_agent_t),
                ctypes.c_uint64,
                ctypes.POINTER(hsa_signal_t),
            ],
        ),
        Function("hsa_signal_destroy", hsa_status_t, [hsa_signal_t]),
        Function("hsa_signal_store_relaxed", None, [hsa_signal_t, hsa_signal_value_t]),
        Function("hsa_signal_load_scacquire", hsa_signal_value_t, [hsa_signal_t]),
        Function(
            "hsa_amd_signal_value_pointer",
            hsa_status_t,
            [hsa_signal_t, ctypes.POINTER(ctypes.c_void_p)],
        ),
        # hsa_amd_memory_copy_engine_status(dst_agent, src_agent, uint32_t* engine_ids_mask)
        Function(
            "hsa_amd_memory_copy_engine_status",
            hsa_status_t,
            [hsa_agent_t, hsa_agent_t, ctypes.POINTER(ctypes.c_uint32)],
        ),
        # hsa_amd_memory_async_copy_on_engine(dst, dst_agent, src, src_agent, size,
        #   num_dep_signals, dep_signals, completion_signal, engine_id, force_copy_on_sdma)
        Function(
            "hsa_amd_memory_async_copy_on_engine",
            hsa_status_t,
            [
                ctypes.c_void_p,
                hsa_agent_t,
                ctypes.c_void_p,
                hsa_agent_t,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(hsa_signal_t),
                hsa_signal_t,
                ctypes.c_int,
                ctypes.c_bool,
            ],
        ),
        # hsa_amd_memory_async_batch_copy(copy_ops, num_copy_ops, num_dep_signals, dep_signals)
        Function(
            "hsa_amd_memory_async_batch_copy",
            hsa_status_t,
            [
                ctypes.POINTER(hsa_amd_memory_copy_op_t),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(hsa_signal_t),
            ],
        ),
    ]

    def __init__(self, so_file: str | None = None):
        so_file = so_file or find_hsa_runtime_library()
        self.lib = ctypes.CDLL(so_file)
        self._funcs: dict[str, Any] = {}
        missing = []
        for func in HSARuntimeLibrary.exported_functions:
            try:
                f = getattr(self.lib, func.name)
            except AttributeError:
                missing.append(func.name)
                continue
            f.restype = func.restype
            f.argtypes = func.argtypes
            self._funcs[func.name] = f
        if missing:
            raise RuntimeError(
                f"{so_file} lacks {missing}; SDMA collectives need a ROCm runtime with the batch copy API"
            )
        # HIP has initialized HSA already; this only takes a reference.
        self.HSA_CHECK(self._funcs["hsa_init"]())

    def HSA_CHECK(self, status: int) -> None:
        if status not in (hsaEnum.HSA_STATUS_SUCCESS, hsaEnum.HSA_STATUS_INFO_BREAK):
            msg = ctypes.c_char_p()
            self._funcs["hsa_status_string"](status, ctypes.byref(msg))
            text = msg.value.decode() if msg.value else f"status 0x{status:x}"
            raise RuntimeError(f"HSA runtime error: {text}")

    def agents(self) -> list[tuple[int, int]]:
        """(agent handle, hsa_device_type_t) of every agent."""
        found: list[tuple[int, int]] = []

        def cb(agent, _data):
            kind = ctypes.c_uint32()
            self._funcs["hsa_agent_get_info"](agent, hsaEnum.HSA_AGENT_INFO_DEVICE, ctypes.byref(kind))
            found.append((agent, kind.value))
            return hsaEnum.HSA_STATUS_SUCCESS

        self.HSA_CHECK(self._funcs["hsa_iterate_agents"](_AGENT_CALLBACK(cb), None))
        return found

    def agent_info_u32(self, agent: int, attribute: int) -> int:
        value = ctypes.c_uint32()
        self.HSA_CHECK(self._funcs["hsa_agent_get_info"](agent, attribute, ctypes.byref(value)))
        return value.value

    def gpu_agent(self, device: int) -> int:
        """The HSA agent of torch device ``device``, matched by PCI location (HSA numbers every GPU)."""
        props = torch.cuda.get_device_properties(device)
        want = (props.pci_domain_id, props.pci_bus_id, props.pci_device_id)
        for agent, kind in self.agents():
            if kind != hsaEnum.HSA_DEVICE_TYPE_GPU:
                continue
            bdf = self.agent_info_u32(agent, hsaEnum.HSA_AMD_AGENT_INFO_BDFID)
            domain = self.agent_info_u32(agent, hsaEnum.HSA_AMD_AGENT_INFO_DOMAIN)
            if (domain, bdf >> 8, (bdf >> 3) & 0x1F) == want:
                return agent
        raise RuntimeError(f"no HSA GPU agent at PCI {want} for device {device}")

    def cpu_agent(self) -> int:
        for agent, kind in self.agents():
            if kind == hsaEnum.HSA_DEVICE_TYPE_CPU:
                return agent
        raise RuntimeError("no HSA CPU agent")

    def signal_create(self, value: int, gpu_only: bool = True) -> int:
        signal = hsa_signal_t()
        attributes = hsaEnum.HSA_AMD_SIGNAL_AMD_GPU_ONLY if gpu_only else 0
        self.HSA_CHECK(self._funcs["hsa_amd_signal_create"](value, 0, None, attributes, ctypes.byref(signal)))
        return signal.value

    def signal_destroy(self, signal: int) -> None:
        self.HSA_CHECK(self._funcs["hsa_signal_destroy"](signal))

    def signal_store(self, signal: int, value: int) -> None:
        self._funcs["hsa_signal_store_relaxed"](signal, value)

    def signal_load(self, signal: int) -> int:
        return self._funcs["hsa_signal_load_scacquire"](signal)

    def signal_value_pointer(self, signal: int) -> int:
        """Address of the signal's 64-bit value, for a kernel to read or write it (GPU-only signals)."""
        ptr = ctypes.c_void_p()
        self.HSA_CHECK(self._funcs["hsa_amd_signal_value_pointer"](signal, ctypes.byref(ptr)))
        return ptr.value

    def copy_engine_mask(self, dst_agent: int, src_agent: int) -> int:
        mask = ctypes.c_uint32()
        self.HSA_CHECK(
            self._funcs["hsa_amd_memory_copy_engine_status"](dst_agent, src_agent, ctypes.byref(mask))
        )
        return mask.value

    def copy_on_engine(
        self,
        dst: int,
        dst_agent: int,
        src: int,
        src_agent: int,
        size: int,
        deps: Sequence[int],
        completion: int,
        engine: int,
    ) -> None:
        """SDMA copy on engine index ``engine`` once every signal in ``deps`` reads 0; ``completion`` -= 1 after."""
        dep_arr = (hsa_signal_t * len(deps))(*deps) if deps else None
        self.HSA_CHECK(
            self._funcs["hsa_amd_memory_async_copy_on_engine"](
                dst, dst_agent, src, src_agent, size, len(deps), dep_arr, completion, 1 << engine, True
            )
        )

    def batch_copy(self, ops: Sequence[hsa_amd_memory_copy_op_t], deps: Sequence[int]) -> None:
        """Submit ``ops`` (sharing one completion signal, preset to ``len(ops)``) after ``deps`` read 0."""
        op_arr = (hsa_amd_memory_copy_op_t * len(ops))(*ops)
        dep_arr = (hsa_signal_t * len(deps))(*deps) if deps else None
        self.HSA_CHECK(self._funcs["hsa_amd_memory_async_batch_copy"](op_arr, len(ops), len(deps), dep_arr))


@lru_cache(maxsize=1)
def get_hsa_runtime_lib() -> HSARuntimeLibrary:
    return HSARuntimeLibrary()


__all__ = [
    "HSARuntimeLibrary",
    "get_hsa_runtime_lib",
    "hsaEnum",
    "hsaCopyEnum",
    "hsa_amd_memory_copy_op_t",
]
