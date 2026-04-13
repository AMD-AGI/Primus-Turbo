import ctypes
import platform
from dataclasses import dataclass
from typing import Any

import torch

hipError_t = ctypes.c_int
hipStream_t = ctypes.c_void_p
hipMemcpyKind_t = ctypes.c_int
buffer_type = ctypes.c_void_p


class hipMemcpyKindEnum:
    hipMemcpyHostToHost = 0
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3
    hipMemcpyDefault = 4

class hipExtMallocFlagsEnum:
    hipDeviceMallocDefault = 0x0
    hipDeviceMallocFinegrained = 0x1
    hipMallocSignalMemory = 0x2
    hipDeviceMallocUncached = 0x3
    hipDeviceMallocContiguous = 0x4


class hipIpcMemFlagsEnum:
    hipIpcMemLazyEnablePeerAccess = 1


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


def find_hip_runtime_library() -> str:
    if torch.version.hip is None:
        raise ValueError("HIP runtime wrapper only supports ROCm backend.")
    return "libamdhip64.so"


def _as_void_p(ptr: int | buffer_type | None) -> buffer_type:
    if ptr is None:
        return buffer_type()
    if isinstance(ptr, ctypes.c_void_p):
        return ptr
    return buffer_type(ptr)


class HIPRuntimeLibrary:
    exported_functions = [
        # const char* hipGetErrorString(hipError_t hip_error)
        Function("hipGetErrorString", ctypes.c_char_p, [hipError_t]),
        # hipError_t hipMemcpyAsync(
        #   void* dst, const void* src, size_t sizeBytes,
        #   hipMemcpyKind kind, hipStream_t stream)
        Function(
            "hipMemcpyAsync",
            hipError_t,
            [buffer_type, buffer_type, ctypes.c_size_t, hipMemcpyKind_t, hipStream_t],
        ),
        # hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* dev_ptr)
        Function(
            "hipIpcGetMemHandle",
            hipError_t,
            [ctypes.POINTER(hipIpcMemHandle_t), buffer_type],
        ),
        # hipError_t hipIpcOpenMemHandle(
        #   void** dev_ptr, hipIpcMemHandle_t handle, unsigned int flags)
        Function(
            "hipIpcOpenMemHandle",
            hipError_t,
            [ctypes.POINTER(buffer_type), hipIpcMemHandle_t, ctypes.c_uint],
        ),
        # hipError_t hipMalloc(void** ptr, size_t size)
        Function("hipMalloc", hipError_t, [ctypes.POINTER(buffer_type), ctypes.c_size_t]),
        # hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags)
        Function("hipExtMallocWithFlags", hipError_t, [ctypes.POINTER(buffer_type), ctypes.c_size_t, ctypes.c_uint]),
        # hipError_t hipFree(void* ptr)
        Function("hipFree", hipError_t, [buffer_type]),
        # hipError_t hipMemset(void* ptr, int value, size_t sizeBytes)
        Function("hipMemset", hipError_t, [buffer_type, ctypes.c_int, ctypes.c_size_t]),
        # hipError_t hipIpcCloseMemHandle(void* dev_ptr)
        Function("hipIpcCloseMemHandle", hipError_t, [buffer_type]),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        so_file = so_file or find_hip_runtime_library()

        try:
            if so_file not in HIPRuntimeLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                HIPRuntimeLibrary.path_to_library_cache[so_file] = lib
            self.lib = HIPRuntimeLibrary.path_to_library_cache[so_file]
        except Exception as e:
            print(
                "Failed to load HIP runtime library from %s. "
                "This is expected if you are not running on ROCm GPUs. "
                "Otherwise, the HIP runtime library might not exist, be corrupted, "
                "or does not support the current platform %s.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in HIPRuntimeLibrary.path_to_dict_mapping:
            funcs: dict[str, Any] = {}
            for func in HIPRuntimeLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                funcs[func.name] = f
            HIPRuntimeLibrary.path_to_dict_mapping[so_file] = funcs
        self._funcs = HIPRuntimeLibrary.path_to_dict_mapping[so_file]

    def hipGetErrorString(self, result: hipError_t) -> str:
        error = self._funcs["hipGetErrorString"](result)
        if error is None:
            return f"Unknown HIP runtime error code: {result}"
        return error.decode("utf-8")

    def HIP_CHECK(self, result: hipError_t) -> None:
        if result != 0:
            error_str = self.hipGetErrorString(result)
            raise RuntimeError(f"HIP runtime error: {error_str}")

    def hipMemcpyAsync(
        self,
        dst: int | buffer_type,
        src: int | buffer_type,
        size_bytes: int,
        kind: int,
        stream: int | hipStream_t | None = None,
    ) -> None:
        self.HIP_CHECK(
            self._funcs["hipMemcpyAsync"](
                _as_void_p(dst),
                _as_void_p(src),
                size_bytes,
                kind,
                _as_void_p(stream),
            )
        )

    def hipIpcGetMemHandle(self, ptr: int | buffer_type) -> hipIpcMemHandle_t:
        handle = hipIpcMemHandle_t()
        self.HIP_CHECK(self._funcs["hipIpcGetMemHandle"](ctypes.byref(handle), _as_void_p(ptr)))
        return handle

    def hipIpcOpenMemHandle(
        self,
        handle: hipIpcMemHandle_t | bytes,
        flags: int = hipIpcMemFlagsEnum.hipIpcMemLazyEnablePeerAccess,
    ) -> buffer_type:
        if isinstance(handle, bytes):
            handle = self.mem_handle_from_bytes(handle)
        ptr = buffer_type()
        self.HIP_CHECK(self._funcs["hipIpcOpenMemHandle"](ctypes.byref(ptr), handle, flags))
        return ptr

    def hipMalloc(self, size_bytes: int) -> buffer_type:
        ptr = buffer_type()
        self.HIP_CHECK(self._funcs["hipMalloc"](ctypes.byref(ptr), size_bytes))
        return ptr
    
    def hipMallocUncached(self, size_bytes: int) -> buffer_type:
        return self.hipExtMallocWithFlags(size_bytes, hipExtMallocFlagsEnum.hipDeviceMallocUncached)
    
    def hipExtMallocWithFlags(self, size_bytes: int, flags: hipExtMallocFlagsEnum) -> buffer_type:
        ptr = buffer_type()
        self.HIP_CHECK(self._funcs["hipExtMallocWithFlags"](ctypes.byref(ptr), size_bytes, flags))
        return ptr

    def hipFree(self, ptr: int | buffer_type) -> None:
        self.HIP_CHECK(self._funcs["hipFree"](_as_void_p(ptr)))

    def hipMemset(self, ptr: int | buffer_type, value: int, size_bytes: int) -> None:
        self.HIP_CHECK(self._funcs["hipMemset"](_as_void_p(ptr), value, size_bytes))

    def hipIpcCloseMemHandle(self, ptr: int | buffer_type) -> None:
        self.HIP_CHECK(self._funcs["hipIpcCloseMemHandle"](_as_void_p(ptr)))

    @staticmethod
    def mem_handle_to_bytes(handle: hipIpcMemHandle_t) -> bytes:
        return bytes(handle.reserved)

    @staticmethod
    def mem_handle_from_bytes(data: bytes) -> hipIpcMemHandle_t:
        handle_size = ctypes.sizeof(hipIpcMemHandle_t)
        if len(data) != handle_size:
            raise ValueError(
                f"Expected {handle_size} bytes for hipIpcMemHandle_t, got {len(data)} bytes"
            )
        handle = hipIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(handle), data, handle_size)
        return handle




__all__ = [
    "HIPRuntimeLibrary",
    "hipMemcpyKindEnum",
    "hipIpcMemFlagsEnum",
    "hipIpcMemHandle_t",
    "hipStream_t",
    "buffer_type",
]
