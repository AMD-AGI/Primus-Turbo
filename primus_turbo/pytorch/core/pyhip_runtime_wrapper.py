###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ctypes
import platform
from dataclasses import dataclass
from functools import lru_cache
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
    hipMemcpyDeviceToDeviceNoCU = 1024


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


class hipDeviceAttributeEnum:
    hipDeviceAttributeVirtualMemoryManagementSupported = 89
    hipDeviceAttributeHandleTypeFabricSupported = 95


class hipMemAllocationTypeEnum:
    hipMemAllocationTypePinned = 0x1
    hipMemAllocationTypeUncached = 0x40000000


class hipMemAllocationHandleTypeEnum:
    hipMemHandleTypeNone = 0x0
    hipMemHandleTypePosixFileDescriptor = 0x1
    hipMemHandleTypeFabric = 0x8


class hipMemLocationTypeEnum:
    hipMemLocationTypeDevice = 0x1


class hipMemAccessFlagsEnum:
    hipMemAccessFlagsProtReadWrite = 0x3


class hipMemAllocationGranularityEnum:
    hipMemAllocationGranularityMinimum = 0x0
    hipMemAllocationGranularityRecommended = 0x1


class hipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class hipMemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class hipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", hipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", hipMemAllocFlags),
    ]


class hipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", hipMemLocation), ("flags", ctypes.c_int)]


class hipMemFabricHandle_t(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 64)]


# opaque ihipMemGenericAllocationHandle*
hipMemGenericAllocationHandle_t = ctypes.c_void_p


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
        Function(
            "hipExtMallocWithFlags", hipError_t, [ctypes.POINTER(buffer_type), ctypes.c_size_t, ctypes.c_uint]
        ),
        # hipError_t hipFree(void* ptr)
        Function("hipFree", hipError_t, [buffer_type]),
        # hipError_t hipMemset(void* ptr, int value, size_t sizeBytes)
        Function("hipMemset", hipError_t, [buffer_type, ctypes.c_int, ctypes.c_size_t]),
        # hipError_t hipIpcCloseMemHandle(void* dev_ptr)
        Function("hipIpcCloseMemHandle", hipError_t, [buffer_type]),
        # hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)
        Function(
            "hipDeviceGetAttribute", hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        ),
        # hipError_t hipMemGetAllocationGranularity(
        #   size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option)
        Function(
            "hipMemGetAllocationGranularity",
            hipError_t,
            [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(hipMemAllocationProp), ctypes.c_int],
        ),
        # hipError_t hipMemCreate(
        #   hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop,
        #   unsigned long long flags)
        Function(
            "hipMemCreate",
            hipError_t,
            [
                ctypes.POINTER(hipMemGenericAllocationHandle_t),
                ctypes.c_size_t,
                ctypes.POINTER(hipMemAllocationProp),
                ctypes.c_ulonglong,
            ],
        ),
        # hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)
        Function("hipMemRelease", hipError_t, [hipMemGenericAllocationHandle_t]),
        # hipError_t hipMemExportToShareableHandle(
        #   void* shareableHandle, hipMemGenericAllocationHandle_t handle,
        #   hipMemAllocationHandleType handleType, unsigned long long flags)
        Function(
            "hipMemExportToShareableHandle",
            hipError_t,
            [ctypes.c_void_p, hipMemGenericAllocationHandle_t, ctypes.c_int, ctypes.c_ulonglong],
        ),
        # hipError_t hipMemImportFromShareableHandle(
        #   hipMemGenericAllocationHandle_t* handle, void* osHandle, hipMemAllocationHandleType shHandleType)
        Function(
            "hipMemImportFromShareableHandle",
            hipError_t,
            [ctypes.POINTER(hipMemGenericAllocationHandle_t), ctypes.c_void_p, ctypes.c_int],
        ),
        # hipError_t hipMemAddressReserve(
        #   void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags)
        Function(
            "hipMemAddressReserve",
            hipError_t,
            [ctypes.POINTER(buffer_type), ctypes.c_size_t, ctypes.c_size_t, buffer_type, ctypes.c_ulonglong],
        ),
        # hipError_t hipMemAddressFree(void* devPtr, size_t size)
        Function("hipMemAddressFree", hipError_t, [buffer_type, ctypes.c_size_t]),
        # hipError_t hipMemMap(
        #   void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
        #   unsigned long long flags)
        Function(
            "hipMemMap",
            hipError_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ctypes.c_size_t,
                hipMemGenericAllocationHandle_t,
                ctypes.c_ulonglong,
            ],
        ),
        # hipError_t hipMemUnmap(void* ptr, size_t size)
        Function("hipMemUnmap", hipError_t, [buffer_type, ctypes.c_size_t]),
        # hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count)
        Function(
            "hipMemSetAccess",
            hipError_t,
            [buffer_type, ctypes.c_size_t, ctypes.POINTER(hipMemAccessDesc), ctypes.c_size_t],
        ),
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
                f"Failed to load HIP runtime library from {so_file}. "
                f"This is expected if you are not running on ROCm GPUs. "
                f"Otherwise, the HIP runtime library might not exist, be corrupted, "
                f"or does not support the current platform {platform.platform()}."
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

    def hipDeviceGetAttribute(self, attr: int, device: int) -> int:
        value = ctypes.c_int()
        self.HIP_CHECK(self._funcs["hipDeviceGetAttribute"](ctypes.byref(value), attr, device))
        return value.value

    def hipMemGetAllocationGranularity(self, prop: hipMemAllocationProp, option: int) -> int:
        granularity = ctypes.c_size_t()
        self.HIP_CHECK(
            self._funcs["hipMemGetAllocationGranularity"](
                ctypes.byref(granularity), ctypes.byref(prop), option
            )
        )
        return granularity.value

    def hipMemCreate(self, size_bytes: int, prop: hipMemAllocationProp) -> hipMemGenericAllocationHandle_t:
        handle = hipMemGenericAllocationHandle_t()
        self.HIP_CHECK(self._funcs["hipMemCreate"](ctypes.byref(handle), size_bytes, ctypes.byref(prop), 0))
        return handle

    def hipMemRelease(self, handle: hipMemGenericAllocationHandle_t) -> None:
        self.HIP_CHECK(self._funcs["hipMemRelease"](handle))

    def hipMemExportToFabricHandle(self, handle: hipMemGenericAllocationHandle_t) -> bytes:
        fabric = hipMemFabricHandle_t()
        self.HIP_CHECK(
            self._funcs["hipMemExportToShareableHandle"](
                ctypes.byref(fabric), handle, hipMemAllocationHandleTypeEnum.hipMemHandleTypeFabric, 0
            )
        )
        return bytes(fabric.data)

    def hipMemImportFromFabricHandle(self, data: bytes) -> hipMemGenericAllocationHandle_t:
        if len(data) != ctypes.sizeof(hipMemFabricHandle_t):
            raise ValueError(
                f"Expected {ctypes.sizeof(hipMemFabricHandle_t)} bytes for a fabric handle, got {len(data)}"
            )
        fabric = hipMemFabricHandle_t.from_buffer_copy(data)
        handle = hipMemGenericAllocationHandle_t()
        self.HIP_CHECK(
            self._funcs["hipMemImportFromShareableHandle"](
                ctypes.byref(handle),
                ctypes.byref(fabric),
                hipMemAllocationHandleTypeEnum.hipMemHandleTypeFabric,
            )
        )
        return handle

    def hipMemAddressReserve(self, size_bytes: int, alignment: int = 0) -> buffer_type:
        ptr = buffer_type()
        self.HIP_CHECK(self._funcs["hipMemAddressReserve"](ctypes.byref(ptr), size_bytes, alignment, None, 0))
        return ptr

    def hipMemAddressFree(self, ptr: int | buffer_type, size_bytes: int) -> None:
        self.HIP_CHECK(self._funcs["hipMemAddressFree"](_as_void_p(ptr), size_bytes))

    def hipMemMap(
        self, ptr: int | buffer_type, size_bytes: int, offset: int, handle: hipMemGenericAllocationHandle_t
    ) -> None:
        self.HIP_CHECK(self._funcs["hipMemMap"](_as_void_p(ptr), size_bytes, offset, handle, 0))

    def hipMemUnmap(self, ptr: int | buffer_type, size_bytes: int) -> None:
        self.HIP_CHECK(self._funcs["hipMemUnmap"](_as_void_p(ptr), size_bytes))

    def hipMemSetAccess(self, ptr: int | buffer_type, size_bytes: int, devices: list[int]) -> None:
        descs = (hipMemAccessDesc * len(devices))()
        for i, device in enumerate(devices):
            descs[i].location.type = hipMemLocationTypeEnum.hipMemLocationTypeDevice
            descs[i].location.id = device
            descs[i].flags = hipMemAccessFlagsEnum.hipMemAccessFlagsProtReadWrite
        self.HIP_CHECK(self._funcs["hipMemSetAccess"](_as_void_p(ptr), size_bytes, descs, len(devices)))

    @staticmethod
    def mem_handle_to_bytes(handle: hipIpcMemHandle_t) -> bytes:
        return bytes(handle.reserved)

    @staticmethod
    def mem_handle_from_bytes(data: bytes) -> hipIpcMemHandle_t:
        handle_size = ctypes.sizeof(hipIpcMemHandle_t)
        if len(data) != handle_size:
            raise ValueError(f"Expected {handle_size} bytes for hipIpcMemHandle_t, got {len(data)} bytes")
        handle = hipIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(handle), data, handle_size)
        return handle


@lru_cache(maxsize=1)
def get_hip_runtime_lib() -> HIPRuntimeLibrary:
    return HIPRuntimeLibrary()


__all__ = [
    "HIPRuntimeLibrary",
    "get_hip_runtime_lib",
    "hipMemcpyKindEnum",
    "hipIpcMemFlagsEnum",
    "hipIpcMemHandle_t",
    "hipDeviceAttributeEnum",
    "hipMemAllocationTypeEnum",
    "hipMemAllocationHandleTypeEnum",
    "hipMemLocationTypeEnum",
    "hipMemAccessFlagsEnum",
    "hipMemAllocationGranularityEnum",
    "hipMemAllocationProp",
    "hipMemGenericAllocationHandle_t",
    "hipStream_t",
    "buffer_type",
]
