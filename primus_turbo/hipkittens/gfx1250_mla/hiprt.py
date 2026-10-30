###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch a prebuilt gfx1250 code object on a torch stream through the HIP runtime torch loaded.

    kernels = load_kernels(names)                     # {name: Kernel}, sha256-checked bytes (no GPU work)
    kernels[name].call(grid, block, values, stream)   # pack the kernel arguments + hipModuleLaunchKernel

The libamdhip64 that torch mapped is the only runtime (found in /proc/self/maps, opened with
RTLD_NOLOAD), so the launches share torch's device context and streams. Every hip* return code is
checked; a launch goes to the torch stream object it is given (raw ints / None raise TypeError, so
a hard-coded stream 0 cannot slip in); nothing here synchronizes a device or a stream.
"""

import ctypes
import os
import struct
import threading

from . import hsaco

__all__ = [
    "HipError",
    "Kernel",
    "Module",
    "check_device",
    "load_kernels",
    "pack_args",
    "runtime_unavailable_reason",
]


class HipError(RuntimeError):
    pass


_LIB = None
_LOCK = threading.RLock()  # re-entrant: Module.handle() may be the first caller of _lib()
_P_BUF = ctypes.c_void_p(1)  # HIP_LAUNCH_PARAM_BUFFER_POINTER
_P_SIZE = ctypes.c_void_p(2)  # HIP_LAUNCH_PARAM_BUFFER_SIZE
_P_END = ctypes.c_void_p(3)  # HIP_LAUNCH_PARAM_END


def _mapped_libhip():
    paths = set()
    with open("/proc/self/maps") as f:
        for ln in f:
            i = ln.find("/")
            if i >= 0 and "libamdhip64.so" in ln[i:]:
                paths.add(ln[i:].strip())
    return sorted(paths), {os.path.realpath(p) for p in paths}


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        import torch  # noqa: F401  (maps torch's libamdhip64)

        paths, real = _mapped_libhip()
        if not paths:
            raise HipError("libamdhip64 is not mapped in this process (import a ROCm build of torch first)")
        if len(real) != 1:
            raise HipError(f"more than one libamdhip64 mapped: {sorted(real)} (two HIP runtimes)")
        try:
            lib = ctypes.CDLL(paths[0], mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError as e:
            raise HipError(f"cannot open the mapped {paths[0]} with RTLD_NOLOAD: {e}") from e
        c_p, c_u, c_i = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(c_p), c_p]
        lib.hipModuleLoadData.restype = c_i
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(c_p), c_p, ctypes.c_char_p]
        lib.hipModuleGetFunction.restype = c_i
        lib.hipModuleLaunchKernel.argtypes = [
            c_p, c_u, c_u, c_u, c_u, c_u, c_u, c_u, c_p, ctypes.POINTER(c_p), ctypes.POINTER(c_p)
        ]  # fmt: skip
        lib.hipModuleLaunchKernel.restype = c_i
        lib.hipGetDevice.argtypes = [ctypes.POINTER(c_i)]
        lib.hipGetDevice.restype = c_i
        lib.hipGetErrorName.argtypes = [c_i]
        lib.hipGetErrorName.restype = ctypes.c_char_p
        lib.hipGetErrorString.argtypes = [c_i]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        _LIB = lib
        return lib


_RUNTIME_REASON = []


def runtime_unavailable_reason():
    """None when the HIP runtime torch loaded can be used here, else why not (checked once per process)."""
    if not _RUNTIME_REASON:
        try:
            _lib()
            _RUNTIME_REASON.append(None)
        except HipError as e:
            _RUNTIME_REASON.append(str(e))
    return _RUNTIME_REASON[0]


def _check(rc, what):
    if rc != 0:
        try:
            name = _LIB.hipGetErrorName(rc).decode()
            msg = _LIB.hipGetErrorString(rc).decode()
        except Exception:  # noqa: BLE001
            name, msg = "?", "?"
        raise HipError(f"{what} failed: {rc} {name}: {msg}")


def _device():
    d = ctypes.c_int(-1)
    _check(_lib().hipGetDevice(ctypes.byref(d)), "hipGetDevice")
    return d.value


def _dim3(x, what):
    t = (x, 1, 1) if isinstance(x, int) else tuple(int(v) for v in x)
    if not 1 <= len(t) <= 3:
        raise ValueError(f"{what} {x!r}: 1 to 3 dims")
    t = t + (1,) * (3 - len(t))
    if any(v < 1 or v >= (1 << 32) for v in t):
        raise ValueError(f"{what} {x!r}: every dim in [1, 2^32)")
    return t


class Module:
    """Code object bytes whose sha256 was checked; loaded into a device's context at first use."""

    def __init__(self, path, sha256):
        import hashlib

        self.path = str(path)
        with open(path, "rb") as f:
            data = f.read()
        got = hashlib.sha256(data).hexdigest()
        if got != str(sha256).lower():
            raise HipError(f"{path}: sha256 {got} != expected {sha256} (stale or foreign code object)")
        self._data = ctypes.create_string_buffer(data, len(data))
        self._mods = {}  # device -> hipModule_t

    def handle(self, dev):
        h = self._mods.get(dev)
        if h is None:
            with _LOCK:
                h = self._mods.get(dev)
                if h is None:
                    m = ctypes.c_void_p()
                    _check(
                        _lib().hipModuleLoadData(ctypes.byref(m), ctypes.cast(self._data, ctypes.c_void_p)),
                        f"hipModuleLoadData({self.path})",
                    )
                    h = self._mods[dev] = m
        return h


class Kernel:
    """One kernel of a Module with its manifest entry; launch() is the only call that touches the GPU."""

    def __init__(self, module, name, meta):
        self.module, self.name, self.meta = module, name, meta
        self._fns = {}

    def _fn(self, dev):
        f = self._fns.get(dev)
        if f is None:
            f = ctypes.c_void_p()
            _check(
                _lib().hipModuleGetFunction(ctypes.byref(f), self.module.handle(dev), self.name.encode()),
                f"hipModuleGetFunction({self.name})",
            )
            self._fns[dev] = f
        return f

    def launch(self, grid, block, kernarg_bytes, stream):
        handle = _stream_handle(stream)
        gx, gy, gz = _dim3(grid, "grid")
        bx, by, bz = _dim3(block, "block")
        if bx * by * bz != self.meta["block"]:
            raise HipError(f"{self.name}: block {block} != the compiled workgroup size {self.meta['block']}")
        n, size = len(kernarg_bytes), self.meta["kernarg_size"]
        if n > size:
            raise HipError(f"{self.name}: {n} kernarg bytes > kernarg_segment_size {size}")
        kernarg_bytes = bytes(kernarg_bytes) + bytes(size - n)  # tail padding of the segment
        dev = _stream_device(stream)
        cur = _device()
        if dev is not None and dev != cur:
            raise HipError(f"{self.name}: stream on device {dev} but the current device is {cur}")
        f = self._fn(cur)
        buf = ctypes.create_string_buffer(kernarg_bytes, max(size, 1))
        nbytes = ctypes.c_size_t(size)
        extra = (ctypes.c_void_p * 5)(
            _P_BUF,
            ctypes.cast(buf, ctypes.c_void_p),
            _P_SIZE,
            ctypes.cast(ctypes.pointer(nbytes), ctypes.c_void_p),
            _P_END,
        )
        _check(
            _LIB.hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, 0, ctypes.c_void_p(handle), None, extra),
            f"hipModuleLaunchKernel({self.name})",
        )

    def call(self, grid, block, values, stream):
        self.launch(grid, block, pack_args(self.meta["args"], values), stream)


def _stream_handle(stream):
    if stream is None or isinstance(stream, (int, bool)):
        raise TypeError(
            "hiprt: pass a torch stream object (torch.cuda.current_stream()), "
            f"not {stream!r}: a hard-coded stream must never reach a launch"
        )
    h = getattr(stream, "cuda_stream", None)
    if h is None:
        raise TypeError(f"hiprt: {type(stream).__name__} has no .cuda_stream")
    return int(h)


def _stream_device(stream):
    idx = getattr(getattr(stream, "device", None), "index", None)
    return idx if isinstance(idx, int) else None


# ---------------------------------------------------------------- kernarg packing

_FMT = {"ptr": "Q", "i32": "i", "u32": "I", "i64": "q", "u64": "Q", "f32": "f", "f64": "d"}
_PACKERS = {}


def _packer(args_meta):
    p = _PACKERS.get(id(args_meta))
    if p is not None and p[0] is args_meta:
        return p
    fmt, pos, kinds, names = "<", 0, [], []
    for a in args_meta:
        off, size, kind = int(a["offset"]), int(a["size"]), a["kind"]
        if off < pos:
            raise ValueError(f"kernarg {a['name']}: offset {off} overlaps the previous argument")
        c = _FMT.get(kind)
        if c is None or struct.calcsize("<" + c) != size:
            raise ValueError(f"kernarg {a['name']}: kind {kind} size {size}")
        fmt += "x" * (off - pos) + c
        pos = off + size
        kinds.append(kind)
        names.append(a["name"])
    p = (args_meta, struct.Struct(fmt), tuple(kinds), tuple(names))
    _PACKERS[id(args_meta)] = p
    return p


def pack_args(args_meta, values):
    """Kernarg bytes for a manifest ``args`` list; ``values`` in the same order."""
    _, st, kinds, names = _packer(args_meta)
    if len(values) != len(kinds):
        raise ValueError(f"{len(values)} kernarg values for {len(kinds)} arguments {list(names)}")
    out = []
    for kind, v, n in zip(kinds, values, names):
        if kind == "ptr":
            p = v.data_ptr() if hasattr(v, "data_ptr") else v
            if not isinstance(p, int) or isinstance(p, bool) or p == 0:
                raise ValueError(f"kernarg {n}: pointer must be a tensor or a non-zero int, got {v!r}")
            out.append(p)
        elif hasattr(v, "data_ptr"):
            raise TypeError(f"kernarg {n} ({kind}): got a tensor (scalars only: no hidden .item() syncs)")
        elif kind in ("f32", "f64"):
            out.append(float(v))
        else:
            if isinstance(v, bool) or int(v) != v:
                raise TypeError(f"kernarg {n} ({kind}): integer required, got {v!r}")
            out.append(int(v))
    return st.pack(*out)


def load_kernels(names, bin_dir=hsaco.BIN_DIR):
    """{name: Kernel} for ``names``; raises HipError unless every code object is the one the manifest pins. Reads
    files only (no GPU work until the first launch)."""
    reason = hsaco.unavailable_reason(names, bin_dir)
    if reason is not None:
        raise HipError(f"gfx1250 MLA code objects unavailable: {reason}")
    meta = hsaco.load_manifest()["kernels"]
    return {
        n: Kernel(Module(hsaco.binary_path(n, bin_dir), meta[n]["hsaco_sha256"]), n, meta[n]) for n in names
    }


_DEVICES_CHECKED = set()


def check_device(device):
    """Raise unless ``device`` is a gfx1250 GPU: the code objects are gfx1250 wave32 WMMA code."""
    if device not in _DEVICES_CHECKED:
        import torch

        if device.type != "cuda":
            raise HipError(f"the gfx1250 MLA kernels run on a GPU, got {device}")
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if "gfx1250" not in arch:
            raise HipError(f"the gfx1250 MLA kernels cannot run on {arch}")
        _DEVICES_CHECKED.add(device)
