###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch a prebuilt gfx1250 code object on a torch stream through the HIP runtime torch loaded.
(The ladder's copy of primus_turbo/hipkittens/gfx1250_attn/hiprt.py at cade7db6 without the
manifest loader load_kernels.)

    kernels = load_hsaco(path, specs)                 # any code object, no manifest (lab / tests):
                                                      # specs = {name: {"block": n, "args": [(name, kind)]}}
                                                      # checked against the code object's own metadata

The libamdhip64 that torch mapped is the only runtime (found in /proc/self/maps, opened with
RTLD_NOLOAD), so the launches share torch's device context and streams. Every hip* return code is
checked; a launch goes to the torch stream object it is given (raw ints / None raise TypeError, so
a hard-coded stream 0 cannot slip in); nothing here synchronizes a device or a stream.
"""

import ctypes
import os
import struct
import threading

from . import codeobj

__all__ = ["HipError", "Kernel", "Module", "load_hsaco", "kernel_meta", "pack_args"]


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
    """Code object bytes whose sha256 was checked; loaded into a device's context at first use.

    sha256=None skips the pin (load_hsaco: lab and test code objects, whose kernels are checked
    against the code object's own metadata instead); production kernels always come pinned from
    the manifest (load_kernels)."""

    def __init__(self, path, sha256=None):
        import hashlib

        self.path = str(path)
        with open(path, "rb") as f:
            data = f.read()
        got = hashlib.sha256(data).hexdigest()
        if sha256 is not None and got != str(sha256).lower():
            raise HipError(f"{path}: sha256 {got} != expected {sha256} (stale or foreign code object)")
        self.sha256 = got
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


# ---------------------------------------------------------------- code objects without a manifest

_KIND_SIZE = {k: struct.calcsize("<" + c) for k, c in _FMT.items()}


def kernel_meta(co_kernel, name, spec):
    """Launch metadata ({"block", "kernarg_size", "args"}) of one kernel from the code object's own
    table (codeobj.kernels) and a spec {"block": n, "args": [(arg name, kind), ...]}; raises HipError
    unless the arguments match in count and size, the kernarg segment holds no hidden arguments and
    the block fits the compiled workgroup size."""
    args, want = co_kernel["args"], list(spec["args"])
    if len(args) != len(want):
        raise HipError(f"{name}: code object has {len(args)} kernel arguments, spec lists {len(want)}")
    out, end = [], 0
    for a, (an, kind) in zip(args, want):
        size = _KIND_SIZE.get(kind)
        if size is None:
            raise HipError(f"{name}.{an}: unknown kind {kind!r}")
        if a["size"] != size:
            raise HipError(f"{name}.{an}: code object size {a['size']} != {kind} ({size})")
        if (kind == "ptr") != (a["value_kind"] == "global_buffer"):
            raise HipError(f"{name}.{an}: value_kind {a['value_kind']} vs spec kind {kind}")
        out.append({"name": an, "kind": kind, "offset": a["offset"], "size": a["size"]})
        end = max(end, a["offset"] + a["size"])
    if co_kernel["kernarg_size"] != end:
        raise HipError(
            f"{name}: kernarg_segment_size {co_kernel['kernarg_size']} != {end} of the explicit arguments "
            "(hidden kernel arguments are not supported)"
        )
    block = int(spec["block"])
    if not 1 <= block <= co_kernel["max_block"]:
        raise HipError(
            f"{name}: block {block} outside [1, {co_kernel['max_block']}] (max_flat_workgroup_size)"
        )
    return {"block": block, "kernarg_size": end, "args": out}


def load_hsaco(path, specs, sha256=None):
    """{name: Kernel} for the kernels named in ``specs`` of the code object at ``path`` (no manifest;
    see kernel_meta). Reads the file only; the GPU is touched at the first launch."""
    table = codeobj.kernels(path)
    module = Module(path, sha256)
    out = {}
    for name, spec in specs.items():
        if name not in table:
            raise HipError(f"{path}: kernel {name} not in the code object ({sorted(table)})")
        out[name] = Kernel(module, name, kernel_meta(table[name], name, spec))
    return out
