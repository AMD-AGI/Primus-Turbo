import os
os.environ["FLYDSL_GPU_ARCH"] = "gfx1250"

print("=== 1. env var alone ===")
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
print(f"  get_rocm_arch() = {get_rocm_arch()!r}")
print(f"  is_rdna_arch()  = {is_rdna_arch()}   <-- env var does NOT fix it")

print("\n=== 2. no device at all (env unset) ===")
os.environ.pop("FLYDSL_GPU_ARCH", None)
import flydsl.runtime.device as dev
dev.get_rocm_arch.__wrapped__ if hasattr(dev.get_rocm_arch,'__wrapped__') else None
print(f"  get_rocm_arch() = {get_rocm_arch()!r}   <-- empty in a GPU-less container")
print(f"  is_rdna_arch()  = {is_rdna_arch()}   <-- silently CDNA/wave64")
os.environ["FLYDSL_GPU_ARCH"] = "gfx1250"

print("\n=== 3. does patching runtime.device reach the consumers? ===")
import flydsl.compiler.backends.rocm as rocm_be
import flydsl.expr.buffer_ops as bops
print(f"  before patch: rocm_be.is_rdna_arch('gfx1250') = {rocm_be.is_rdna_arch('gfx1250')}")
_orig = dev.is_rdna_arch
dev.is_rdna_arch = lambda arch=None: True if (arch or '').lower().startswith('gfx125') else _orig(arch)
print(f"  after  patching flydsl.runtime.device only:")
print(f"    dev.is_rdna_arch('gfx1250')      = {dev.is_rdna_arch('gfx1250')}")
print(f"    rocm_be.is_rdna_arch('gfx1250')  = {rocm_be.is_rdna_arch('gfx1250')}  <-- stale binding?")
print(f"    bops.is_rdna_arch('gfx1250')     = {bops.is_rdna_arch('gfx1250')}  <-- stale binding?")

print("\n=== 4. buffer descriptor flags differ ===")
for arch, label in [("gfx1250","misclassified CDNA"), ("gfx1201","true RDNA")]:
    f = bops.create_buffer_resource.__globals__['_buffer_flags'](arch) if '_buffer_flags' in bops.create_buffer_resource.__globals__ else None
    print(f"  {arch} ({label}): flags = {f if f is None else hex(f)}")
