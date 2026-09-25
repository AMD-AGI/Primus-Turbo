"""Does FlyDSL 0.3.2's fp32 buffer atomic lower to buffer_atomic_add_f32 on gfx1250?
COMPILE_ONLY, zero card time."""
import glob, importlib.util, os, sys, types
AITER="/home/lihuzhan/code/aiter-src"
sys.path.insert(0,"/home/lihuzhan/.local/flydsl032"); sys.path.insert(0,AITER)
os.environ["COMPILE_ONLY"]="1"; os.environ["ARCH"]="gfx1250"; os.environ["FLYDSL_GPU_ARCH"]="gfx1250"
for pkg in ("aiter","aiter.ops","aiter.ops.flydsl","aiter.ops.flydsl.kernels"):
    if pkg not in sys.modules:
        m=types.ModuleType(pkg); m.__path__=[os.path.join(AITER,*pkg.split("."))]; sys.modules[pkg]=m
for dotted in ("aiter.ops.flydsl.kernels.kernels_common","aiter.ops.flydsl.kernels.tensor_shim"):
    path=os.path.join(AITER,*dotted.split("."))+".py"
    sp=importlib.util.spec_from_file_location(dotted,path); mod=importlib.util.module_from_spec(sp)
    sys.modules[dotted]=mod; sp.loader.exec_module(mod)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr

DUMP="/tmp/atomic_probe_dump"; os.makedirs(DUMP,exist_ok=True)
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]=DUMP

print("  rocdl 里的原子符号:", [n for n in dir(rocdl) if "tomic" in n])

@flyc.kernel
def k_at(OUT, val: fx.Float32):
    lane=fx.Int32(fx.thread_idx.x)
    b=fx.rocdl.make_buffer_tensor(OUT, num_records_bytes=fx.Int64(1<<20))
    try:
        rocdl.raw_ptr_buffer_atomic_fadd(val, fx.get_iter(b), lane*fx.Int32(4), fx.Int32(0))
    except Exception as e:
        print("  raw_ptr_buffer_atomic_fadd 调用失败:", type(e).__name__, e)
        raise

@flyc.jit
def go(OUT, val: fx.Float32):
    k_at(OUT, val).launch(grid=(1,1,1), block=(32,1,1))

out=torch.zeros((1<<14,), dtype=torch.float32, device="cpu")
try:
    go(out, fx.Float32(1.0)); print("  BUILD ok")
except Exception as e:
    print(f"  BUILD FAILED {type(e).__name__}: {e}")
for p in sorted(glob.glob(os.path.join(DUMP,"**","*_final_isa.s"),recursive=True)):
    txt=open(p).read()
    import re
    hits=sorted(set(re.findall(r"\b(buffer_atomic\w*|global_atomic\w*|ds_add\w*|flat_atomic\w*)", txt)))
    print(f"  {os.path.basename(os.path.dirname(p))}: 原子指令 = {hits if hits else '无'}")
