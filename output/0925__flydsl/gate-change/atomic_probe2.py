"""fp32 buffer atomic add via the copy-atom path, exactly as our kernel writes stores.
COMPILE_ONLY, zero card time."""
import glob, importlib.util, os, re, sys, types
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

DUMP="/tmp/atomic_probe2_dump"; os.makedirs(DUMP,exist_ok=True)
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]=DUMP

def _bv(t, nb, dt, vec=1):
    nt=(1<<31)//(vec*(dt.width//8))
    b=fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt,vec),(vec,1))))

@flyc.kernel
def k_at(OUT):
    lane=fx.Int32(fx.thread_idx.x)
    g=_bv(OUT, 1<<20, fx.Float32, 1)
    f=fx.make_rmem_tensor(fx.make_layout(1,1), fx.Float32)
    fx.memref_store_vec(fx.Vector.from_elements([fx.Float32(1.0)], dtype=fx.Float32), f)
    atom=fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.Float32), fx.Float32)
    fx.copy_atom_call(atom, f, fx.slice(g, (lane, None)))

@flyc.jit
def go(OUT):
    k_at(OUT).launch(grid=(1,1,1), block=(32,1,1))

out=torch.zeros((1<<14,), dtype=torch.float32, device="cpu")
try:
    go(out); print("  BUILD ok")
except Exception as e:
    print(f"  BUILD FAILED {type(e).__name__}: {str(e)[:300]}")
for p in sorted(glob.glob(os.path.join(DUMP,"**","*_final_isa.s"),recursive=True)):
    txt=open(p).read()
    hits=sorted(set(re.findall(r"\b(buffer_atomic\w*|global_atomic\w*|flat_atomic\w*)", txt)))
    print(f"  ISA 原子指令 = {hits if hits else '无'}")
    for ln in txt.split("\n"):
        if "atomic" in ln: print("   ", ln.strip())
