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
from flydsl.expr import rocdl
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
V=sys.argv[1]; D=f"/tmp/g0_{V}"; os.makedirs(D,exist_ok=True)
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]=D
out=torch.zeros((1<<14,), dtype=torch.float32, device="cpu")

if V=="uni_agent_plain":
    @flyc.kernel
    def kk(OUT):
        lane=fx.Int32(fx.thread_idx.x)
        t=fx.Tensor(fx.make_view(fx.get_iter(OUT), fx.make_layout((1<<14,1),(1,1))))
        f=fx.make_rmem_tensor(fx.make_layout(1,1), fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([fx.Float32(1.0)],dtype=fx.Float32), f)
        atom=fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent), fx.Float32)
        fx.copy_atom_call(atom, f, fx.slice(t,(lane,None)))
elif V=="uni_agent_buf":
    @flyc.kernel
    def kk(OUT):
        lane=fx.Int32(fx.thread_idx.x)
        b=fx.rocdl.make_buffer_tensor(OUT, num_records_bytes=fx.Int64(1<<20))
        t=fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((1<<14,1),(1,1))))
        f=fx.make_rmem_tensor(fx.make_layout(1,1), fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([fx.Float32(1.0)],dtype=fx.Float32), f)
        atom=fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent), fx.Float32)
        fx.copy_atom_call(atom, f, fx.slice(t,(lane,None)))
elif V=="uni_system_buf":
    @flyc.kernel
    def kk(OUT):
        lane=fx.Int32(fx.thread_idx.x)
        b=fx.rocdl.make_buffer_tensor(OUT, num_records_bytes=fx.Int64(1<<20))
        t=fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((1<<14,1),(1,1))))
        f=fx.make_rmem_tensor(fx.make_layout(1,1), fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([fx.Float32(1.0)],dtype=fx.Float32), f)
        atom=fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Float32, fx.SyncScope.System), fx.Float32)
        fx.copy_atom_call(atom, f, fx.slice(t,(lane,None)))
@flyc.jit
def go(OUT): kk(OUT).launch(grid=(1,1,1), block=(32,1,1))
try:
    go(out); print(f"  [{V}] BUILD ok")
except Exception as e:
    print(f"  [{V}] BUILD FAILED {type(e).__name__}: {str(e)[:200]}"); sys.exit(0)
for p in sorted(glob.glob(os.path.join(D,"**","*_final_isa.s"),recursive=True)):
    txt=open(p).read()
    at=[l.strip() for l in txt.split("\n") if re.search(r"atomic",l,re.I)]
    dev=any("SCOPE_DEV" in l for l in at); bad=[l for l in at if "TH_ATOMIC_RETURN" in l or "cmpswap" in l.lower()]
    print(f"  [{V}] ISA: {at if at else '无'}")
    print(f"  [{V}] SCOPE_DEV={dev} 禁忌={'有' if bad else '无'} -> {'*** PASS ***' if dev and not bad else 'fail'}")
