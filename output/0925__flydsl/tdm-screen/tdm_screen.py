"""TDM screen v2, typed LDS storage. COMPILE ONLY -- never launched.
A descriptor with a wrong extent waits on a counter that never retires and HANGS the card."""
import glob, importlib.util, os, re, sys, types
AITER="/home/lihuzhan/code/aiter-src"
sys.path.insert(0,"/home/lihuzhan/.local/flydsl032"); sys.path.insert(0,AITER)
os.environ["COMPILE_ONLY"]="1"; os.environ["ARCH"]="gfx1250"; os.environ["FLYDSL_GPU_ARCH"]="gfx1250"
for pkg in ("aiter","aiter.ops","aiter.ops.flydsl","aiter.ops.flydsl.kernels"):
    if pkg not in sys.modules:
        m=types.ModuleType(pkg); m.__path__=[os.path.join(AITER,*pkg.split("."))]; sys.modules[pkg]=m
for dotted in ("aiter.ops.flydsl.kernels.kernels_common","aiter.ops.flydsl.kernels.tensor_shim"):
    p=os.path.join(AITER,*dotted.split("."))+".py"
    sp=importlib.util.spec_from_file_location(dotted,p); mod=importlib.util.module_from_spec(sp)
    sys.modules[dotted]=mod; sp.loader.exec_module(mod)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
# aiter ships a shim because FlyDSL 0.3.2's own tdm_ops CANNOT take a Fly shared view --
# tdm_ops_gfx1250._FlyAwareMemrefDialect patches extract_aligned_pointer_as_index to accept one.
_tp="aiter.ops.flydsl.kernels.tdm_ops_gfx1250"
_sp=importlib.util.spec_from_file_location(_tp, os.path.join(AITER,*_tp.split("."))+".py")
tdm_ops=importlib.util.module_from_spec(_sp); sys.modules[_tp]=tdm_ops; _sp.loader.exec_module(tdm_ops)

D="/home/lihuzhan/tdm2_dump"; os.makedirs(D,exist_ok=True)
os.environ["FLYDSL_DUMP_IR"]="1"; os.environ["FLYDSL_DUMP_DIR"]=D
ROWS, COLS = 32, 128

@fx.struct
class TdmStore:
    tile: fx.Array[fx.BFloat16, ROWS*COLS, 16]

@flyc.kernel(known_block_size=[32,1,1])
def k_tdm(SRC):
    lds = fx.SharedAllocator().allocate(TdmStore).peek()
    desc = tdm_ops.make_tensor_descriptor_2d(
        global_ptr=SRC,
        lds_memref=fx.Tensor(fx.make_view(lds.tile.ptr, fx.make_layout((ROWS,COLS),(COLS,1)))),
        global_offset=(0,0), tensor_shape=(ROWS,COLS), strides=(COLS,1),
        tile_shape=(ROWS,COLS), elem_bytes=2, num_warps=1)
    tdm_ops.tensor_load_2d(desc)
    tdm_ops.tensor_wait(0)
    fx.barrier()

@flyc.jit
def go(SRC): k_tdm(SRC).launch(grid=(1,1,1), block=(32,1,1))

src=torch.zeros((ROWS,COLS), dtype=torch.bfloat16, device="cpu")
try:
    go(src); print("  BUILD ok")
except Exception as e:
    print(f"  BUILD FAILED {type(e).__name__}: {str(e)[:260]}"); sys.exit(0)
for p in sorted(glob.glob(os.path.join(D,"**","*_final_isa.s"),recursive=True)):
    t=open(p).read()
    hits=[l.strip() for l in t.split("\n") if re.search(r"tensor_load|tensor_store|0xd031|tensorcnt|asynccnt", l, re.I)]
    print(f"  TDM/异步: {hits[:5] if hits else '无'}")
    print(f"  buffer_load={t.count('buffer_load')}  ds_store={t.count('ds_store')}")
