"""Does the dq job's operand come free on gfx1250, as the gfx942 template says?

dkdv's does not (accumulator_operand_probe.py). dq is the other half, and the template's
claim is different there: it computes the scores TRANSPOSED (S^T = K @ Q^T) so that the
dS^T fragment IS, for free, the dS operand that dQ = dS @ K needs.

On gfx1250 the layout algebra says this one SHOULD hold, and for a reason worth stating:

    accumulator for kv-tile j   lane l holds S^T[j*16 + (l//16)*8 + si, l%16], si in 0..7
    dS A-operand (M=q, K=kv)    lane l needs dS[l%16, (l//16)*8 + (e%8) + (e//8)*16]

    dS[q, kv] == S^T[kv, q], and l%16 is q on BOTH sides, so the lane already owns the
    right q. Within the lane: e = 0..7 wants kv = half*8 + 0..7 -- exactly tile j's eight
    values -- and e = 8..15 wants kv = half*8 + 16..23 -- exactly tile j+1's eight. Two
    consecutive kv-tile accumulators CONCATENATE in-lane into one v16 operand.

That is a plausible-sounding argument of the kind that is worth not trusting, so: measure.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# flydsl 0.3.2 is required (0.2.4 cannot build aiter's gfx1250 kernels) and is installed
# side by side rather than over the image's 0.2.4, which other things in this container
# still need. Durable path first: /tmp does not survive a container restart, and the
# unattended op-evolve job runs for up to 48h with a supervisor that restarts things.
for _p in ("/home/lihuzhan/.local/flydsl032", "/tmp/flydsl032"):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir
print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")

def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))
def _atom(dt, vec): return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)
def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f); return f.load()
def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))

@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(K0: fx.Tensor, K1: fx.Tensor, Q: fx.Tensor, KV: fx.Tensor,
         S0o: fx.Tensor, S1o: fx.Tensor, DQo: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    row = lane % fx.Int32(16); half = lane // fx.Int32(16)
    g_k0 = _bv(K0, 16*32*2, fx.BFloat16, 8); g_k1 = _bv(K1, 16*32*2, fx.BFloat16, 8)
    g_q  = _bv(Q,  16*32*2, fx.BFloat16, 8); g_kv = _bv(KV, 16*32*2, fx.BFloat16, 8)
    g_s0 = _bv(S0o, 16*16*4, fx.Float32); g_s1 = _bv(S1o, 16*16*4, fx.Float32)
    g_dq = _bv(DQo, 16*16*4, fx.Float32)

    def frag(buf):
        t = row * fx.Int32(4) + half
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))
    v8 = fx.Vector.make_type(8, fx.Float32)
    z = lambda: _ir(fx.Vector.filled(8, 0.0, fx.Float32))
    qf = frag(g_q)
    # S^T for two consecutive kv-tiles: K is A (M = kv), Q is B (N = q)
    s0 = rocdl.wmma_f32_16x16x32_bf16(v8, _ir(frag(g_k0)), _ir(qf), z(), reuseA=False, reuseB=False).result
    s1 = rocdl.wmma_f32_16x16x32_bf16(v8, _ir(frag(g_k1)), _ir(qf), z(), reuseA=False, reuseB=False).result
    s0v, s1v = fx.Vector(s0), fx.Vector(s1)
    for si in range_constexpr(8):
        r = half * fx.Int32(8) + fx.Int32(si)
        _st1(s0v[si], g_s0, r * fx.Int32(16) + row, fx.Float32)
        _st1(s1v[si], g_s1, r * fx.Int32(16) + row, fx.Float32)
    # THE CLAIM: concatenate the two accumulators IN LANE into one v16 A-operand.
    ds = fx.Vector.from_elements(
        [s0v[i].to(fx.BFloat16) for i in range(8)] + [s1v[i].to(fx.BFloat16) for i in range(8)],
        dtype=fx.BFloat16)
    dq = rocdl.wmma_f32_16x16x32_bf16(v8, _ir(ds), _ir(frag(g_kv)), z(),
                                      reuseA=False, reuseB=False).result
    dqv = fx.Vector(dq)
    for si in range_constexpr(8):
        _st1(dqv[si], g_dq, (half * fx.Int32(8) + fx.Int32(si)) * fx.Int32(16) + row, fx.Float32)

@flyc.jit
def launch(K0, K1, Q, KV, S0, S1, DQ, stream: fx.Stream):
    k_dq(K0, K1, Q, KV, S0, S1, DQ).launch(grid=(1,1,1), block=(32,1,1), stream=stream)

torch.manual_seed(0)
mk = lambda: torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
K0, K1, Q, KV = mk(), mk(), mk(), mk()
S0 = torch.full((16,16), float("nan"), device="cuda", dtype=torch.float32)
S1 = S0.clone(); DQ = S0.clone()
launch(K0, K1, Q, KV, S0, S1, DQ, torch.cuda.current_stream()); torch.cuda.synchronize()

def db(r, g): return float(10*torch.log10(r.pow(2).mean()/(r-g).pow(2).mean().clamp_min(1e-30)))
ref_s0 = K0.float() @ Q.float().T
ref_s1 = K1.float() @ Q.float().T
print(f"  S^T tile 0 (sanity)   {db(ref_s0, S0):7.2f} dB")
print(f"  S^T tile 1 (sanity)   {db(ref_s1, S1):7.2f} dB")
# dS[q, kv] over 32 kv = [S0^T | S1^T]; the WMMA truncates it to bf16, so compare against
# the same truncation, isolating the LAYOUT question from bf16 rounding.
dS = torch.cat([S0.T, S1.T], dim=1).to(torch.bfloat16).float()      # [16 q, 32 kv]
ref_dq = dS @ KV.float().T                                          # KV is [16 d, 32 kv] -> B[32,16]
print(f"  dQ via in-lane concat {db(ref_dq, DQ):7.2f} dB")
free = db(ref_dq, DQ) >= 40
print("\nVERDICT:", "dq's operand IS free on gfx1250 -- two kv-tile accumulators concatenate in-lane"
      if free else "dq's operand is NOT free either")
json.dump({"s0_db": round(db(ref_s0,S0),2), "s1_db": round(db(ref_s1,S1),2),
           "dq_db": round(db(ref_dq,DQ),2), "dq_operand_free": free},
          open("/tmp/dq_free.json","w"), indent=2)
