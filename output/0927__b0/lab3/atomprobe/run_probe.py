"""ON-CARD P2 driver -- DO NOT RUN while another GPU job holds the card.
One process = one (MODE, KVG) point (rule 3: one shape per process). Prints ms and GB/s.
usage: ATOM_MODE=atom python run_probe.py <KVG> [iters]
"""
import os, sys, time, importlib.util
T0 = time.time()
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
import torch, flydsl, flydsl.compiler as flyc
assert flydsl.__version__ == "0.3.4.1"
sp = importlib.util.spec_from_file_location("k_atom", os.path.join(os.path.dirname(__file__), "k_atom.py"))
K = importlib.util.module_from_spec(sp); sp.loader.exec_module(K)
kvg = int(sys.argv[1]); iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
b, sq, hq, hkv, d = 4, 8192, 32, 8, 128
acc = torch.zeros((b, sq, hq, d), dtype=torch.float32, device="cuda")
s = torch.cuda.current_stream()
args = (acc, sq, sq, hq, hq // hkv, sq // 32, 0, kvg, b, hkv, sq // (32 * kvg), b, s)
T1 = time.time()
fn = flyc.compile(K.launch_atom, *args)
torch.cuda.synchronize()
T2 = time.time()
# lane-atomics issued (host count, same walk as the kernel)
n_it = sum(hq // hkv * (sq // 32 - (bid * 32 * kvg) // 32) for bid in range(sq // (32 * kvg))) * hkv * b
lane_ops = n_it * 2 * 8 * 8 * 32
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
ts = []
for _ in range(iters):
    acc.zero_(); torch.cuda.synchronize()
    ev[0].record(); fn(*args); ev[1].record(); torch.cuda.synchronize()
    ts.append(ev[0].elapsed_time(ev[1]))
ts.sort(); med = ts[len(ts) // 2]
# lab3 witness: the kernel adds val=(it+1) at 32 lanes x 128 slots per iteration of each WG,
# so sum(acc) is exact (per-element partials < 2^24) and equals 4096 * sum_WG n(n+1)/2.
G_ = hq // hkv
exp = 4096 * sum((lambda n: n * (n + 1) // 2)(G_ * (sq // 32 - bid * kvg)) for bid in range(sq // (32 * kvg))) * hkv * b
got = float(acc.double().sum().item()); nz = int((acc != 0).sum().item())
fin = bool(torch.isfinite(acc).all().item())
print(f"WITNESS pid={os.getpid()} dev={torch.cuda.get_device_name(0)} ndev={torch.cuda.device_count()} "
      f"sum={got:.6e} expected_atom={exp:.6e} match={got == exp} nonzero={nz}/{acc.numel()} finite={fin} "
      f"t_init={T1-T0:.1f}s t_compile={T2-T1:.1f}s t_timed={time.time()-T2:.1f}s n={len(ts)}")
if K.MODE == "atom":
    assert fin and got == exp, "atomic witness mismatch"
print(f"MODE={K.MODE} KVG={kvg} wave_instr={lane_ops//32} lane_ops={lane_ops} bytes={lane_ops*4/1e9:.2f}GB "
      f"median={med:.3f}ms min={ts[0]:.3f}ms payload={lane_ops*4/med/1e9:.2f}TB/s")
