"""r6 bitwise determinism over N runs on adversarial inputs, one shape per process.
usage: determinism.py SHAPE CAUSAL KIND[,KIND] [N=20]"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from adv_inputs import make  # noqa: E402
from card_run_loader import load_impl  # noqa: E402

shape, causal, kinds = sys.argv[1], bool(int(sys.argv[2])), sys.argv[3].split(",")
n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
fn = load_impl(HERE / "arms" / "r6")
h = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
ok = True
for kind in kinds:
    q, k, v = (x.cuda() for x in make(shape, kind))
    o0, l0 = fn(q, k, v, causal=causal)
    o0, l0 = o0.clone(), l0.clone()
    torch.cuda.synchronize()
    bad = []
    for i in range(1, n):
        o, l = fn(q, k, v, causal=causal)
        torch.cuda.synchronize()
        if not (torch.equal(o.view(torch.int16), o0.view(torch.int16)) and torch.equal(l.view(torch.int32), l0.view(torch.int32))):
            bad.append(i)
    ok &= not bad
    print(f"DET {shape} {'causal' if causal else 'full'} {kind}: {n - len(bad)}/{n} bitwise identical "
          f"o={h(o0)} lse={h(l0)}" + (f" first mismatch run {bad[0]}" if bad else ""), flush=True)
print("DET_RESULT", "PASS" if ok else "FAIL")
