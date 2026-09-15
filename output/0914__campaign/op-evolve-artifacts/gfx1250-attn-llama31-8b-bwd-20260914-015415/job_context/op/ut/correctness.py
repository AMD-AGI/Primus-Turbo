"""Correctness of an implementation against op/eager/, in SQNR dB.

Importable (validation.py uses `check_shape`) and runnable:

    python op/ut/correctness.py [impl_dir]      # default: op/baseline

SQNR is computed SEPARATELY for out, dq, dk and dv, over the WHOLE tensor.

Both of those are load-bearing, and neither is fussiness:

  * Separately, because the failures this op actually produces are localised to
    one output. A configuration is on record that is 1.31x FASTER with dq at
    9.6 dB and dk/dv perfect; another gives dk at -0.22 dB with out perfect.
    A check on `out` alone, or a mean over all four, passes both of them.

  * Over the whole tensor, because dk and dv accumulate over all four query heads
    of the GQA group. A sample can be drawn entirely from rows whose group
    contribution happened to be right, and the group-sum bug `op.reference.logic`
    warns about is exactly a bug that a sample does not localise.

The reference is fp32 `op/eager/`, never a library and never the implementation
under test. bf16 storage of a correct result floors SQNR near 55 dB, so the
50 dB gate is roughly 5 dB of headroom over the dtype itself -- it is a gate on
being wrong, not on being imprecise.
"""

import argparse
import importlib.util
import os
import sys

import torch

_OP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_OP, "ut"))

from shapes import ALL_SHAPES, DIAGNOSTIC_SHAPES  # noqa: E402

SQNR_DB = 50.0


def load_impl(impl_dir):
    """Load an implementation's `attention` by path, so nothing depends on cwd."""
    path = os.path.join(impl_dir, "impl.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    name = "impl_" + os.path.basename(os.path.normpath(impl_dir))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sqnr_db(ref, test):
    """10 log10 (signal power / error power), in fp64 so the metric is not the noise."""
    ref = ref.double()
    err = ref - test.double()
    sig_p = (ref * ref).sum().item()
    err_p = (err * err).sum().item()
    if err_p == 0.0:
        return float("inf")
    if sig_p == 0.0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(sig_p / err_p)).item()


def make_inputs(shape, device="cuda", seed=0):
    """Fixed seed: correctness must not depend on which draw it got."""
    gen = torch.Generator(device=device).manual_seed(seed)
    b, d = shape["batch"], shape["head_dim"]
    sq, skv = shape["seqlen_q"], shape["seqlen_kv"]
    hq, hkv = shape["heads_q"], shape["heads_kv"]
    kw = dict(device=device, dtype=torch.bfloat16, generator=gen)
    q = torch.randn(b, sq, hq, d, **kw)
    k = torch.randn(b, skv, hkv, d, **kw)
    v = torch.randn(b, skv, hkv, d, **kw)
    do = torch.randn(b, sq, hq, d, **kw)
    return q, k, v, do


def run_impl(attention, q, k, v, do, causal):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    out = attention(q, k, v, causal=causal)
    out.backward(do)
    return out.detach(), q.grad, k.grad, v.grad


def check_shape(impl_dir, shape, device="cuda", threshold=SQNR_DB):
    """Returns (passed, {tensor_name: sqnr_db}). Reference is always op/eager/."""
    if "eager_ref" not in sys.modules:
        # So that any caller -- this file's main(), or op/validation.py -- gets
        # the same reference without having to know to install it first.
        _install_eager()
    import eager_ref

    q, k, v, do = make_inputs(shape, device)
    causal = shape["causal"]

    ref_out, ref_lse = eager_ref.reference_forward(q, k, v, causal)
    ref_dq, ref_dk, ref_dv = eager_ref.reference_backward(do, q, k, v, ref_out, ref_lse, causal)

    attention = load_impl(impl_dir).attention
    out, dq, dk, dv = run_impl(attention, q, k, v, do, causal)

    scores = {
        "out": sqnr_db(ref_out, out),
        "dq": sqnr_db(ref_dq, dq),
        "dk": sqnr_db(ref_dk, dk),
        "dv": sqnr_db(ref_dv, dv),
    }
    return all(s >= threshold for s in scores.values()), scores


def _install_eager():
    """Make op/eager/impl.py importable as `eager_ref` regardless of cwd."""
    spec = importlib.util.spec_from_file_location(
        "eager_ref", os.path.join(_OP, "eager", "impl.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eager_ref"] = mod
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("impl_dir", nargs="?", default=os.path.join(_OP, "baseline"))
    args = ap.parse_args()

    _install_eager()
    print(f"reference: {os.path.join(_OP, 'eager', 'impl.py')} (fp32)")
    print(f"under test: {os.path.join(args.impl_dir, 'impl.py')}")
    print(f"gate: every tensor >= {SQNR_DB:.1f} dB\n")
    print(f"{'shape':<24} {'out':>8} {'dq':>8} {'dk':>8} {'dv':>8}  result")

    diagnostic = {s["name"] for s in DIAGNOSTIC_SHAPES}
    failures = []
    for shape in ALL_SHAPES:
        tag = "  (diagnostic, not gated)" if shape["name"] in diagnostic else ""
        try:
            ok, s = check_shape(args.impl_dir, shape)
        except Exception as exc:  # a shape the impl refuses is a failure, not a skip
            print(f"{shape['name']:<24} {'ERROR':>8} {'':>8} {'':>8} {'':>8}  FAIL  {exc}{tag}")
            if shape["name"] not in diagnostic:
                failures.append(f"{shape['name']}: {type(exc).__name__}: {exc}")
            continue
        print(
            f"{shape['name']:<24} {s['out']:>8.2f} {s['dq']:>8.2f} {s['dk']:>8.2f} "
            f"{s['dv']:>8.2f}  {'PASS' if ok else 'FAIL'}{tag}"
        )
        if not ok and shape["name"] not in diagnostic:
            bad = ", ".join(f"{n}={val:.2f} dB" for n, val in s.items() if val < SQNR_DB)
            failures.append(f"{shape['name']}: below {SQNR_DB:.0f} dB on {bad}")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nPASSED: {len(ALL_SHAPES) - len(diagnostic)} gated shapes, "
          f"all four tensors >= {SQNR_DB:.0f} dB "
          f"({len(diagnostic)} diagnostic shape(s) reported above, not gated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
