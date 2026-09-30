# arm dqg_dkdv_streams

## Feasibility (from the code)
- r29 does NOT run k_dqg first: `kernels.py:1173 DQ_DFUSE = False`, so r29's order is
  k_delta -> k_dkdv -> k_dqg, all on `torch.cuda.current_stream()`. The only dependence
  between the dK/dV chain and the dQ chain is `delta` (written by k_delta, read by both).
  Outputs are disjoint (dk/dv vs dq). So the fork after k_delta is legal: the spec's
  fallback is exactly the r29 structure with the dQ chain moved to a side stream.
- `_launch` cache does not bake the stream: the key is (name, tensor dtype/rank, device);
  `flyc.compile`'s CompiledFunction refills every runtime arg per call, and
  `fx.Stream.__c_abi_spec__` (flydsl032 `expr/typing.py:1254-1285`) re-reads
  `raw.cuda_stream` on every call; `__cache_signature__` is `(type(self),)` only.

## What changed (impl.py only; kernels.py and _env.py byte-identical to r29)
- impl.py:110-121 new module constants `DQ_SIDE_STREAM = True` (arm switch; False == r29
  single-stream order), `DQ_SIDE_FIRST = False` (variant b: submit dQ chain before dK/dV
  chain), `_side_stream(dev)` caches one `torch.cuda.Stream` per device.
- impl.py:171-178 `_dqg(out=None, st=stream)` takes a preallocated output and a stream.
- impl.py:188-219 nsp_q derivation moved ahead of k_dkdv; dq_o (and dqp for the split-K
  path) allocated on the main stream BEFORE the fork; then
  `split = DQ_SIDE_STREAM and not fuse`, `s2.wait_stream(main)`.
- impl.py:221-244 `_dq_chain()` = r29's dQ branch (k_dqg / k_dq / k_dq_sp+k_redsp_q) with
  `s2` as the stream; called before (variant b) or after (default) the dK/dV chain.
- impl.py:304-314 default call of `_dq_chain()`, then `record_stream(s2)` on
  q,k,v,do,o,lse,delta,dq_o,dqp, then `main.wait_stream(s2)` before return.
- DQ_DFUSE=True path (k_dkdv depends on k_dqg) stays serial on main.
- Resulting queue graph (prod): delta@main | fork | dkdv@main || dqg@side | join.
  fast/toy (split-K): dkdv_sp,redsp@main || dq_sp,redsp_q@side.

## ISA evidence
No kernel changes: `compile.sh . dkdv dqg` RC=0; dkdv 729 VGPR / 70656 B LDS / 0 spill /
0 scratch, wmma 128 ds 224; dqg 991 VGPR / 8704 B / 0 spill / 0 scratch, wmma 288 ds 96.
Both 21_final_isa.s identical to r29's (diff ignoring comments). The effect is host-side
(queue assignment), not visible in ISA; bounds_proof.py verifies the launch graph.

## Bitwise
Expected bitwise identical to r29 for dq, dk, dv: identical kernels, identical launch
arguments (checked), deterministic kernels, no atomics across chains, disjoint outputs.

## Bounds proof
No index/address/predicate expression changed. bounds_proof.py (host python3, stub torch)
runs r29 and this arm's impl.py for prod b4 s8192 hq32 hkv8, fast b1 s1024 hq8 hkv2, toy
b1 s128 hq2 hkv1, both DQ_DFUSE values and both DQ_SIDE_FIRST values, and asserts every
kernel launch has identical scalar args and tensor shapes/dtypes to r29, and the stream
graph (fork after delta, dQ chain on side, dK/dV chain on main, join last). Result: PASS.

## Risks / what to measure
- With 729/991 VGPR one wave per SIMD of either kernel: overlap only at k_dkdv's causal
  tail/ramp and dqg's tail. Kill: prod < +0.4%, or kbench kernel sum rises by more than the
  saved boundary. Try variant b (DQ_SIDE_FIRST=True) too.
- Side stream is a normal-priority torch Stream; under CUDA/HIP graph capture the
  wait_stream fork/join pattern is capture-legal.
- record_stream adds a small host cost (allocator event per freed block).
