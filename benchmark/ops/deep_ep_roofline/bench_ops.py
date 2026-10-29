###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# DeepEP intranode dispatch / combine against RCCL all_to_all_single, same tokens and routing.
# Launch with torchrun, one process per GPU (docs/flydsl_deepep_mi455_roofline.md).
#
# DeepEP (Buffer API, PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1 so the kernels run on the timed stream):
#   dispatch_sync    the notify, its host wait for the receive counts, and the dispatch
#   dispatch_cached  a dispatch reusing the handle: receive counts known, no notify, no host wait
#   dispatch_worst   num_worst_tokens = T * world: notify on the GPU only, no host wait
#   combine          combine of the received rows with their top-k weights
#   layout           get_dispatch_layout (excluded from the dispatch times, as RCCL's local counting is)
# RCCL all_to_all_single (default RCCL settings), two row sets:
#   megatron  every (token, expert) pair, sorted by destination rank: what Megatron's alltoall
#             dispatcher sends after its permute
#   dedup     each token once per destination rank it has an expert on: the rows DeepEP sends
#   for each: <set>_dispatch (splits known), <set>_dispatch+counts (the count all_gather and the D2H
#   copy of the splits that Megatron does first, then the all-to-all), <set>_combine (the same rows
#   back to their source rank, splits known).
# Bytes: "xgmi_in" / "xgmi_out" are the most hidden-row bytes a rank receives from / sends to its peers
# (bf16 rows; DeepEP's per-row metadata, 4 + 12 * topk bytes, is not counted).
#
# env: TOKENS_LIST (8192), HIDDEN (7168), TOPK (8), EXPERTS (128), ROUTING (dsv3 | uniform), GROUPS (8),
#      GROUP_TOPK (4), BACKENDS (flydsl_v2,flydsl,rccl), CU_LIST (32,48,64,80,96,128,160,192,224,256),
#      CHECK (1: check DeepEP outputs once per configuration), ITERS, ROUNDS, FLUSH_MB, OUT_JSON, TAG,
#      PRIMUS_TURBO_DEEPEP_V2_PHASE_TIMING (1: also an "<op>_phases" row per flydsl_v2 op, from the
#      kernels' stamps of the op's last call; use its own FLYDSL_RUNTIME_CACHE_DIR)
import os

import torch
import torch.distributed as dist
from common import RANK, WORLD, Results, bench, finish, gather_matrix, init, routing, xgmi_rows

TOKENS_LIST = [int(t) for t in os.environ.get("TOKENS_LIST", "8192").split(",")]
H = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
E = int(os.environ.get("EXPERTS", 128))
ROUTING = os.environ.get("ROUTING", "dsv3")
GROUPS = int(os.environ.get("GROUPS", 8))
GROUP_TOPK = int(os.environ.get("GROUP_TOPK", 4))
BACKENDS = os.environ.get("BACKENDS", "flydsl_v2,flydsl,rccl").split(",")
CU_LIST = [int(c) for c in os.environ.get("CU_LIST", "32,48,64,80,96,128,160,192,224,256").split(",")]
CHECK = os.environ.get("CHECK", "1") == "1"
ROW_BYTES = H * 2
NLE = E // WORLD

group = init()
import primus_turbo.pytorch as pt  # noqa: E402
from primus_turbo.flydsl.deep_ep.intranode_v2_kernel import (  # noqa: E402
    MAX_CHANNELS,
    PHASE_EVENTS,
    PHASE_TIMING,
    PHASE_TIMING_OFFSET,
)

Buffer = pt.deep_ep.Buffer
results = Results(
    dict(
        bench="ops",
        tokens_list=TOKENS_LIST,
        H=H,
        K=K,
        E=E,
        routing=ROUTING,
        groups=GROUPS,
        group_topk=GROUP_TOPK,
        backends=BACKENDS,
        cu_list=CU_LIST,
        tag=os.environ.get("TAG", ""),
        rccl_env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "RCCL_"))},
    )
)
if RANK == 0:
    print(
        f"ops EP{WORLD} T={TOKENS_LIST} H={H} top-{K}/{E} routing={ROUTING} groups={GROUPS}/{GROUP_TOPK} "
        f"backends={BACKENDS} cus={CU_LIST} iters={os.environ.get('ITERS', 30)}x{os.environ.get('ROUNDS', 3)}",
        flush=True,
    )


def heap_bytes(backend):
    if backend == "flydsl_v2":
        return Buffer.get_token_buffer_size_hint(max(TOKENS_LIST), ROW_BYTES, WORLD, K, backend=backend)
    num_bytes = 0
    for cu in CU_LIST:
        Buffer.set_num_sms(cu)
        for cfg in (
            Buffer.get_dispatch_config(WORLD, backend=backend),
            Buffer.get_combine_config(WORLD, backend=backend),
        ):
            num_bytes = max(num_bytes, cfg.get_nvl_buffer_size_hint(ROW_BYTES, WORLD))
    return num_bytes


def check_deepep(x, d, combined, itr):
    """recv_x rows are the senders' rows of recv_src_idx, in source-rank order; the combine of the
    unmodified rows is each token times the number of ranks it went to (exact for <= 2 ranks)."""
    recv_x, handle = d[0], d[4]
    rank_prefix, recv_src_idx = handle[0], handle[3]
    xs = [torch.empty_like(x) for _ in range(WORLD)]
    dist.all_gather(xs, x)
    prefix = rank_prefix[:, RANK].long()
    rows = torch.arange(recv_x.shape[0], device="cuda")
    src_rank = torch.searchsorted(prefix, rows, right=True)
    expect = torch.stack(xs)[src_rank, recv_src_idx.long()]
    ok_dispatch = bool(torch.equal(recv_x, expect))
    ranks_hit = itr.sum(1, keepdim=True).to(torch.float32)
    expect_c = (x.float() * ranks_hit).to(torch.bfloat16)
    ok_combine = (
        bool(torch.equal(combined, expect_c))
        if WORLD <= 2
        else bool(torch.allclose(combined.float(), expect_c.float(), rtol=1e-2, atol=1e-2))
    )
    oks = [None] * WORLD
    dist.all_gather_object(oks, (ok_dispatch, ok_combine))
    return all(o[0] for o in oks), all(o[1] for o in oks)


def run_deepep(backend, T, x, topk_idx, topk_w, buf):
    for cu in CU_LIST:
        try:
            run_deepep_cu(backend, T, x, topk_idx, topk_w, buf, cu)
        except (RuntimeError, AssertionError, ValueError) as e:
            # a host-side check refuses the configuration on every rank alike, before any launch
            results.add(impl=backend, T=T, sms=cu, op="error", error=str(e).splitlines()[0][:200])


def run_deepep_cu(backend, T, x, topk_idx, topk_w, buf, cu):
    Buffer.set_num_sms(cu)
    ntr, _, nte, itr, _ = buf.get_dispatch_layout(topk_idx, E)
    m = gather_matrix(ntr)
    rows_in, rows_out = xgmi_rows(m)
    # rows a rank sends to itself: they stay in HBM, but the receiver copies them out like the peers' rows
    rows_self = max(int(m[r][r]) for r in range(WORLD))
    common = dict(
        impl=backend,
        T=T,
        sms=cu,
        rows_in=rows_in,
        rows_out=rows_out,
        rows_self=rows_self,
        xgmi_in=rows_in * ROW_BYTES,
        xgmi_out=rows_out * ROW_BYTES,
    )
    st = {}

    def layout():
        buf.get_dispatch_layout(topk_idx, E)

    def d_sync():
        st["d"] = buf.dispatch(
            x,
            num_tokens_per_rank=ntr,
            is_token_in_rank=itr,
            num_tokens_per_expert=nte,
            topk_idx=topk_idx,
            topk_weights=topk_w,
        )

    def d_worst():
        buf.dispatch(
            x,
            num_tokens_per_rank=ntr,
            is_token_in_rank=itr,
            num_tokens_per_expert=nte,
            topk_idx=topk_idx,
            topk_weights=topk_w,
            num_worst_tokens=T * WORLD,
        )

    def d_cached():
        buf.dispatch(x, handle=st["d"][4])

    def combine():
        st["c"] = buf.combine(st["d"][0], st["d"][4], topk_weights=st["d"][2])

    d_sync()
    combine()
    torch.cuda.synchronize()
    if CHECK:
        ok_d, ok_c = check_deepep(x, st["d"], st["c"][0], itr)
        results.add(impl=backend, T=T, sms=cu, op="check", dispatch_ok=ok_d, combine_ok=ok_c)
    results.add(op="layout", **common, **bench(layout))
    for op, fn in (
        ("dispatch_sync", d_sync),
        ("dispatch_cached", d_cached),
        ("dispatch_worst", d_worst),
        ("combine", combine),
    ):
        results.add(op=op, **common, **bench(fn))
        if PHASE_TIMING and backend == "flydsl_v2":
            results.add(
                op=f"{op}_phases", **common, **phases(buf, "combine" if op == "combine" else "dispatch", cu)
            )


def phases(buf, kernel, num_blocks):
    """The stamps of the last call of ``kernel`` on every rank (PRIMUS_TURBO_DEEPEP_V2_PHASE_TIMING), in µs
    from the rank's first block start. dispatch: send = every block has sent, recv = every block has
    copied what it received, per block medians of the same; combine: push = warp 0 of every block has
    pushed, end = every block is done."""
    torch.cuda.synchronize()
    runtime = buf.intranode_runtime
    k = 0 if kernel == "dispatch" else 1
    first = (PHASE_TIMING_OFFSET + k * MAX_CHANNELS * PHASE_EVENTS * 8) // 8
    stamps = runtime.symm.get_buffer(RANK, (num_blocks, PHASE_EVENTS), torch.int64, first).clone()
    gathered = [torch.empty_like(stamps) for _ in range(WORLD)]
    dist.all_gather(gathered, stamps)
    out = {}
    for r, s in enumerate(gathered):
        s = (s.cpu().double() - s[:, 0].min().item()) / 100.0  # 100 MHz ticks -> µs
        if kernel == "dispatch":
            out[f"r{r}"] = dict(
                send=s[:, 1].max().item(),
                recv=s[:, 2].max().item(),
                send_block_median=(s[:, 1] - s[:, 0]).median().item(),
                recv_block_median=(s[:, 2] - s[:, 1]).median().item(),
                first_send_end=s[:, 1].min().item(),
                start_spread=s[:, 0].max().item(),
            )
        else:
            out[f"r{r}"] = dict(
                push=s[:, 1].max().item(),
                end=s[:, 3].max().item(),
                push_block_median=(s[:, 1] - s[:, 0]).median().item(),
                reduce_w0_block_median=(s[:, 2] - s[:, 1]).median().item(),
                end_block_median=(s[:, 3] - s[:, 0]).median().item(),
                start_spread=s[:, 0].max().item(),
            )
    return dict(phases=out)


def run_rccl(T, x, topk_idx, itr):
    sets = {}
    dest = (topk_idx // NLE).flatten()
    order = torch.sort(dest, stable=True).indices
    sets["megatron"] = (order // K, torch.bincount(dest, minlength=WORLD))
    rows = [torch.nonzero(itr[:, d]).flatten() for d in range(WORLD)]
    sets["dedup"] = (torch.cat(rows), torch.tensor([r.numel() for r in rows], device="cuda"))
    for name, (src_rows, send) in sets.items():
        m = gather_matrix(send)
        in_splits, out_splits = m[RANK].tolist(), m[:, RANK].tolist()
        rows_in, rows_out = xgmi_rows(m)
        common = dict(
            impl="rccl",
            T=T,
            rows_in=rows_in,
            rows_out=rows_out,
            xgmi_in=rows_in * ROW_BYTES,
            xgmi_out=rows_out * ROW_BYTES,
        )
        send_x = x.index_select(0, src_rows)
        recv_x = torch.empty((sum(out_splits), H), dtype=x.dtype, device="cuda")
        back_x = torch.empty_like(send_x)

        def dispatch(recv_x=recv_x, send_x=send_x, out_splits=out_splits, in_splits=in_splits):
            dist.all_to_all_single(recv_x, send_x, out_splits, in_splits)

        def dispatch_counts(recv_x=recv_x, send_x=send_x, send=send):
            gathered = torch.empty(WORLD * WORLD, dtype=send.dtype, device="cuda")
            dist.all_gather_into_tensor(gathered, send)
            g = gathered.view(WORLD, WORLD).cpu()
            dist.all_to_all_single(recv_x, send_x, g[:, RANK].tolist(), g[RANK].tolist())

        def combine(back_x=back_x, recv_x=recv_x, in_splits=in_splits, out_splits=out_splits):
            dist.all_to_all_single(back_x, recv_x, in_splits, out_splits)

        dispatch()
        combine()
        torch.cuda.synchronize()
        if CHECK:
            ok = bool(torch.equal(back_x, send_x))
            oks = [None] * WORLD
            dist.all_gather_object(oks, ok)
            results.add(impl="rccl", T=T, op=f"{name}_check", roundtrip_ok=all(oks))
        results.add(op=f"{name}_dispatch", **common, **bench(dispatch))
        results.add(op=f"{name}_dispatch+counts", **common, **bench(dispatch_counts))
        results.add(op=f"{name}_combine", **common, **bench(combine))
        del send_x, recv_x, back_x


inputs = {}
for T in TOKENS_LIST:
    topk_idx, topk_w = routing(T, E, K, ROUTING, 1234 + RANK, GROUPS, GROUP_TOPK)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4321 + RANK)
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=gen)
    inputs[T] = (x, topk_idx, topk_w)

for backend in [b for b in BACKENDS if b != "rccl"]:
    buf = Buffer(group, heap_bytes(backend), 0, explicitly_destroy=True, intranode_backend=backend)
    assert buf.intranode_backend == backend, buf.intranode_backend
    for T in TOKENS_LIST:
        if RANK == 0:
            print(f"== {backend} T={T} heap={buf.num_nvl_bytes / 2**20:.0f} MiB", flush=True)
        run_deepep(backend, T, *inputs[T], buf)
    torch.cuda.synchronize()
    dist.barrier()
    buf.destroy()

if "rccl" in BACKENDS:
    for T in TOKENS_LIST:
        x, topk_idx, _ = inputs[T]
        is_in_rank = torch.zeros(T, WORLD, dtype=torch.bool, device="cuda")
        is_in_rank.scatter_(1, topk_idx // NLE, True)
        if RANK == 0:
            print(f"== rccl T={T}", flush=True)
        run_rccl(T, x, topk_idx, is_in_rank)

finish(results)
