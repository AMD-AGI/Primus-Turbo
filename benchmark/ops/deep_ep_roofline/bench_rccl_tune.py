###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# RCCL all-to-all at the PR #583 shape in the variants of the RCCL team's rccl-tests sweep (as
# wenxie-amd/minimax-poc-20261006 examples/06-rccl): buffers x launch here, NCCL_CTA_POLICY / RCCL_FORCE_CE per
# process. Same routing, row sets and timing as bench_ops.py; launch with torchrun, one process per GPU.
#   plain_torch  dist.all_to_all_single on caching-allocator tensors: what Megatron runs (the PR's baseline)
#   plain        ncclAlltoAllv on caching-allocator memory
#   local        ncclMemAlloc + ncclCommRegister (rccl-tests -R 1)
#   symm         ncclMemAlloc + ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC) (rccl-tests -R 2)
# launch: eager, or graph (one all-to-all captured in a hipGraph, replayed per iteration; not for plain_torch and
# not for the count exchange, which needs the host). Every variant is checked against torch's all-to-all first.
#
# EQUAL=1 adds `equal_alltoall`: ncclAllToAll with even splits moving the dedup rows' volume, the collective that
# 06-rccl's busbw figures measure; EQUAL_MB sets its per-rank buffer instead (06-rccl's "size", 2^20 B). B2B=N
# also times it as 06-rccl does (`equal_alltoall_b2b`, eager only): N calls back to back between two events, no L2
# flush, the mean per call of the slowest rank.
#
# env: BUFFERS (plain_torch,plain,local,symm), LAUNCH (eager,graph), EQUAL (1), EQUAL_MB, B2B, TOKENS (8192),
#      HIDDEN, TOPK, EXPERTS, ROUTING, GROUPS, GROUP_TOPK, ITERS, ROUNDS, FLUSH_MB, OUT_JSON, TAG
import ctypes
import os

import torch
import torch.distributed as dist
from common import RANK, WORLD, Results, bench, finish, gather_matrix, init, routing, xgmi_rows

T = int(os.environ.get("TOKENS", 8192))
H = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
E = int(os.environ.get("EXPERTS", 128))
ROUTING = os.environ.get("ROUTING", "dsv3")
GROUPS = int(os.environ.get("GROUPS", 8))
GROUP_TOPK = int(os.environ.get("GROUP_TOPK", 4))
BUFFERS = os.environ.get("BUFFERS", "plain_torch,plain,local,symm").split(",")
LAUNCH = os.environ.get("LAUNCH", "eager,graph").split(",")
EQUAL = os.environ.get("EQUAL", "1") == "1"
EQUAL_MB = int(os.environ.get("EQUAL_MB", 0))
B2B = int(os.environ.get("B2B", 0))
ROW = H * 2
NLE = E // WORLD
NCCL_UINT8, NCCL_WIN_COLL_SYMMETRIC = 1, 0x01

group = init()
backend = group._get_backend(torch.device("cuda"))
results = Results(
    dict(
        bench="rccl_tune",
        T=T,
        H=H,
        K=K,
        E=E,
        routing=ROUTING,
        buffers=BUFFERS,
        launch=LAUNCH,
        tag=os.environ.get("TAG", ""),
        rccl_env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "RCCL_"))},
    )
)


def loaded(name):
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1]
            if os.path.basename(p).startswith(name):
                return p
    raise RuntimeError(f"{name} not loaded")


# the communicator exists once a collective ran
dist.all_reduce(torch.zeros(1, device="cuda"))
rccl = ctypes.CDLL(loaded("librccl.so"))
hip = ctypes.CDLL(loaded("libamdhip64.so"))
comm = ctypes.c_void_p(backend._comm_ptr())
if RANK == 0:
    version = ctypes.c_int()
    rccl.ncclGetVersion(ctypes.byref(version))
    print(
        f"rccl_tune EP{WORLD} T={T} {ROUTING} buffers={BUFFERS} launch={LAUNCH} rccl={version.value} "
        f"env={results.data['rccl_env']}",
        flush=True,
    )


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: {rc}")


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


topk_idx, _ = routing(T, E, K, ROUTING, 1234 + RANK, GROUPS, GROUP_TOPK)
gen = torch.Generator(device="cuda")
gen.manual_seed(4321 + RANK)
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=gen)
dest = (topk_idx // NLE).flatten()
in_rank = torch.zeros(T, WORLD, dtype=torch.bool, device="cuda").scatter_(1, topk_idx // NLE, True)
dedup_rows = [torch.nonzero(in_rank[:, d]).flatten() for d in range(WORLD)]
SETS = {
    "megatron": (torch.sort(dest, stable=True).indices // K, torch.bincount(dest, minlength=WORLD)),
    "dedup": (torch.cat(dedup_rows), torch.tensor([r.numel() for r in dedup_rows], device="cuda")),
}


class Rows:
    """One row set: host splits, the rows to send, torch's received rows as the reference."""

    def __init__(self, src_rows, send):
        self.send = send
        self.m = gather_matrix(send)
        self.in_splits, self.out_splits = self.m[RANK].tolist(), self.m[:, RANK].tolist()
        self.rows_in, self.rows_out = xgmi_rows(self.m)
        self.send_x = x.index_select(0, src_rows).contiguous()
        self.n_send, self.n_recv = sum(self.in_splits), sum(self.out_splits)
        self.ref = torch.empty(self.n_recv, H, dtype=x.dtype, device="cuda")
        dist.all_to_all_single(self.ref, self.send_x, self.out_splits, self.in_splits)

    @staticmethod
    def arrays(splits):
        counts = [int(v) * ROW for v in splits]
        displs = [sum(counts[:i]) for i in range(len(counts))]
        return (ctypes.c_size_t * WORLD)(*counts), (ctypes.c_size_t * WORLD)(*displs)


rowsets = {name: Rows(*v) for name, v in SETS.items()}
# equal-split ncclAllToAll moving the dedup rows' volume (the largest rank's, split evenly): the collective of
# 06-rccl's busbw roofline, to set against ncclAlltoAllv of the same bytes with the real splits
n_eq = torch.tensor([rowsets["dedup"].n_send], device="cuda")
dist.all_reduce(n_eq, op=dist.ReduceOp.MAX)
EQ = -(-int(n_eq.item()) // WORLD) if EQUAL else 0
if EQUAL_MB:
    EQ = (EQUAL_MB << 20) // WORLD // ROW
size = max(max(max(r.n_send, r.n_recv) for r in rowsets.values()), EQ * WORLD) * ROW
size = (size + (1 << 21) - 1) >> 21 << 21


def alloc(kind):
    """(pointer, keep-alive) of a `size`-byte buffer of the given kind."""
    if kind in ("plain", "plain_torch"):
        t = torch.empty(size, dtype=torch.uint8, device="cuda")
        return t.data_ptr(), t
    p = ctypes.c_void_p()
    ck(rccl.ncclMemAlloc(ctypes.byref(p), ctypes.c_size_t(size)), "ncclMemAlloc")
    h = ctypes.c_void_p()
    if kind == "local":
        ck(rccl.ncclCommRegister(comm, p, ctypes.c_size_t(size), ctypes.byref(h)), "ncclCommRegister")
    else:
        ck(
            rccl.ncclCommWindowRegister(
                comm, p, ctypes.c_size_t(size), ctypes.byref(h), NCCL_WIN_COLL_SYMMETRIC
            ),
            "ncclCommWindowRegister",
        )
    return p.value, (p, h)


def copy(dst, src, nbytes):
    ck(
        hip.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), 3, stream()),
        "hipMemcpyAsync",
    )


def read(ptr, n):
    out = torch.empty(n, H, dtype=torch.bfloat16, device="cuda")
    copy(out.data_ptr(), ptr, n * ROW)
    return out


def a2av(sbuf, s_arrays, rbuf, r_arrays):
    ck(
        rccl.ncclAlltoAllv(
            ctypes.c_void_p(sbuf),
            s_arrays[0],
            s_arrays[1],
            ctypes.c_void_p(rbuf),
            r_arrays[0],
            r_arrays[1],
            NCCL_UINT8,
            comm,
            stream(),
        ),
        "ncclAlltoAllv",
    )


def graphed(fn):
    """fn captured once in a hipGraph; replays it."""
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()  # warm the communicator for this buffer before capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    hip.hipGetLastError()  # RCCL's buffer probe during capture leaves an expected error behind
    torch.cuda.synchronize()
    return g.replay


def measure(kind, sbuf, rbuf, bbuf, name, rs):
    """Dispatch, combine and dispatch with the count exchange of one row set, on one buffer kind."""
    fields = dict(
        impl=f"rccl_{kind}",
        T=T,
        rows_in=rs.rows_in,
        rows_out=rs.rows_out,
        xgmi_in=rs.rows_in * ROW,
        xgmi_out=rs.rows_out * ROW,
    )
    sa, ra = Rows.arrays(rs.in_splits), Rows.arrays(rs.out_splits)
    copy(sbuf, rs.send_x.data_ptr(), rs.n_send * ROW)
    torch.cuda.synchronize()
    if kind == "plain_torch":
        send_t = torch.empty(rs.n_send, H, dtype=x.dtype, device="cuda").copy_(rs.send_x)
        recv_t = torch.empty(rs.n_recv, H, dtype=x.dtype, device="cuda")
        back_t = torch.empty_like(send_t)

        def dispatch():
            dist.all_to_all_single(recv_t, send_t, rs.out_splits, rs.in_splits)

        def combine():
            dist.all_to_all_single(back_t, recv_t, rs.in_splits, rs.out_splits)

        def dispatch_counts():
            gathered = torch.empty(WORLD * WORLD, dtype=rs.send.dtype, device="cuda")
            dist.all_gather_into_tensor(gathered, rs.send)
            gm = gathered.view(WORLD, WORLD).cpu()
            dist.all_to_all_single(recv_t, send_t, gm[:, RANK].tolist(), gm[RANK].tolist())

        def outputs():
            return recv_t, back_t
    else:

        def dispatch():
            a2av(sbuf, sa, rbuf, ra)

        def combine():
            a2av(rbuf, ra, bbuf, sa)

        def dispatch_counts():
            gathered = torch.empty(WORLD * WORLD, dtype=rs.send.dtype, device="cuda")
            dist.all_gather_into_tensor(gathered, rs.send)
            gm = gathered.view(WORLD, WORLD).cpu()
            a2av(sbuf, Rows.arrays(gm[RANK].tolist()), rbuf, Rows.arrays(gm[:, RANK].tolist()))

        def outputs():
            return read(rbuf, rs.n_recv), read(bbuf, rs.n_send)

    dispatch()
    combine()
    torch.cuda.synchronize()
    got, back = outputs()
    torch.cuda.synchronize()
    oks = [None] * WORLD
    dist.all_gather_object(oks, (bool(torch.equal(got, rs.ref)), bool(torch.equal(back, rs.send_x))))
    ok = all(o[0] and o[1] for o in oks)
    results.add(impl=f"rccl_{kind}", T=T, op=f"{name}_check", ok=ok)
    if not ok:
        return
    for launch in LAUNCH:
        if launch == "graph" and kind == "plain_torch":
            continue
        ops = [("dispatch", dispatch), ("combine", combine)]
        if launch == "eager":
            ops.append(("dispatch+counts", dispatch_counts))
        for op, fn in ops:
            try:
                run = graphed(fn) if launch == "graph" else fn
                results.add(op=f"{name}_{op}", launch=launch, **fields, **bench(run))
            except RuntimeError as e:
                results.add(impl=f"rccl_{kind}", T=T, op=f"{name}_{op}", launch=launch, error=str(e)[:200])
        if launch == "graph":
            run_check = graphed(dispatch)
            run_check()
            torch.cuda.synchronize()
            got, _ = outputs()
            oks = [None] * WORLD
            dist.all_gather_object(oks, bool(torch.equal(got, rs.ref)))
            results.add(impl=f"rccl_{kind}", T=T, op=f"{name}_graph_check", ok=all(oks))


def measure_equal(kind, sbuf, rbuf):
    """Equal-split all-to-all of EQ rows per peer, on one buffer kind."""
    send_eq = torch.randn(EQ * WORLD, H, dtype=x.dtype, device="cuda", generator=gen)
    ref_eq = torch.empty_like(send_eq)
    dist.all_to_all_single(ref_eq, send_eq)
    if kind == "plain_torch":
        recv_eq = torch.empty_like(send_eq)

        def equal():
            dist.all_to_all_single(recv_eq, send_eq)

        def received():
            return recv_eq
    else:
        copy(sbuf, send_eq.data_ptr(), EQ * WORLD * ROW)

        def equal():
            ck(
                rccl.ncclAllToAll(
                    ctypes.c_void_p(sbuf),
                    ctypes.c_void_p(rbuf),
                    ctypes.c_size_t(EQ * ROW),
                    NCCL_UINT8,
                    comm,
                    stream(),
                ),
                "ncclAllToAll",
            )

        def received():
            return read(rbuf, EQ * WORLD)

    equal()
    torch.cuda.synchronize()
    oks = [None] * WORLD
    dist.all_gather_object(oks, bool(torch.equal(received(), ref_eq)))
    results.add(impl=f"rccl_{kind}", T=T, op="equal_check", ok=all(oks))
    if not all(oks):
        return
    link = EQ * (WORLD - 1)
    fields = dict(
        impl=f"rccl_{kind}", T=T, rows_in=link, rows_out=link, xgmi_in=link * ROW, xgmi_out=link * ROW
    )
    for launch in LAUNCH:
        if launch == "graph" and kind == "plain_torch":
            continue
        try:
            results.add(
                op="equal_alltoall",
                launch=launch,
                **fields,
                **bench(graphed(equal) if launch == "graph" else equal),
            )
        except RuntimeError as e:
            results.add(impl=f"rccl_{kind}", T=T, op="equal_alltoall", launch=launch, error=str(e)[:200])
    if B2B and "eager" in LAUNCH:
        results.add(op="equal_alltoall_b2b", launch="eager", **fields, us=back_to_back(equal, B2B))


def back_to_back(fn, iters, warmup=10):
    """Mean µs per call of `iters` calls between two events, slowest rank (06-rccl's timing)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) * 1e3 / iters], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


for kind in BUFFERS:
    sbuf, keep_s = alloc(kind)
    rbuf, keep_r = alloc(kind)
    bbuf, keep_b = alloc(kind)
    for name, rs in rowsets.items():
        measure(kind, sbuf, rbuf, bbuf, name, rs)
    if EQUAL:
        measure_equal(kind, sbuf, rbuf)

finish(results)
