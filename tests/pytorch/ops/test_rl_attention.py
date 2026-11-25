###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RL attention: correctness against an fp32 reference, and the bitwise guarantees RL relies on.
A token's forward output must not depend on the execution mode (paged vs contiguous, split vs
fused), the launch parameters, the other sequences in the batch, or whether it is computed in a
rollout prefill / decode step or the training forward."""

import pytest
import torch

from primus_turbo.common import envs
from primus_turbo.pytorch.kernels.attention import rl_attention_impl
from primus_turbo.pytorch.ops.attention import flash_attn_func, flash_attn_varlen_func
from primus_turbo.pytorch.ops.attention.rl_attention import (
    _resolve_options,
    create_block_mask,
    rl_flash_attn_func,
    rl_flash_attn_varlen_func,
    rl_paged_attention_func,
)
from tests.pytorch.ref.rl_attention_ref import rl_attention_ref

DEVICE = "cuda"
DTYPES = [torch.bfloat16, torch.float16]
# (num_q_heads, num_kv_heads, head_size) of Llama-3 8B and of gpt-oss-120b; MODEL_SHAPES adds each
# model's (window, sinks): gpt-oss's sliding-window layers use window 128 and attention sinks.
MODEL_HEADS = [(32, 8, 128), (64, 8, 64)]
MODEL_IDS = ["llama3", "gpt-oss"]
MODEL_SHAPES = [(32, 8, 128, -1, False), (64, 8, 64, 128, True)]
# Edge shapes neither model has: one KV head, a GQA group of 3 (padded to 4), no GQA, head size 256
# (32-key tiles).
EDGE_HEADS = [(8, 1, 64), (6, 2, 64), (4, 4, 128), (8, 2, 256)]
EDGE_IDS = ["mqa", "group3", "mha", "d256"]
WINDOWS = [-1, 37, 128]
LENS = [1, 77, 300, 129]


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _cu(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEVICE)


def _varlen(lens, hq, hkv, d, dtype, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(lens)
    q = torch.randn(total, hq, d, generator=gen).to(device=DEVICE, dtype=dtype)
    k = torch.randn(total, hkv, d, generator=gen).to(device=DEVICE, dtype=dtype)
    v = torch.randn(total, hkv, d, generator=gen).to(device=DEVICE, dtype=dtype)
    return q, k, v, _cu(lens)


def _paged(k, v, cu_k, block_size, seed=0, extra_blocks=3):
    """Scatter contiguous K/V into a paged cache with shuffled physical blocks. Unused slots hold
    large garbage so a missing key mask would show."""
    seq_lens = (cu_k[1:] - cu_k[:-1]).tolist()
    blocks_per_seq = [(n + block_size - 1) // block_size for n in seq_lens]
    num_blocks = sum(blocks_per_seq) + extra_blocks
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(num_blocks, generator=gen)
    shape = (num_blocks, block_size, k.shape[1], k.shape[2])
    k_cache = (torch.randn(shape, generator=gen) * 100).to(device=k.device, dtype=k.dtype)
    v_cache = (torch.randn(shape, generator=gen) * 100).to(device=v.device, dtype=v.dtype)
    block_table = torch.zeros(len(seq_lens), max(blocks_per_seq), dtype=torch.int32, device=k.device)
    nxt = 0
    for s, n in enumerate(seq_lens):
        k0 = int(cu_k[s])
        for b in range(blocks_per_seq[s]):
            phys = int(perm[nxt])
            nxt += 1
            block_table[s, b] = phys
            lo, hi = b * block_size, min((b + 1) * block_size, n)
            k_cache[phys, : hi - lo] = k[k0 + lo : k0 + hi]
            v_cache[phys, : hi - lo] = v[k0 + lo : k0 + hi]
    return k_cache, v_cache, block_table, torch.tensor(seq_lens, dtype=torch.int32, device=k.device)


def _fwd(q, k, v, cu, scale, mode="fused", window_left=-1, mask=None, sinks=None, kernel_options=None):
    """Contiguous forward in a given mode, through the same custom op the training path uses."""
    max_len = int((cu[1:] - cu[:-1]).max())
    if mask is not None:
        mask = mask.with_window(window_left)
        window_left = -1
    opts = _resolve_options({"MODE": mode, **(kernel_options or {})})
    out, _ = rl_attention_impl.rl_attention_fwd_impl(
        q, k, v, cu, cu, None, None, max_len, max_len, scale, window_left,
        mask.lo if mask is not None else None, mask.hi if mask is not None else None,
        sinks.float() if sinks is not None else None, False, *opts.fwd_args(),
    )  # fmt: skip
    return out


def _to_global(lo_rel, hi_rel, q_lens, k_starts):
    base = torch.repeat_interleave(torch.tensor(k_starts), torch.tensor(q_lens)).to(DEVICE).int()
    return (lo_rel + base).int(), (hi_rel + base).int()


def _causal_rel(q_lens, k_lens, window_left=-1):
    lo, hi = [], []
    for ql, kl in zip(q_lens, k_lens):
        pos = torch.arange(kl - ql, kl)
        lo.append(torch.clamp(pos - window_left, min=0) if window_left >= 0 else torch.zeros_like(pos))
        hi.append(pos)
    return torch.cat(lo).int().to(DEVICE), torch.cat(hi).int().to(DEVICE)


def _mixed_rel(seq_lens, seed=0):
    """Sequence-relative intervals: a bidirectional span (an image) inside a causal sequence, rows
    with random non-causal ranges, and a few empty rows."""
    gen = torch.Generator().manual_seed(seed)
    lo, hi = [], []
    for n in seq_lens:
        pos = torch.arange(n)
        l, h = torch.zeros(n, dtype=torch.long), pos.clone()
        if n >= 8:
            s = int(torch.randint(0, n // 2, (1,), generator=gen))
            e = min(n, s + max(2, n // 4))
            l[s:e], h[s:e] = 0, e - 1
            for t in torch.randperm(n, generator=gen)[: max(1, n // 8)].tolist():
                a = int(torch.randint(0, n, (1,), generator=gen))
                b = int(torch.randint(a, n, (1,), generator=gen))
                l[t], h[t] = a, b
            empty = torch.randperm(n, generator=gen)[:2]
            l[empty], h[empty] = h[empty] + 1, h[empty]
        lo.append(l)
        hi.append(h)
    return torch.cat(lo).int().to(DEVICE), torch.cat(hi).int().to(DEVICE)


def _mixed_mask(lens, seed=0):
    starts = [0] + torch.tensor(lens).cumsum(0)[:-1].tolist()
    lo, hi = _to_global(*_mixed_rel(lens, seed), lens, starts)
    return lo, hi, create_block_mask(torch.stack([lo, hi]))


def _sinks(hq, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(hq, generator=gen) * 2).to(DEVICE)


def _set_env_options(value):
    envs.set_env("PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS", value)


@pytest.fixture
def small_segments():
    """Short KV segments, set process-wide through the environment as a rollout and a trainer would,
    so split mode and multi-segment fused mode are exercised at test sizes."""
    saved = envs.PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS
    _set_env_options("SEGMENT_SIZE=128")
    yield
    _set_env_options(saved)


# ----------------------------------------------------------------------------- causal / window


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + EDGE_HEADS, ids=MODEL_IDS + EDGE_IDS)
@pytest.mark.parametrize("window", WINDOWS)
def test_causal_matches_reference(dtype, heads, window, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype)
    ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, window_left=window)
    for mode in ("fused", "split"):
        out = _fwd(q, k, v, cu, d**-0.5, mode, window)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + EDGE_HEADS, ids=MODEL_IDS + EDGE_IDS)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_paged_split_fused_contiguous_bitwise(dtype, heads, window, block_size, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=1)
    contiguous = _fwd(q, k, v, cu, d**-0.5, "fused", window)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu, block_size)
    for mode in ("fused", "split"):
        paged = rl_paged_attention_func(
            q,
            k_cache,
            v_cache,
            cu,
            seq_lens,
            block_table,
            300,
            300,
            d**-0.5,
            window_left=window,
            kernel_options={"MODE": mode},
        )
        assert _bits_equal(paged, contiguous), f"paged {mode} (block_size={block_size})"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + EDGE_HEADS[:2], ids=MODEL_IDS + EDGE_IDS[:2])
@pytest.mark.parametrize("with_mask", [False, True])
def test_launch_invariance(dtype, heads, with_mask, small_segments):
    """Rows per program and warps only schedule work: every (BLOCK_M, num_warps) gives the same bits."""
    hq, hkv, d = heads
    lens = [77, 300, 129]
    q, k, v, cu = _varlen(lens, hq, hkv, d, dtype, seed=2)
    mask = _mixed_mask(lens, seed=2)[2] if with_mask else None
    outs = {}
    for block_m in (16, 32, 64, 128):
        for warps in (1, 2, 4, 8):
            options = {"BLOCK_M": block_m, "num_warps": warps}
            outs[(block_m, warps)] = _fwd(q, k, v, cu, d**-0.5, "fused", mask=mask, kernel_options=options)
    for key, out in outs.items():
        assert _bits_equal(out, outs[(16, 2)]), f"(BLOCK_M, num_warps)={key}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
@pytest.mark.parametrize("window", WINDOWS)
def test_batch_invariance(dtype, heads, window, small_segments):
    """A sequence's output is the same alone and packed with other sequences of any length."""
    hq, hkv, d = heads
    q, k, v, cu = _varlen([200], hq, hkv, d, dtype, seed=3)
    alone = _fwd(q, k, v, cu, d**-0.5, window_left=window)
    for others, seed in (([1, 5], 10), ([333, 64, 17], 11), ([129] * 6, 12)):
        oq, ok, ov, _ = _varlen(others, hq, hkv, d, dtype, seed=seed)
        lens = others[:1] + [200] + others[1:]
        split = others[0]
        bq = torch.cat([oq[:split], q, oq[split:]])
        bk = torch.cat([ok[:split], k, ok[split:]])
        bv = torch.cat([ov[:split], v, ov[split:]])
        for mode in ("fused", "split"):
            batched = _fwd(bq, bk, bv, _cu(lens), d**-0.5, mode, window)
            assert _bits_equal(batched[split : split + 200], alone), f"batched with {others} ({mode})"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("block_size", [16, 64])
def test_causal_rollout_matches_training(dtype, heads, window, block_size, small_segments):
    """Rollout: paged prefill of a prompt, then one token per step, batched with another sequence.
    Training: one varlen forward over the whole sequence. Every token must match bitwise."""
    hq, hkv, d = heads
    prompt, generated = 150, 40
    total = prompt + generated
    q, k, v, cu = _varlen([total], hq, hkv, d, dtype, seed=4)
    training = rl_flash_attn_varlen_func(q, k, v, cu, cu, total, total, d**-0.5, window_left=window)
    other_q, other_k, other_v, _ = _varlen([500], hq, hkv, d, dtype, seed=5)
    k_cache, v_cache, block_table, _ = _paged(
        torch.cat([k, other_k]), torch.cat([v, other_v]), _cu([total, 500]), block_size
    )
    for lo, hi in [(0, prompt)] + [(t, t + 1) for t in range(prompt, total)]:
        other_pos = 300 + lo
        step_q = torch.cat([q[lo:hi], other_q[other_pos : other_pos + 1]])
        seq_lens = torch.tensor([hi, other_pos + 1], dtype=torch.int32, device=DEVICE)
        for mode in ("fused", "split"):
            out = rl_paged_attention_func(
                step_q, k_cache, v_cache, _cu([hi - lo, 1]), seq_lens, block_table, hi - lo,
                max(hi, other_pos + 1), d**-0.5, window_left=window, kernel_options={"MODE": mode},
            )  # fmt: skip
            assert _bits_equal(out[: hi - lo], training[lo:hi]), f"tokens [{lo}, {hi}) ({mode})"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=MODEL_IDS)
@pytest.mark.parametrize("chunk", [None, 1000])
def test_model_shape_rollout_matches_training(dtype, shape, chunk):
    """Llama-3 8B and gpt-oss-120b head shapes at the default 1024-key segments: a 2990-token prompt prefilled whole
    or in 1000-token chunks (crossing segment boundaries), then decoded alone and next to another
    long sequence, in every mode. Every token must match one training forward bitwise."""
    hq, hkv, d, window, has_sinks = shape
    prompt, total = 2990, 3000
    q, k, v, cu = _varlen([total], hq, hkv, d, dtype, seed=21)
    sinks = _sinks(hq, 21) if has_sinks else None
    training = rl_flash_attn_varlen_func(q, k, v, cu, cu, total, total, d**-0.5, window, sinks=sinks)
    oq, ok, ov, _ = _varlen([4000], hq, hkv, d, dtype, seed=22)
    k_cache, v_cache, block_table, _ = _paged(torch.cat([k, ok]), torch.cat([v, ov]), _cu([total, 4000]), 64)
    steps = [(0, prompt)] if chunk is None else [(s, min(s + chunk, prompt)) for s in range(0, prompt, chunk)]
    steps += [(t, t + 1) for t in range(prompt, total)]
    for t0, t1 in steps:
        for with_other in (False, True):
            other = 3000 + t0 - prompt if t1 - t0 == 1 else 3500
            step_q = torch.cat([q[t0:t1], oq[other : other + 1]]) if with_other else q[t0:t1]
            lens = [t1, other + 1] if with_other else [t1]
            q_lens = [t1 - t0, 1] if with_other else [t1 - t0]
            for mode in ("auto", "fused", "split"):
                out = rl_paged_attention_func(
                    step_q, k_cache, v_cache, _cu(q_lens), torch.tensor(lens, dtype=torch.int32, device=DEVICE),
                    block_table[: len(lens)], t1 - t0, max(lens), d**-0.5, window_left=window, sinks=sinks,
                    kernel_options={"MODE": mode},
                )  # fmt: skip
                assert _bits_equal(out[: t1 - t0], training[t0:t1]), f"tokens [{t0}, {t1}) other={with_other} {mode}"


# ----------------------------------------------------------------------------- visibility, sinks


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + [(8, 1, 128)], ids=MODEL_IDS + ["mqa-d128"])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_visibility_matches_reference(dtype, heads, with_sinks, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=1)
    lo, hi, mask = _mixed_mask(LENS, seed=1)
    sinks = _sinks(hq) if with_sinks else None
    ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, lo, hi, sinks)
    for mode in ("fused", "split"):
        out = _fwd(q, k, v, cu, d**-0.5, mode, mask=mask, sinks=sinks)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
        assert torch.all(out[lo > hi] == 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
@pytest.mark.parametrize("window", WINDOWS)
def test_causal_intervals_equal_builtin(dtype, heads, window, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=2)
    mask = create_block_mask(*_to_global(*_causal_rel(LENS, LENS, window), LENS, cu[:-1].tolist()))
    for mode in ("fused", "split"):
        builtin = _fwd(q, k, v, cu, d**-0.5, mode, window)
        assert _bits_equal(builtin, _fwd(q, k, v, cu, d**-0.5, mode, mask=mask)), mode


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("with_sinks", [False, True])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_visibility_paged_split_fused_bitwise(dtype, block_size, with_sinks, heads, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=3)
    mask = _mixed_mask(LENS, seed=3)[2]
    sinks = _sinks(hq, 3) if with_sinks else None
    contiguous = _fwd(q, k, v, cu, d**-0.5, mask=mask, sinks=sinks)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu, block_size)
    for mode in ("fused", "split"):
        paged = rl_paged_attention_func(
            q, k_cache, v_cache, cu, seq_lens, block_table, 300, 300, d**-0.5, block_mask=mask, sinks=sinks,
            kernel_options={"MODE": mode},
        )  # fmt: skip
        assert _bits_equal(paged, contiguous), mode


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_paged_out_buffer(dtype, flat, heads, small_segments):
    """out= writes the returned output's bits into a caller's padded buffer, rows past the tokens
    untouched."""
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=15)
    mask = _mixed_mask(LENS, seed=15)[2]
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu, 16)
    rows = q.shape[0] + 5
    for mode in ("fused", "split"):
        args = (q, k_cache, v_cache, cu, seq_lens, block_table, 300, 300, d**-0.5)
        kw = dict(block_mask=mask, kernel_options={"MODE": mode})
        want = rl_paged_attention_func(*args, **kw)
        buf = torch.full((rows, hq, d), 7.0, dtype=dtype, device=DEVICE)
        out = buf.view(rows, hq * d) if flat else buf
        assert rl_paged_attention_func(*args, out=out, **kw) is out
        assert _bits_equal(buf[: q.shape[0]], want), mode
        assert bool((buf[q.shape[0] :] == 7.0).all()), f"padded rows written ({mode})"
    with pytest.raises(ValueError, match="out must be"):
        rl_paged_attention_func(*args, out=torch.empty(q.shape[0] - 1, hq, d, dtype=dtype, device=DEVICE))


def _graph_decode_check(
    hq, hkv, d, page, bound, steps, kernel_options, sinks=None, window_left=-1, seed=0, masked=False
):
    """Capture one decode graph with max_seqlen_k = bound, as a serving engine does, then replay it
    for steps of other context lengths (q and seq_lens updated in place): every replay must give
    the bits of an eager call at the real max_seqlen_k. With ``masked``, the call builds a mask from
    per-sequence bounds (sequence positions, a nonzero lower bound) that are also updated in place,
    narrowed by the window inside the call, so the graph must recompute it from the live values."""
    batch = len(steps[0])
    _, k, v, cu_k = _varlen([bound] * batch, hq, hkv, d, torch.bfloat16, seed=seed)
    k_cache, v_cache, block_table, _ = _paged(k, v, cu_k, page, seed=seed)
    q = torch.zeros(batch, hq, d, dtype=torch.bfloat16, device=DEVICE)
    seq_lens = torch.tensor(steps[0], dtype=torch.int32, device=DEVICE)
    cu_q = _cu([1] * batch)
    rel_lo = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    rel_hi = seq_lens - 1

    def call(max_k):
        mask = None
        if masked:
            mask = create_block_mask(rel_lo, rel_hi, cu_seqlens_q=cu_q, seq_lens=seq_lens)
        return rl_paged_attention_func(
            q, k_cache, v_cache, cu_q, seq_lens, block_table, 1, max_k, d**-0.5, window_left=window_left,
            block_mask=mask, sinks=sinks, kernel_options=kernel_options,
        )  # fmt: skip

    call(bound)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(bound)
    gen = torch.Generator().manual_seed(seed + 1)
    for lens in steps:
        q.copy_(torch.randn(batch, hq, d, generator=gen))
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        rel_hi.copy_(seq_lens - 1)
        rel_lo.copy_(torch.tensor([int(torch.randint(0, n, (1,), generator=gen)) for n in lens], dtype=torch.int32))
        graph.replay()
        assert _bits_equal(out, call(max(lens))), f"graph replay at lens {lens}"


@pytest.mark.parametrize("mode", ["auto", "split"])
@pytest.mark.parametrize("with_sinks", [False, True])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_decode_graph_replay(mode, with_sinks, heads, small_segments):
    """One graph captured at max_seqlen_k = 1024 (eight 128-key segments) serves shorter and
    full-length decode steps with the eager bits."""
    hq, hkv, d = heads
    steps = [[100, 700, 129], [101, 1024, 300], [5, 6, 7]]
    _graph_decode_check(
        hq, hkv, d, 16, 1024, steps, {"MODE": mode}, sinks=_sinks(hq, 19) if with_sinks else None, seed=19
    )


@pytest.mark.parametrize("mode", ["auto", "split"])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_masked_decode_graph_replay(mode, heads, small_segments):
    """A captured graph whose mask (sequence positions, window narrowing) is built inside the call
    from buffers the engine updates in place follows the live bounds on every replay."""
    hq, hkv, d = heads
    steps = [[100, 700, 129], [101, 1024, 300], [5, 6, 7]]
    _graph_decode_check(hq, hkv, d, 16, 1024, steps, {"MODE": mode}, window_left=300, seed=25, masked=True)


@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_masked_paged_under_inference_mode(heads, small_segments):
    """vLLM runs the model under torch.inference_mode(), where tensors carry no version counter:
    masked paged calls (global and sequence-position masks, a window) work there and give the bits
    of the same calls outside it."""
    hq, hkv, d = heads
    lens = [300, 7, 129]
    _, k, v, cu_k = _varlen(lens, hq, hkv, d, torch.bfloat16, seed=26)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu_k, 16, seed=26)
    q = _varlen([1] * len(lens), hq, hkv, d, torch.bfloat16, seed=27)[0]
    rlo = torch.tensor([37, 0, 100], dtype=torch.int32, device=DEVICE)
    rhi = torch.tensor(lens, dtype=torch.int32, device=DEVICE) - 1

    def run(q, seq_lens, rlo, rhi):
        cu_q = _cu([1] * len(lens))
        rel = create_block_mask(rlo, rhi, cu_seqlens_q=cu_q, seq_lens=seq_lens)
        glob = create_block_mask(rel.lo.clone(), rel.hi.clone())
        args = (q, k_cache, v_cache, cu_q, seq_lens, block_table, 1, max(lens), d**-0.5)
        return [rl_paged_attention_func(*args, block_mask=m, window_left=w) for m in (rel, glob) for w in (-1, 50)]

    want = run(q, seq_lens, rlo, rhi)
    with torch.inference_mode():
        got = run(q.clone(), seq_lens.clone(), rlo.clone(), rhi.clone())
    for i, (g, w) in enumerate(zip(got, want)):
        assert _bits_equal(g, w), f"call {i} under inference_mode"


def test_sequence_position_mask():
    """Bounds in each token's own sequence positions give the global mask, for contiguous K/V and
    for a paged cache whose sequences have more keys than queries."""
    rlo, rhi = _mixed_rel(LENS, seed=16)
    cu = _cu(LENS)
    glo, ghi = _to_global(rlo, rhi, LENS, cu[:-1].tolist())
    mask = create_block_mask(rlo, rhi, cu_seqlens_q=cu, cu_seqlens_k=cu)
    assert torch.equal(mask.lo, glo) and torch.equal(mask.hi, ghi)
    q_lens, k_lens = [1, 3, 1], [40, 7, 129]
    rlo, rhi = _causal_rel(q_lens, k_lens, window_left=5)
    seq_lens = torch.tensor(k_lens, dtype=torch.int32, device=DEVICE)
    starts = [0] + torch.tensor(k_lens).cumsum(0)[:-1].tolist()
    glo, ghi = _to_global(rlo, rhi, q_lens, starts)
    mask = create_block_mask(torch.stack([rlo, rhi]), cu_seqlens_q=_cu(q_lens), seq_lens=seq_lens)
    assert torch.equal(mask.lo, glo) and torch.equal(mask.hi, ghi)
    with pytest.raises(ValueError, match="exactly one"):
        create_block_mask(rlo, rhi, cu_seqlens_q=_cu(q_lens))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_visibility_batch_invariance(dtype, heads, small_segments):
    hq, hkv, d = heads
    q, k, v, cu = _varlen([200], hq, hkv, d, dtype, seed=4)
    rlo, rhi = _mixed_rel([200], seed=4)
    sinks = _sinks(hq, 4)
    alone = _fwd(q, k, v, cu, d**-0.5, mask=create_block_mask(rlo, rhi), sinks=sinks)
    others = [333, 64, 17]
    oq, ok, ov, _ = _varlen(others, hq, hkv, d, dtype, seed=5)
    orlo, orhi = _mixed_rel(others, seed=5)
    split = others[0]
    lens = [others[0], 200] + others[1:]
    bq = torch.cat([oq[:split], q, oq[split:]])
    bk = torch.cat([ok[:split], k, ok[split:]])
    bv = torch.cat([ov[:split], v, ov[split:]])
    bcu = _cu(lens)
    blo, bhi = _to_global(
        torch.cat([orlo[:split], rlo, orlo[split:]]), torch.cat([orhi[:split], rhi, orhi[split:]]),
        lens, bcu[:-1].tolist(),
    )  # fmt: skip
    for mode in ("fused", "split"):
        batched = _fwd(bq, bk, bv, bcu, d**-0.5, mode, mask=create_block_mask(blo, bhi), sinks=sinks)
        assert _bits_equal(batched[split : split + 200], alone), mode


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("chunk", [None, 37])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_visibility_rollout_matches_training(dtype, block_size, chunk, heads, small_segments):
    """Prompt with a bidirectional image span, prefilled at once or in 37-token chunks, then causal
    decode one token per step next to another sequence; training runs one varlen forward with the
    same intervals and sinks. Every token must match bitwise."""
    hq, hkv, d = heads
    prompt, generated = 150, 30
    total = prompt + generated
    q, k, v, cu = _varlen([total], hq, hkv, d, dtype, seed=6)
    sinks = _sinks(hq, 6)
    lo = torch.zeros(total, dtype=torch.int32, device=DEVICE)
    hi = torch.arange(total, dtype=torch.int32, device=DEVICE)
    hi[40:90] = 89
    training = rl_flash_attn_varlen_func(
        q, k, v, cu, cu, total, total, d**-0.5, block_mask=create_block_mask(lo, hi), sinks=sinks
    )
    other_q, other_k, other_v, _ = _varlen([500], hq, hkv, d, dtype, seed=7)
    k_cache, v_cache, block_table, _ = _paged(
        torch.cat([k, other_k]), torch.cat([v, other_v]), _cu([total, 500]), block_size
    )
    steps = [(0, prompt)] if chunk is None else [(s, min(s + chunk, prompt)) for s in range(0, prompt, chunk)]
    steps += [(t, t + 1) for t in range(prompt, total)]
    for t0, t1 in steps:
        other_pos = 300 + t0
        keys_known = max(t1, int(hi[t0:t1].max()) + 1)
        step_q = torch.cat([q[t0:t1], other_q[other_pos : other_pos + 1]])
        # global key indices of this step's batch: this sequence's keys, then the other sequence's
        other = torch.tensor([keys_known, keys_known + other_pos], dtype=torch.int32, device=DEVICE)
        step_mask = create_block_mask(torch.cat([lo[t0:t1], other[:1]]), torch.cat([hi[t0:t1], other[1:]]))
        seq_lens = torch.tensor([keys_known, other_pos + 1], dtype=torch.int32, device=DEVICE)
        for mode in ("fused", "split"):
            out = rl_paged_attention_func(
                step_q, k_cache, v_cache, _cu([t1 - t0, 1]), seq_lens, block_table, t1 - t0,
                max(keys_known, other_pos + 1), d**-0.5, block_mask=step_mask, sinks=sinks, kernel_options={"MODE": mode},
            )  # fmt: skip
            assert _bits_equal(out[: t1 - t0], training[t0:t1]), f"tokens [{t0}, {t1}) ({mode})"


def test_window_folding():
    lo = torch.tensor([0, 0, 3, 5], dtype=torch.int32)
    hi = torch.tensor([4, 9, 7, 4], dtype=torch.int32)
    mask = create_block_mask(torch.stack([lo, hi]), window_left=2)
    assert mask.lo.tolist() == [2, 7, 5, 5] and mask.hi.tolist() == [4, 9, 7, 4]


def test_validate_rejects_foreign_keys():
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    good = create_block_mask(
        torch.tensor([0, 0, 2, 5, 4, 4, 4, 4, 4, 4]), torch.tensor([0, 1, 3, 4, 4, 5, 6, 7, 8, 9])
    )
    good.validate(cu, cu)
    bad = create_block_mask(
        torch.tensor([0, 0, 0, 0, 3, 4, 4, 4, 4, 4]), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    with pytest.raises(ValueError):
        bad.validate(cu, cu)


# ----------------------------------------------------------------------------- backward


def _grads(q, k, v, cu, max_len, scale, dout, sinks=None, **kw):
    qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
    ss = sinks.clone().requires_grad_() if sinks is not None else None
    out = rl_flash_attn_varlen_func(qq, kk, vv, cu, cu, max_len, max_len, scale, sinks=ss, **kw)
    out.backward(dout)
    return (out.detach(), qq.grad, kk.grad, vv.grad) + ((ss.grad,) if ss is not None else ())


def _ref_grads(q, k, v, cu, scale, dout, sinks=None, **kw):
    qr, kr, vr = (t.float().clone().requires_grad_() for t in (q, k, v))
    sr = sinks.float().clone().requires_grad_() if sinks is not None else None
    rl_attention_ref(qr, kr, vr, cu, cu, scale, sinks=sr, **kw).backward(dout.float())
    return (qr.grad, kr.grad, vr.grad) + ((sr.grad,) if sr is not None else ())


def _assert_deterministic(a, b):
    for x, y in zip(a, b):
        assert torch.equal(x, y), "backward is not deterministic"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + [(4, 4, 128)], ids=MODEL_IDS + ["mha"])
@pytest.mark.parametrize("window", WINDOWS)
def test_causal_backward(dtype, heads, window):
    hq, hkv, d = heads
    q, k, v, cu = _varlen([33, 200, 129], hq, hkv, d, dtype, seed=6)
    dout = torch.randn(q.shape[0], hq, d, generator=torch.Generator().manual_seed(7)).to(DEVICE, dtype)
    got = _grads(q, k, v, cu, 200, d**-0.5, dout, window_left=window)
    _assert_deterministic(got, _grads(q, k, v, cu, 200, d**-0.5, dout, window_left=window))
    want = _ref_grads(q, k, v, cu, d**-0.5, dout, window_left=window)
    for name, g, w in zip(("dq", "dk", "dv"), got[1:], want):
        torch.testing.assert_close(g.float(), w, atol=5e-2, rtol=5e-2, msg=name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", MODEL_HEADS + [(4, 4, 128)], ids=MODEL_IDS + ["mha"])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_visibility_backward(dtype, heads, with_sinks):
    hq, hkv, d = heads
    lens = [33, 200, 129]
    q, k, v, cu = _varlen(lens, hq, hkv, d, dtype, seed=9)
    lo, hi, mask = _mixed_mask(lens, seed=9)
    sinks = _sinks(hq, 9) if with_sinks else None
    dout = torch.randn(q.shape[0], hq, d, generator=torch.Generator().manual_seed(10)).to(DEVICE, dtype)
    got = _grads(q, k, v, cu, 200, d**-0.5, dout, sinks, block_mask=mask)
    _assert_deterministic(got, _grads(q, k, v, cu, 200, d**-0.5, dout, sinks, block_mask=mask))
    want = _ref_grads(q, k, v, cu, d**-0.5, dout, sinks, lo=lo, hi=hi)
    for name, g, w in zip(("dq", "dk", "dv", "dsink"), got[1:], want):
        torch.testing.assert_close(g.float(), w, atol=5e-2, rtol=5e-2, msg=name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_dsink_fp32_accuracy(dtype, with_mask, heads):
    """dsink takes each row's do . o in fp32 from the backward's own P and dP, not from the rounded
    forward output (which costs ~1e-3 relative error in bf16)."""
    hq, hkv, d = heads
    lens = [33, 200, 129]
    q, k, v, cu = _varlen(lens, hq, hkv, d, dtype, seed=13)
    sinks = _sinks(hq, 13)
    dout = torch.randn(q.shape[0], hq, d, generator=torch.Generator().manual_seed(14)).to(DEVICE, dtype)
    if with_mask:
        lo, hi, mask = _mixed_mask(lens, seed=13)
        got = _grads(q, k, v, cu, 200, d**-0.5, dout, sinks, block_mask=mask)[-1]
        want = _ref_grads(q, k, v, cu, d**-0.5, dout, sinks, lo=lo, hi=hi)[-1]
    else:
        got = _grads(q, k, v, cu, 200, d**-0.5, dout, sinks)[-1]
        want = _ref_grads(q, k, v, cu, d**-0.5, dout, sinks)[-1]
    err = ((got.float() - want).norm() / want.norm()).item()
    assert err < 1e-4, f"dsink relative error {err:.2e}"


@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_visibility_backward_sparse_intervals(heads):
    """Short intervals scattered over long sequences, so the pruned dK/dV and dQ walks skip most
    tiles."""
    hq, hkv, d = heads
    lens = [700, 300]
    q, k, v, cu = _varlen(lens, hq, hkv, d, torch.bfloat16, seed=11)
    gen = torch.Generator().manual_seed(12)
    lo_rel, hi_rel = [], []
    for n in lens:
        start = torch.randint(0, n - 40, (n,), generator=gen)
        lo_rel.append(start)
        hi_rel.append(start + torch.randint(0, 40, (n,), generator=gen))
    lo, hi = _to_global(
        torch.cat(lo_rel).int().to(DEVICE), torch.cat(hi_rel).int().to(DEVICE), lens, [0, 700]
    )
    dout = torch.randn(q.shape[0], hq, d, generator=torch.Generator().manual_seed(13)).to(
        DEVICE, torch.bfloat16
    )
    got = _grads(q, k, v, cu, 700, d**-0.5, dout, block_mask=create_block_mask(lo, hi))
    want = _ref_grads(q, k, v, cu, d**-0.5, dout, lo=lo, hi=hi)
    for name, g, w in zip(("dq", "dk", "dv"), got[1:], want):
        torch.testing.assert_close(g.float(), w, atol=5e-2, rtol=5e-2, msg=name)


def test_bshd_matches_varlen():
    hq, hkv, d = MODEL_HEADS[0]
    q, k, v, cu = _varlen([96, 96], hq, hkv, d, torch.bfloat16, seed=8)
    varlen = rl_flash_attn_varlen_func(q, k, v, cu, cu, 96, 96)
    bshd = rl_flash_attn_func(q.view(2, 96, hq, d), k.view(2, 96, hkv, d), v.view(2, 96, hkv, d))
    assert _bits_equal(bshd.reshape_as(varlen), varlen)


# ----------------------------------------------------------------------------- kernel options


def test_kernel_options_validation():
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_options({"BLOCK_N": 64})
    with pytest.raises(ValueError, match="MODE"):
        _resolve_options({"MODE": "fast"})
    with pytest.raises(ValueError, match="BACKEND"):
        _resolve_options({"BACKEND": "CK"})
    with pytest.raises(ValueError, match="integer"):
        _resolve_options({"TILE_SIZE": "big"})
    q, k, v, cu = _varlen([64], *MODEL_HEADS[0], torch.bfloat16)
    with pytest.raises(ValueError, match="TILE_SIZE"):
        rl_flash_attn_varlen_func(q, k, v, cu, cu, 64, 64, kernel_options={"TILE_SIZE": 48})


def test_env_options_and_per_call_override():
    """The environment sets defaults; kernel_options overrides them key by key; equal settings
    through either route give the same bits, and the reduction order does change them."""
    q, k, v, cu = _varlen(LENS, *MODEL_HEADS[0], torch.bfloat16, seed=14)
    saved = envs.PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS
    try:
        _set_env_options("TILE_SIZE=32, SEGMENT_SIZE=128,MODE=split")
        opts = _resolve_options({"MODE": "fused"})
        assert (opts.tile_size, opts.segment_size, opts.mode) == (32, 128, "fused")
        from_env = rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300)
        _set_env_options(None)
        per_call = rl_flash_attn_varlen_func(
            q, k, v, cu, cu, 300, 300, kernel_options={"TILE_SIZE": 32, "SEGMENT_SIZE": 128}
        )
        default = rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300)
        assert _bits_equal(from_env, per_call)
        assert not _bits_equal(per_call, default)
        _set_env_options("TILE_SIZE")
        with pytest.raises(ValueError, match="KEY=VALUE"):
            rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300)
    finally:
        _set_env_options(saved)


# ----------------------------------------------------------------------------- flash_attn_* interface


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", [-1, 37])
def test_flash_attn_varlen_func_routes_to_rl_kernel(dtype, window, small_segments):
    """flash_attn_varlen_func(batch_invariant=True) is the RL kernel: same bits as
    rl_flash_attn_varlen_func, LSE in the varlen layout [Hq, total_q], and gradients flow."""
    hq, hkv, d = MODEL_HEADS[0]
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=15)
    want, want_lse = rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300, window_left=window, return_lse=True)
    qq = q.clone().requires_grad_()
    got, lse = flash_attn_varlen_func(
        qq,
        k,
        v,
        cu,
        cu,
        300,
        300,
        causal=True,
        window_size=(window, 0),
        return_lse=True,
        batch_invariant=True,
    )
    assert _bits_equal(got, want)
    assert lse.shape == (hq, q.shape[0]) and torch.equal(lse, want_lse.t())
    got.float().sum().backward()
    assert qq.grad is not None and torch.isfinite(qq.grad).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("heads", MODEL_HEADS, ids=MODEL_IDS)
def test_flash_attn_varlen_func_paged_rollout_matches_training(dtype, block_size, heads, small_segments):
    """Both sides through the normal interface: paged decode with block_table / seqused_k
    (rollout) and a varlen forward (training) give the same bits, with a mask and sinks."""
    hq, hkv, d = heads
    prompt, total = 100, 120
    q, k, v, cu = _varlen([total], hq, hkv, d, dtype, seed=16)
    sinks = _sinks(hq, 16)
    lo = torch.zeros(total, dtype=torch.int32, device=DEVICE)
    hi = torch.arange(total, dtype=torch.int32, device=DEVICE)
    hi[20:60] = 59
    training = flash_attn_varlen_func(
        q, k, v, cu, cu, total, total, sink=sinks, block_mask=create_block_mask(lo, hi)
    )
    k_cache, v_cache, block_table, _ = _paged(k, v, cu, block_size)
    for t0, t1 in [(0, prompt)] + [(t, t + 1) for t in range(prompt, total)]:
        out = flash_attn_varlen_func(
            q[t0:t1], k_cache, v_cache, _cu([t1 - t0]), None, t1 - t0, t1, sink=sinks,
            block_mask=create_block_mask(lo[t0:t1], hi[t0:t1]), block_table=block_table,
            seqused_k=torch.tensor([t1], dtype=torch.int32, device=DEVICE),
        )  # fmt: skip
        assert _bits_equal(out, training[t0:t1]), f"tokens [{t0}, {t1})"


@pytest.mark.parametrize("dtype", DTYPES)
def test_flash_attn_func_routes_to_rl_kernel(dtype):
    hq, hkv, d = MODEL_HEADS[0]
    q, k, v, cu = _varlen([96, 96], hq, hkv, d, dtype, seed=17)
    shape = lambda t, h: t.view(2, 96, h, d)  # noqa: E731
    want, want_lse = rl_flash_attn_func(shape(q, hq), shape(k, hkv), shape(v, hkv), return_lse=True)
    got, lse = flash_attn_func(
        shape(q, hq), shape(k, hkv), shape(v, hkv), causal=True, return_lse=True, batch_invariant=True
    )
    assert _bits_equal(got, want)
    assert lse.shape == (2, hq, 96) and torch.equal(lse, want_lse.permute(0, 2, 1))
    assert _bits_equal(got.reshape(-1, hq, d), rl_flash_attn_varlen_func(q, k, v, cu, cu, 96, 96))


def test_env_switch_routes_flash_attn():
    hq, hkv, d = MODEL_HEADS[0]
    q, k, v, cu = _varlen(LENS, hq, hkv, d, torch.bfloat16, seed=18)
    want = rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300)
    saved = envs.PRIMUS_TURBO_ATTN_BATCH_INVARIANT
    try:
        envs.set_env("PRIMUS_TURBO_ATTN_BATCH_INVARIANT", "1")
        got = flash_attn_varlen_func(q, k, v, cu, cu, 300, 300, causal=True)
    finally:
        envs.set_env("PRIMUS_TURBO_ATTN_BATCH_INVARIANT", "1" if saved else None)
    assert _bits_equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_flash_attn_non_causal(dtype):
    """causal=False is full attention over each sequence's keys, expressed as intervals."""
    hq, hkv, d = MODEL_HEADS[0]
    q, k, v, cu = _varlen(LENS, hq, hkv, d, dtype, seed=19)
    starts = torch.repeat_interleave(cu[:-1], cu[1:] - cu[:-1])
    ends = torch.repeat_interleave(cu[1:] - 1, cu[1:] - cu[:-1])
    ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, starts, ends)
    out = flash_attn_varlen_func(q, k, v, cu, cu, 300, 300, causal=False, batch_invariant=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_flash_attn_rejects_unsupported():
    q, k, v, cu = _varlen([64], *MODEL_HEADS[0], torch.bfloat16)
    with pytest.raises(ValueError, match="dropout"):
        flash_attn_varlen_func(q, k, v, cu, cu, 64, 64, dropout_p=0.1, causal=True, batch_invariant=True)
    with pytest.raises(ValueError, match="window"):
        flash_attn_varlen_func(q, k, v, cu, cu, 64, 64, causal=True, window_size=(8, 8), batch_invariant=True)
    with pytest.raises(ValueError, match="seqused_k"):
        flash_attn_varlen_func(q, k, v, cu, None, 64, 64, causal=True, block_table=cu[None, :1])
