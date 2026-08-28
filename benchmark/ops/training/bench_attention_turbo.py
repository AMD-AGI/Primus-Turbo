###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import (
    BATCH_SIZE_LIST,
    compute_snr,
    gen_attention_test_cases,
    gen_attention_varlen_test_cases,
    get_platform_info,
)
from tabulate import tabulate
from torch.nn.attention import SDPBackend, sdpa_kernel


# Disable FP32 atomic for better performance on gfx950
def _is_gfx950():
    """Check if current GPU is gfx950 using torch."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 5


if _is_gfx950():
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "0"

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)

# Flash-attn selects its backend on the bf16/fp16/fp32 precision bucket.
_ATTN_PRECISION = PrecisionType.BF16_FP16_FP32

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# PyTorch SDPA backends for reference implementation
ATTN_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _bottom_right_mask(seqlen_q, seqlen_kv, window_left, device):
    """Bool [Sq, Skv]: bottom-right causal, optionally bounded on the left by a window."""
    i = torch.arange(seqlen_q, device=device).view(seqlen_q, 1)
    j = torch.arange(seqlen_kv, device=device).view(1, seqlen_kv)
    offset = seqlen_kv - seqlen_q
    keep = j <= i + offset
    if window_left >= 0:
        keep &= j >= (i + offset - window_left)  # the left-edge column is kept
    return keep


def _attended_frac(seqlen_q, seqlen_kv, window_left):
    """Fraction of the Sq x Skv score block the mask keeps, for the FLOPs count."""
    i = torch.arange(seqlen_q)
    offset = seqlen_kv - seqlen_q
    hi = torch.clamp(i + offset + 1, max=seqlen_kv)
    lo = torch.zeros_like(i) if window_left < 0 else torch.clamp(i + offset - window_left, min=0)
    return float((hi - lo).clamp(min=0).sum()) / (seqlen_q * seqlen_kv)


def attention_ref(q, k, v, sm_scale, causal, window_left=-1):
    """Reference attention using PyTorch's scaled_dot_product_attention.

    A window or a rectangular shape needs an explicit mask: is_causal is top-left aligned,
    which only coincides with the kernels' bottom-right masking while Sq == Skv, and it
    cannot express a left window at all. Square unwindowed shapes keep is_causal.
    """
    num_heads = q.shape[2]
    n_kv_heads = k.shape[2]
    n_rep = num_heads // n_kv_heads

    # BSHD -> BHSD
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    mask = None
    if causal and (window_left >= 0 or q.shape[2] != k.shape[2]):
        mask = _bottom_right_mask(q.shape[2], k.shape[2], window_left, q.device)

    with sdpa_kernel(ATTN_BACKENDS):
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal and mask is None,
            scale=sm_scale,
            enable_gqa=n_rep > 1,
        )

    # BHSD -> BSHD
    return o_ref.transpose(1, 2)


def check_attention_correctness(q, k, v, q_ref, k_ref, v_ref, o, o_ref, grad_out, use_fp8):
    """Check correctness of attention forward and backward against PyTorch reference."""
    # Backward pass
    o_ref.backward(grad_out, retain_graph=True)
    o.backward(grad_out, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(o_ref, o)
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)

    # SNR thresholds: bf16 requires higher SNR (40), fp8 allows lower (20)
    threshold = 20 if use_fp8 else 40

    correct = all(snr > threshold for snr in [out_snr, dq_snr, dk_snr, dv_snr])
    status = "PASS" if correct else "FAIL"
    print(
        f"Correctness Check (SNR>thr={threshold} vs torch-ref): "
        f"{status} (out={out_snr:.1f}, dq={dq_snr:.1f}, dk={dk_snr:.1f}, dv={dv_snr:.1f})"
    )

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    return correct


def check_attention_forward_correctness(o, o_ref):
    """Check forward-only output correctness against the PyTorch reference."""
    out_snr = compute_snr(o_ref, o)
    threshold = 40
    correct = out_snr > threshold
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check (SNR>thr={threshold} vs torch-ref): {status} (out={out_snr:.1f})")
    return correct


def check_attention_determinism(fwd_func, q, k, v, grad_out):
    """Check deterministic for forward outputs and backward gradients (bitwise exact)."""
    # Forward: same input -> same output
    out1 = fwd_func()
    out2 = fwd_func()
    torch.cuda.synchronize()

    out_ok = torch.equal(out1, out2)
    out_max_abs_diff = (out1 - out2).abs().max().item()

    # Backward: same input + same grad_out -> same grads
    q.grad = None
    k.grad = None
    v.grad = None

    out1_bwd = fwd_func()
    out1_bwd.backward(grad_out, retain_graph=False)
    dq1 = q.grad.detach().clone()
    dk1 = k.grad.detach().clone()
    dv1 = v.grad.detach().clone()

    q.grad = None
    k.grad = None
    v.grad = None

    out2_bwd = fwd_func()
    out2_bwd.backward(grad_out, retain_graph=False)
    dq2 = q.grad.detach().clone()
    dk2 = k.grad.detach().clone()
    dv2 = v.grad.detach().clone()
    torch.cuda.synchronize()

    dq_ok = torch.equal(dq1, dq2)
    dk_ok = torch.equal(dk1, dk2)
    dv_ok = torch.equal(dv1, dv2)

    dq_max_abs_diff = (dq1 - dq2).abs().max().item()
    dk_max_abs_diff = (dk1 - dk2).abs().max().item()
    dv_max_abs_diff = (dv1 - dv2).abs().max().item()

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    return (
        out_ok,
        out_max_abs_diff,
        dq_ok,
        dq_max_abs_diff,
        dk_ok,
        dk_max_abs_diff,
        dv_ok,
        dv_max_abs_diff,
    )


def _make_qkv(shape, layout, device, dtype, requires_grad: bool = True):
    """A [b, s, h, d]-shaped leaf whose bytes are in ``layout`` order.

    Non-BSHD layouts are allocated in their physical order and handed to the public op as
    logical [b, s, h, d] views. requires_grad_ comes after the view so the view itself is the
    leaf that collects .grad in the training benchmark.
    """
    b, s, h, d = shape
    if layout == "sbhd":
        tensor = torch.randn((s, b, h, d), device=device, dtype=dtype).permute(1, 0, 2, 3)
    elif layout == "bhsd":
        tensor = torch.randn((b, h, s, d), device=device, dtype=dtype).transpose(1, 2)
    elif layout == "bshd":
        tensor = torch.randn(shape, device=device, dtype=dtype)
    else:
        raise ValueError(f"unsupported layout: {layout}")
    return tensor.requires_grad_() if requires_grad else tensor


def profile_attention(
    batch,
    seqlen,
    num_head_q,
    num_head_kv,
    head_dim_qk,
    head_dim_v,
    causal,
    use_fp8,
    deterministic,
    seqlen_kv=None,
    window_left=-1,
    layout="bshd",
    forward_only: bool = False,
    dtype: str = "bf16",
):
    """Profile attention, optionally measuring a graph-free forward only."""
    if forward_only and use_fp8:
        raise ValueError("forward-only mode does not support FP8")
    try:
        torch_dtype = _DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f"unsupported dtype: {dtype}") from e

    device = "cuda"
    seqlen_kv = seqlen if seqlen_kv is None else seqlen_kv
    window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
    # NOTE: do not empty_cache between cases. Handing the cached blocks back makes the
    # allocator re-acquire large contiguous ones from the driver, which fragments worse than
    # reusing them: --deterministic goes from 10 OOM cases to 18 with one call here.

    needs_backward = not forward_only
    q = _make_qkv(
        (batch, seqlen, num_head_q, head_dim_qk),
        layout,
        device,
        torch_dtype,
        requires_grad=needs_backward,
    )
    k = _make_qkv(
        (batch, seqlen_kv, num_head_kv, head_dim_qk),
        layout,
        device,
        torch_dtype,
        requires_grad=needs_backward,
    )
    v = _make_qkv(
        (batch, seqlen_kv, num_head_kv, head_dim_v),
        layout,
        device,
        torch_dtype,
        requires_grad=needs_backward,
    )

    sm_scale = head_dim_qk ** (-0.5)

    # Which backend this shape resolves to -- worth reporting, since eligibility turns on
    # the head dims, the GQA group and storage order. Resolve before any implementation
    # launch so a pinned backend cannot silently turn an unsupported shape into fallback data.
    backend = "FP8/TRITON"
    if not use_fp8:

        def resolve_backend():
            return resolve_flash_attn_backend(
                varlen=False,
                user_backend=GlobalBackendManager.get_attn_backend(_ATTN_PRECISION),
                q=q,
                k=k,
                v=v,
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=causal,
                window_size=window_size,
                bias=None,
                alibi_slopes=None,
                sink=None,
                qkv_format=layout,
                return_softmax=False,
                needs_backward=needs_backward,
            )

        try:
            if forward_only:
                with torch.inference_mode():
                    backend = resolve_backend().name
            else:
                backend = resolve_backend().name
        except ValueError as e:  # a forced backend that cannot take this shape
            raise RuntimeError(f"backend rejected the shape: {e}") from e

    # Take the reference after backend resolution. The training graph and the measured one
    # would otherwise be alive at once, which costs the long shapes nine more OOM cases under
    # --deterministic. A masked SDPA may materialise [b, h, Sq, Skv] scores, so a reference
    # that does not fit downgrades the correctness check to SKIP.
    q_ref = q.clone().detach()
    k_ref = k.clone().detach()
    v_ref = v.clone().detach()
    if needs_backward:
        q_ref.requires_grad_()
        k_ref.requires_grad_()
        v_ref.requires_grad_()
    try:
        if forward_only:
            with torch.inference_mode():
                o_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal, window_left)
        else:
            o_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal, window_left)
    except torch.cuda.OutOfMemoryError:
        o_ref = None
        q_ref = k_ref = v_ref = None
        torch.cuda.empty_cache()

    # Define forward function
    if use_fp8:

        def raw_fwd_func():
            return turbo.ops.flash_attn_fp8_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=causal,
                window_size=window_size,
                bias=None,
                alibi_slopes=None,
                deterministic=deterministic,
                return_lse=False,
                return_attn_probs=False,
            )
    else:

        def raw_fwd_func():
            return turbo.ops.flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=causal,
                window_size=window_size,
                bias=None,
                alibi_slopes=None,
                deterministic=deterministic,
                return_lse=False,
                return_attn_probs=False,
            )

    # Decorating the callable keeps every warmup and Timer invocation in inference mode,
    # rather than relying on a context that may have ended before Timer calls it.
    fwd_func = torch.inference_mode()(raw_fwd_func) if forward_only else raw_fwd_func

    det_out_ok = None
    det_out_max_abs_diff = None
    det_dq_ok = None
    det_dq_max_abs_diff = None
    det_dk_ok = None
    det_dk_max_abs_diff = None
    det_dv_ok = None
    det_dv_max_abs_diff = None
    if deterministic and forward_only:
        out1 = fwd_func()
        out2 = fwd_func()
        torch.cuda.synchronize()
        det_out_ok = torch.equal(out1, out2)
        det_out_max_abs_diff = (out1 - out2).abs().max().item()
        del out1, out2
    # Forward pass and correctness check
    correct = None
    if forward_only:
        if o_ref is not None:
            out = fwd_func()
            correct = check_attention_forward_correctness(out, o_ref)
            del out, o_ref, q_ref, k_ref, v_ref
    else:
        # Fixed grad_out for deterministic / correctness checks. This allocation and every
        # backward call remain entirely outside the forward-only control-flow path.
        out_for_grad = fwd_func()
        grad_out = torch.randn_like(out_for_grad)
        del out_for_grad
        if deterministic:
            (
                det_out_ok,
                det_out_max_abs_diff,
                det_dq_ok,
                det_dq_max_abs_diff,
                det_dk_ok,
                det_dk_max_abs_diff,
                det_dv_ok,
                det_dv_max_abs_diff,
            ) = check_attention_determinism(fwd_func, q, k, v, grad_out)

        out = fwd_func()
        if o_ref is not None:
            try:
                correct = check_attention_correctness(
                    q, k, v, q_ref, k_ref, v_ref, out, o_ref, grad_out, use_fp8
                )
            except torch.cuda.OutOfMemoryError:
                print("Correctness Check: SKIP (the reference's backward does not fit)")
            # The reference holds a second copy of q/k/v and their grads; the timing below
            # needs none of it, and the long shapes cannot afford to keep it.
            del o_ref, q_ref, k_ref, v_ref

    # Print deterministic status only when deterministic is enabled
    if deterministic:
        determinism_ok = bool(det_out_ok)
        if not forward_only:
            determinism_ok = determinism_ok and bool(det_dq_ok) and bool(det_dk_ok) and bool(det_dv_ok)
        status = "PASS" if determinism_ok else "FAIL"
        if forward_only:
            print(f"Deterministic Check (bitwise; max_abs_diff): {status} (out={det_out_max_abs_diff})")
        else:
            print(
                "Deterministic Check (bitwise; max_abs_diff): "
                f"{status} (out={det_out_max_abs_diff}, dq={det_dq_max_abs_diff}, "
                f"dk={det_dk_max_abs_diff}, dv={det_dv_max_abs_diff})"
            )

    if forward_only and correct is not True:
        timing_status = "SKIP" if correct is None else "FAIL"
        print(f"Forward timing: {timing_status} (correctness did not pass)")
        return (
            backend,
            None,
            None,
            None,
            None,
            correct,
            det_out_ok,
            det_out_max_abs_diff,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # Calculate FLOPs. A window or a rectangular shape needs the real attended fraction;
    # square full-causal keeps the plain halving, which is that fraction to within a row.
    fwd_flops = 2 * batch * seqlen * seqlen_kv * num_head_q * (head_dim_qk + head_dim_v)
    if causal:
        if window_left >= 0 or seqlen != seqlen_kv:
            fwd_flops = int(fwd_flops * _attended_frac(seqlen, seqlen_kv, window_left))
        else:
            fwd_flops //= 2

    if forward_only:
        for _ in range(20):
            fwd_func()
        torch.cuda.synchronize()
        fwd_time = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(100).mean * 1e3
        torch.cuda.synchronize()
        fwd_tflops = fwd_flops / (fwd_time * 1e-3) / 1e12
        print(f"Forward (warm-cache) Mean time: {fwd_time:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
        return (
            backend,
            fwd_time,
            fwd_tflops,
            None,
            None,
            correct,
            det_out_ok,
            det_out_max_abs_diff,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # Prepare and benchmark the original forward+backward training path.
    out = fwd_func()
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    bwd_func()
    bwd_flops = fwd_flops * 2.5

    # Warmup
    for _ in range(20):
        fwd_func()
        bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    fwd_time = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(100).mean * 1e3
    bwd_time = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(100).mean * 1e3
    fwd_tflops = fwd_flops / (fwd_time * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_time * 1e-3) / 1e12

    print(f"Forward  Mean time: {fwd_time:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_time:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return (
        backend,
        fwd_time,
        fwd_tflops,
        bwd_time,
        bwd_tflops,
        correct,
        det_out_ok,
        det_out_max_abs_diff,
        det_dq_ok,
        det_dq_max_abs_diff,
        det_dk_ok,
        det_dk_max_abs_diff,
        det_dv_ok,
        det_dv_max_abs_diff,
    )


def profile_varlen(segments, window_left, num_head_q, num_head_kv, head_dim, backend_enum):
    """Profile the THD (document-packing) backward for one segment layout.

    The high-level dispatcher only routes UNIFORM cu_seqlens to FlyDSL, so a ragged layout
    on that backend is driven through the impl layer -- which is the path a packed batch
    actually takes and the one worth timing. Every other backend goes through the op.
    """
    device, dtype = "cuda", torch.bfloat16
    sm_scale = head_dim ** (-0.5)
    window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    cu = torch.zeros(len(segments) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(segments, device=device, dtype=torch.int32), 0)
    max_seqlen, total = max(segments), int(cu[-1].item())

    q = torch.randn(total, num_head_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, num_head_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, num_head_kv, head_dim, device=device, dtype=dtype)
    do = torch.randn(total, num_head_q, head_dim, device=device, dtype=dtype)

    if backend_enum == BackendType.FLYDSL:
        from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
            flash_attn_varlen_flydsl_backward_impl,
            flash_attn_varlen_flydsl_forward_impl,
        )

        out, lse = flash_attn_varlen_flydsl_forward_impl(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            softmax_scale=sm_scale,
            causal=True,
            window_size=window_size,
            return_lse=True,
        )
        bwd_func = lambda: flash_attn_varlen_flydsl_backward_impl(
            do,
            q,
            k,
            v,
            out,
            lse,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            softmax_scale=sm_scale,
            causal=True,
            window_size=window_size,
        )
        name = "FLYDSL"
    else:
        qg, kg, vg = (t.detach().requires_grad_() for t in (q, k, v))
        out = turbo.ops.flash_attn_varlen_func(
            qg,
            kg,
            vg,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            softmax_scale=sm_scale,
            causal=True,
            window_size=window_size,
        )
        bwd_func = lambda: out.backward(do, retain_graph=True)
        name = (backend_enum or BackendType.AITER).name

    grads = bwd_func()

    # Correctness: per-segment SDPA reference; may OOM on the widest segments -> SKIP.
    check = "SKIP"
    try:
        if backend_enum == BackendType.FLYDSL:
            dq, dk, dv = grads[0], grads[1], grads[2]
        else:
            dq, dk, dv = qg.grad, kg.grad, vg.grad
        ref = [torch.zeros_like(t) for t in (q, k, v)]
        off = 0
        for seg in segments:
            qs, ks, vs = (t[off : off + seg].detach().clone().requires_grad_() for t in (q, k, v))
            mask = _bottom_right_mask(seg, seg, window_left, device)
            qh, kh, vh = (t.unsqueeze(0).transpose(1, 2) for t in (qs, ks, vs))
            with sdpa_kernel(ATTN_BACKENDS):
                o = torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=mask, scale=sm_scale, enable_gqa=num_head_q > num_head_kv
                )
            o.transpose(1, 2).squeeze(0).backward(do[off : off + seg])
            for dst, src in zip(ref, (qs.grad, ks.grad, vs.grad)):
                dst[off : off + seg] = src
            off += seg
        snrs = [compute_snr(r, x) for r, x in zip(ref, (dq, dk, dv))]
        check = "PASS" if all(snr > 40 for snr in snrs) else f"FAIL({min(snrs):.0f})"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    for _ in range(10):
        bwd_func()
    torch.cuda.synchronize()
    bwd_time = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func}).timeit(50).mean * 1e3

    # Block-diagonal: 5 backward GEMMs (2.5x forward) at head_dim wide, summed per segment.
    flops = sum(
        10.0 * num_head_q * seg * seg * head_dim * _attended_frac(seg, seg, window_left) for seg in segments
    )
    return name, check, bwd_time, flops / (bwd_time * 1e-3) / 1e12


def _dense_cases():
    """(batch, causal, Hq, Hkv, Dqk, Dv, Sq, Skv, window_left) for the model shapes."""
    return [
        (
            batch,
            causal,
            case["num_head_q"],
            case["num_head_kv"],
            case["head_dim_qk"],
            case["head_dim_v"],
            case["seqlen"],
            case["seqlen"],
            -1,
        )
        for causal in (False, True)
        for case in gen_attention_test_cases()
        for batch in BATCH_SIZE_LIST
    ]


def _forward_only_cases():
    """Stable reference and MHA/GQA/MQA short/medium/long comparison cases."""
    return [
        ("dod-mha-h64-b32-s512-noncausal", 32, False, 64, 64, 128, 128, 512, 512, -1),
        ("dod-mha-h64-b32-s512-causal", 32, True, 64, 64, 128, 128, 512, 512, -1),
        ("dod-mha-h64-b16-s1024-noncausal", 16, False, 64, 64, 128, 128, 1024, 1024, -1),
        ("dod-mha-h64-b16-s1024-causal", 16, True, 64, 64, 128, 128, 1024, 1024, -1),
        ("dod-mha-h64-b8-s2048-noncausal", 8, False, 64, 64, 128, 128, 2048, 2048, -1),
        ("dod-mha-h64-b8-s2048-causal", 8, True, 64, 64, 128, 128, 2048, 2048, -1),
        ("dod-mha-h64-b4-s4096-noncausal", 4, False, 64, 64, 128, 128, 4096, 4096, -1),
        ("dod-mha-h64-b4-s4096-causal", 4, True, 64, 64, 128, 128, 4096, 4096, -1),
        ("dod-mha-h64-b2-s8192-noncausal", 2, False, 64, 64, 128, 128, 8192, 8192, -1),
        ("dod-mha-h64-b2-s8192-causal", 2, True, 64, 64, 128, 128, 8192, 8192, -1),
        ("dod-mha-h64-b1-s16384-noncausal", 1, False, 64, 64, 128, 128, 16384, 16384, -1),
        ("dod-mha-h64-b1-s16384-causal", 1, True, 64, 64, 128, 128, 16384, 16384, -1),
        ("mha-short-noncausal", 1, False, 32, 32, 128, 128, 128, 128, -1),
        ("mha-short-causal", 1, True, 32, 32, 128, 128, 128, 128, -1),
        ("mha-medium-noncausal", 1, False, 32, 32, 128, 128, 2048, 2048, -1),
        ("mha-medium-causal", 1, True, 32, 32, 128, 128, 2048, 2048, -1),
        ("mha-long-noncausal", 1, False, 32, 32, 128, 128, 8192, 8192, -1),
        ("mha-long-causal", 1, True, 32, 32, 128, 128, 8192, 8192, -1),
        ("gqa-short-noncausal", 1, False, 32, 8, 128, 128, 128, 128, -1),
        ("gqa-short-causal", 1, True, 32, 8, 128, 128, 128, 128, -1),
        ("gqa-medium-noncausal", 1, False, 32, 8, 128, 128, 2048, 2048, -1),
        ("gqa-medium-causal", 1, True, 32, 8, 128, 128, 2048, 2048, -1),
        ("gqa-long-noncausal", 1, False, 32, 8, 128, 128, 8192, 8192, -1),
        ("gqa-long-causal", 1, True, 32, 8, 128, 128, 8192, 8192, -1),
        ("mqa-short-noncausal", 1, False, 32, 1, 128, 128, 128, 128, -1),
        ("mqa-short-causal", 1, True, 32, 1, 128, 128, 128, 128, -1),
        ("mqa-medium-noncausal", 1, False, 32, 1, 128, 128, 2048, 2048, -1),
        ("mqa-medium-causal", 1, True, 32, 1, 128, 128, 2048, 2048, -1),
        ("mqa-long-noncausal", 1, False, 32, 1, 128, 128, 8192, 8192, -1),
        ("mqa-long-causal", 1, True, 32, 1, 128, 128, 8192, 8192, -1),
    ]


def benchmark_varlen(output_csv, backend_enum):
    """Run the THD document-packing backward table."""
    platform, gpu_name = get_platform_info()
    rows = []
    for case in gen_attention_varlen_test_cases():
        segments, window_left = case["segments"], case["window_left"]
        row = {
            "Platform": platform,
            "GPU": gpu_name,
            "Tag": case["name"],
            "Segments": str(segments),
            "Total": sum(segments),
            "Window": window_left if window_left >= 0 else "full",
            "head_dim": case["head_dim"],
        }
        try:
            name, check, bwd_time, bwd_tflops = profile_varlen(
                segments,
                window_left,
                case["num_head_q"],
                case["num_head_kv"],
                case["head_dim"],
                backend_enum,
            )
            row.update(
                {
                    "Backend": name,
                    "Check": check,
                    "Backward Time (ms)": f"{bwd_time:.3f}",
                    "Backward TFLOPS": f"{bwd_tflops:.2f}",
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed varlen {case['name']} window={window_left}: {e}")
            row.update(
                {
                    "Backend": "ERROR",
                    "Check": "ERROR",
                    "Backward Time (ms)": "ERROR",
                    "Backward TFLOPS": "0.00",
                }
            )
        print(row, flush=True)
        rows.append(row)

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))
    print("\nnote: ragged tiles by max_seqlen, so skewed layouts pay early-exit waste;")
    print("TFLOPS is effective (block-diagonal attended fraction), not peak utilization.")
    filename = output_csv or f"attention_varlen_benchmark_result_{datetime.now():%Y%m%d}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def benchmark_attention(
    output_csv=None,
    use_fp8=False,
    deterministic=False,
    layout="bshd",
    forward_only: bool = False,
    dtype: str = "bf16",
):
    """Run attention benchmark."""
    platform, gpu_name = get_platform_info()

    cases = _forward_only_cases() if forward_only else _dense_cases()

    rows = []
    test_id = 0
    mode = "forward-only" if forward_only else "training"
    print(
        f"Total tests: {len(cases)}, mode: {mode}, layout: {layout}, dtype: {dtype}, "
        f"FP8: {use_fp8}, deterministic: {deterministic}"
    )

    for case in cases:
        test_id += 1
        case_id = case[0] if forward_only else test_id
        case_values = case[1:] if forward_only else case
        (
            batch,
            causal,
            num_head_q,
            num_head_kv,
            head_dim_qk,
            head_dim_v,
            seqlen,
            seqlen_kv,
            window_left,
        ) = case_values

        print(f"\n{'=' * 60}")
        print(
            f"TestID: {case_id}, batch={batch}, seqlen={seqlen}x{seqlen_kv}, "
            f"heads={num_head_q}/{num_head_kv}, dim={head_dim_qk}/{head_dim_v}, "
            f"causal={causal}, window={window_left}, layout={layout}, "
            f"dtype={dtype}, fp8={use_fp8}, deterministic={deterministic}"
        )
        print(f"{'=' * 60}")

        row = {
            "TestID": case_id,
            "Platform": platform,
            "GPU": gpu_name,
            "Batch": batch,
            "SeqLen": seqlen,
            "SeqLenKV": seqlen_kv,
            "num_head_q": num_head_q,
            "num_head_kv": num_head_kv,
            "head_dim_qk": head_dim_qk,
            "head_dim_v": head_dim_v,
            "Causal": causal,
            "Window": window_left if window_left >= 0 else "full",
            "Layout": layout,
            "DType": dtype,
            "Deterministic": deterministic,
        }

        try:
            (
                backend,
                fwd_time,
                fwd_tflops,
                bwd_time,
                bwd_tflops,
                correct,
                det_out_ok,
                det_out_max_abs_diff,
                det_dq_ok,
                det_dq_max_abs_diff,
                det_dk_ok,
                det_dk_max_abs_diff,
                det_dv_ok,
                det_dv_max_abs_diff,
            ) = profile_attention(
                batch,
                seqlen,
                num_head_q,
                num_head_kv,
                head_dim_qk,
                head_dim_v,
                causal,
                use_fp8,
                deterministic,
                seqlen_kv=seqlen_kv,
                window_left=window_left,
                layout=layout,
                forward_only=forward_only,
                dtype=dtype,
            )
            check = "SKIP" if correct is None else ("PASS" if correct else "FAIL")
            timing_status = check if fwd_time is None else "warm-cache"
            row.update(
                {
                    "Backend": backend,
                    "Check": check,
                    "Forward Time (ms)": timing_status if fwd_time is None else f"{fwd_time:.2f}",
                    "Forward TFLOPS": timing_status if fwd_tflops is None else f"{fwd_tflops:.2f}",
                }
            )
            if forward_only:
                row["Forward Timing"] = timing_status
            if not forward_only:
                row.update(
                    {
                        "Backward Time (ms)": f"{bwd_time:.2f}",
                        "Backward TFLOPS": f"{bwd_tflops:.2f}",
                    }
                )
            if deterministic:
                deterministic_ok = det_out_ok
                if not forward_only:
                    deterministic_ok = deterministic_ok and det_dq_ok and det_dk_ok and det_dv_ok
                row["Deterministic Check"] = "PASS" if deterministic_ok else "FAIL"
        except Exception as e:
            print(f"Failed: {str(e)}")
            row.update(
                {
                    "Backend": "ERROR",
                    "Check": "ERROR",
                    "Forward Time (ms)": "ERROR",
                    "Forward TFLOPS": "ERROR" if forward_only else "0.00",
                }
            )
            if forward_only:
                row["Forward Timing"] = "not measured"
            if not forward_only:
                row.update(
                    {
                        "Backward Time (ms)": "ERROR",
                        "Backward TFLOPS": "0.00",
                    }
                )
            if deterministic:
                row["Deterministic Check"] = "ERROR"

        rows.append(row)

    # Create DataFrame
    results = pd.DataFrame(rows)
    if deterministic and "Check" in results.columns and "Deterministic Check" in results.columns:
        cols = list(results.columns)
        cols.insert(cols.index("Check") + 1, cols.pop(cols.index("Deterministic Check")))
        results = results[cols]

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Print average TFLOPS, split by head dim where the table spans more than one: the two
    # dims run at different per-flop efficiencies, so one mean over both says little.
    summary_results = results[results["Check"] == "PASS"] if forward_only else results
    avg_fwd = pd.to_numeric(summary_results["Forward TFLOPS"], errors="coerce").mean()
    summary_suffix = " (correctness-PASS rows only)" if forward_only else ""
    print(f"\nAverage Forward TFLOPS{summary_suffix}: {avg_fwd:.2f}")
    if not forward_only:
        avg_bwd = pd.to_numeric(results["Backward TFLOPS"], errors="coerce").mean()
        print(f"Average Backward TFLOPS: {avg_bwd:.2f}")
    else:
        print("Compare pinned backends only on matching PASS TestIDs across their CSV files.")
    head_dims = sorted(summary_results["head_dim_qk"].unique())
    if len(head_dims) > 1:
        for head_dim in head_dims:
            part = summary_results[summary_results["head_dim_qk"] == head_dim]
            part_fwd = pd.to_numeric(part["Forward TFLOPS"], errors="coerce").mean()
            if forward_only:
                print(f"  head_dim={head_dim}: forward {part_fwd:.2f} TFLOPS")
            else:
                part_bwd = pd.to_numeric(part["Backward TFLOPS"], errors="coerce").mean()
                print(f"  head_dim={head_dim}: forward {part_fwd:.2f}, backward {part_bwd:.2f} TFLOPS")

    # Save to CSV
    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        if forward_only:
            prefix = f"attention_forward_only_{dtype}_benchmark_result"
        elif deterministic:
            prefix = (
                "attention_deterministic_fp8_benchmark_result"
                if use_fp8
                else "attention_deterministic_benchmark_result"
            )
        else:
            prefix = "attention_fp8_benchmark_result" if use_fp8 else "attention_benchmark_result"
        filename = f"{prefix}_{timestamp}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark Attention operations")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename. Defaults describe the selected training or forward-only mode.",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 attention benchmark (default: disabled)",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Benchmark and validate forward latency only, without creating an autograd graph.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Dense input dtype (default: bf16).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic kernel mode (default: disabled).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "aiter", "flydsl", "gluon", "triton", "hipkittens"),
        help="Pin the attention backend (default: auto, i.e. whatever resolves). "
        "hipkittens is gfx950-only and takes bf16 causal sbhd with Sq <= Skv; anything "
        "outside that is refused rather than silently handed to another backend.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="bshd",
        choices=("bshd", "sbhd", "bhsd"),
        help="Storage order of q/k/v. The public operation always receives logical BSHD views.",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Benchmark the THD document-packing backward instead of the dense table.",
    )
    return parser


def _parse_args(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # These checks intentionally run before backend setup or benchmark dispatch, and thus
    # before any benchmark tensor allocation.
    if args.varlen and args.backend == "gluon":
        parser.error("--backend gluon is dense-only and cannot be combined with --varlen")
    if args.backend == "gluon" and not args.forward_only:
        parser.error("--backend gluon is forward-only; add --forward-only")
    if args.forward_only and args.fp8:
        parser.error("--forward-only cannot be combined with --fp8; FP8 uses its separate TRITON path")
    if args.varlen and args.forward_only:
        parser.error("--forward-only is supported only by the dense benchmark, not --varlen")

    return args


if __name__ == "__main__":
    args = _parse_args()

    backend_enum = None if args.backend == "auto" else BackendType[args.backend.upper()]
    if backend_enum is not None:
        GlobalBackendManager.set_attn_backend(backend_enum)

    if args.varlen:
        benchmark_varlen(args.output, backend_enum)
    else:
        benchmark_attention(
            output_csv=args.output,
            use_fp8=args.fp8,
            deterministic=args.deterministic,
            layout=args.layout,
            forward_only=args.forward_only,
            dtype=args.dtype,
        )
