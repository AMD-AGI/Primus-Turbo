"""GPU preflight for the experimental fused Q preparation, run before training."""

import json
import math
from pathlib import Path

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import build_flash_attn_bwd_odo_module


def main():
    torch.manual_seed(30279)
    stream = torch.cuda.current_stream()
    results = []
    # Odd S exercises masked tails; batch slices exercise the native D64 plan.
    for seq, batch, lo, size in [(257, 4, 0, 4), (256, 4, 0, 2), (256, 4, 2, 2)]:
        shape = (seq, batch, 64, 64)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        q.flatten()[::97] *= 96
        original_q = q.clone()
        o, do = torch.randn_like(q), torch.randn_like(q)
        delta = torch.full((batch, 64, seq), float("nan"), device="cuda")
        reference_delta = torch.full_like(delta, float("nan"))
        image = torch.full_like(q, 7)
        reference_image = torch.full_like(q, 7)
        scaled = torch.full_like(q, float("nan"))
        scale = math.log2(math.e) / math.sqrt(64)
        kwargs = dict(num_heads=64, head_dim=64, sbhd=True, fill_img=True, bat_lo=lo, bat_all=batch)
        ref = build_flash_attn_bwd_odo_module(**kwargs)
        fused = build_flash_attn_bwd_odo_module(**kwargs, q_scale=scale)
        ref(
            o.flatten(),
            do.flatten(),
            reference_delta.flatten(),
            size,
            seq,
            stream,
            img=reference_image.flatten(),
        )
        fused(
            o.flatten(),
            do.flatten(),
            delta.flatten(),
            size,
            seq,
            stream,
            img=image.flatten(),
            q=q.flatten(),
            qs=scaled.flatten(),
        )
        torch.cuda.synchronize()
        expected_q = q * scale
        assert torch.equal(scaled[:, lo : lo + size], expected_q[:, lo : lo + size]), "Q rounding differs"
        assert torch.equal(delta[lo : lo + size], reference_delta[lo : lo + size]), "delta differs"
        assert torch.equal(image, reference_image), "image initialization differs"
        assert torch.equal(q, original_q), "input Q was modified"
        for outside in (scaled[:, :lo], scaled[:, lo + size :]):
            assert torch.isnan(outside).all(), "Q prep wrote outside its batch slice"
        results.append(
            dict(shape=shape, batch_lo=lo, batch_size=size, exact_q=True, exact_delta=True, exact_image=True)
        )
    payload = dict(passed=True, device=torch.cuda.get_device_name(), cases=results)
    Path("/results/attention_qprep_preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("ATTENTION_QPREP_PREFLIGHT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
