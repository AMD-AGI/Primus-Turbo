"""Host side of the FlyDSL MLA backward (D_QK != D_V), dense causal SBHD, a16 dQ."""

import torch

from . import flash_attn_bwd_mla as fb

_LOG2E = fb._LOG2E
_BWD_CACHE: dict = {}
_PRIMED: set = set()


def _dkdv_kwargs(B, Hq, Hkv, d_qk, d_v, scale, block_kv, block_q, q_split, **over):
    kw = dict(
        num_heads=Hq,
        head_dim=d_qk,
        head_dim_v=d_v,
        causal=True,
        dtype_str="bf16",
        sm_scale=scale,
        num_kv_heads=Hkv,
        window_left=-1,
        q_split=q_split,
        block_kv=block_kv,
        batch_size=B,
        sbhd=True,
        waves_per_eu=1,
        g2d=2,
        dma_grp=1,
        pf_ring=False,
        square=True,
        kv_halves=1,
        q_pref=True,
        g3_defer=True,
        mfma_tie=0,
        g3_dbat=2,
        g3_kreg=False,
        k_reg=True,
        g3_st_at=0,
        g3_st_n=1,
        g3_sb=1,
        block_q=block_q,
        flat_wg=256,
        wsq_a16=True,
    )
    kw.update(over)
    return kw


def _launchers(B, Sq, Hq, Hkv, d_qk, d_v, scale, block_kv, block_q, q_split, n_img, over):
    key = (B, Sq, Hq, Hkv, d_qk, d_v, scale, block_kv, block_q, q_split, n_img, tuple(sorted(over.items())))
    hit = _BWD_CACHE.get(key)
    if hit is None:
        dkdv = fb.build_flash_attn_bwd_dkdv_module(
            **_dkdv_kwargs(B, Hq, Hkv, d_qk, d_v, scale, block_kv, block_q, q_split, a16_imgs=n_img, **over)
        )
        odo = fb.build_flash_attn_bwd_odo_module(num_heads=Hq, head_dim=d_v, sbhd=True)
        # K is staged in LDS prescaled by sm*log2e, so the image carries dQ*log2e.
        dqa16 = fb.build_flash_attn_bwd_dqa16_module(B, Sq, Hq, d_qk, 1.0 / _LOG2E, n_img=n_img)
        hit = (dkdv, odo, dqa16)
        _BWD_CACHE[key] = hit
    return hit


_I32_MAX = 2**31 - 1
# bf16 atomic adds one dQ image takes per element at most, and the most images a call spreads over.
_A16_ADDS_PER_IMG = 128
_A16_MAX_IMGS = 4


def _a16_imgs_for(Skv, block_kv):
    """A dQ row collects one bf16 atomic per kv band, and the running-sum rounding grows with that
    count (measured: about -2 dB of dQ SNR per doubling), so long sequences spread the bands."""
    bands = Skv // block_kv
    return min(_A16_MAX_IMGS, max(1, -(-bands // _A16_ADDS_PER_IMG)))


def _q_split_for(B, Skv, Hkv, d_qk, q_split):
    """The dK slot workspace is one flat kernel argument whose length is passed as int32."""
    if q_split is not None:
        return q_split
    return 2 if 2 * Skv * B * Hkv * d_qk <= _I32_MAX else 1


def mla_bwd_sbhd(dout, q, k, v, out, lse, softmax_scale=None, block_kv=128, block_q=32, q_split=None, **over):
    """dQ, dK, dV for causal MLA in SBHD.

    q [Sq,B,Hq,Dqk], k [Skv,B,Hkv,Dqk], v [Skv,B,Hkv,Dv], out/dout [Sq,B,Hq,Dv],
    lse [B*Sq, Hq] fp32 natural log (the forward's layout). Sq == Skv.
    """
    Sq, B, Hq, d_qk = q.shape
    Skv, _, Hkv, d_v = v.shape
    q_split = _q_split_for(B, Skv, Hkv, d_qk, q_split)
    assert Sq == Skv and k.shape == (Skv, B, Hkv, d_qk) and dout.shape == (Sq, B, Hq, d_v)
    assert Skv % block_kv == 0 and Sq % (block_q * q_split) == 0
    scale = (1.0 / d_qk**0.5) if softmax_scale is None else float(softmax_scale)
    n_img = _a16_imgs_for(Skv, block_kv)
    dkdv, odo, dqa16 = _launchers(B, Sq, Hq, Hkv, d_qk, d_v, scale, block_kv, block_q, q_split, n_img, over)
    st = torch.cuda.current_stream()

    qf = fb._prescale_q(q, scale)
    kf, vf = k.reshape(-1), v.reshape(-1)
    dof = dout.reshape(-1)
    o16 = out.reshape(-1)
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    odo(o16, dof, delta.reshape(-1), B, Sq, st)
    img = torch.zeros(n_img, B * Sq * Hq * d_qk, device=q.device, dtype=q.dtype)
    # A tensor argument's length is packed as int32, so the kernels are handed image 0 alone and
    # reach image i at i * |image| past its base (see A16_IMGS); the stack is one allocation.
    img0 = img[0]
    lse_bhsq = lse.view(B, Sq, Hq).permute(0, 2, 1)
    lsef = fb._prescale_lse(lse_bhsq, st).reshape(-1)
    ws_dk = torch.empty(q_split, Skv, B, Hkv, d_qk, device=q.device, dtype=k.dtype)
    ws_dv = torch.empty(q_split, Skv, B, Hkv, d_v, device=q.device, dtype=v.dtype)
    cu_ph = fb._cu_placeholder(q.device)
    args = (qf, kf, vf, dof, lsef, delta.reshape(-1), ws_dk.reshape(-1), ws_dv.reshape(-1), cu_ph, cu_ph, img0)
    pk = (id(dkdv), B, Sq)
    if pk not in _PRIMED:
        # A compiled launcher dispatches its grid twice on its first call, which the
        # accumulated image would double.
        _PRIMED.add(pk)
        dkdv(*args, B, Sq, Skv, 0, st)
        img.zero_()
    dkdv(*args, B, Sq, Skv, 0, st)
    dq = torch.empty_like(q)
    dqa16(img0, dq.reshape(-1), st)
    dk = ws_dk.sum(dim=0) if q_split > 1 else ws_dk[0]
    dv = ws_dv.sum(dim=0) if q_split > 1 else ws_dv[0]
    return dq, dk, dv
