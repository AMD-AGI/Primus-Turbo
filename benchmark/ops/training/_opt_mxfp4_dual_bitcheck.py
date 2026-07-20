"""Bit-exact probe: FlyDSL mxfp4 row cast (+RHT) vs C++ quantize_mxfp4 (axis=1).

Not part of the scored ruler. Run in-container:
  python -u benchmark/ops/training/_opt_mxfp4_dual_bitcheck.py
"""

import torch

# ensure the C++ ops are registered
import primus_turbo.pytorch.kernels.quantization.quantization_impl  # noqa: F401
from primus_turbo.flydsl.quant.mxfp4_quant_kernel import (
    get_dual2_cast,
    get_dual_cast,
    get_row_cast,
)
from primus_turbo.pytorch.core.low_precision import ScalingRecipe

MXFP4_PAD = 128


def cpp_row(x, use_rht):
    rec = ScalingRecipe(
        use_2d_block=False, use_sr=False, use_rht=use_rht, shuffle_scale=False, shuffle_out=False
    )
    out, scale = torch.ops.primus_turbo_cpp_extension.quantize_mxfp4(
        x,
        torch.float4_e2m1fn_x2,
        1,
        MXFP4_PAD,
        rec.use_2d_block,
        rec.use_sr,
        rec.use_rht,
        rec.shuffle_scale,
        rec.shuffle_out,
    )
    return out.view(torch.uint8), scale.view(torch.uint8)


def fly_row(x, use_rht):
    R, C = x.shape
    x_i32 = x.view(torch.int32)  # [R, C/2]
    out = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
    sc = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()
    fn, grid_x = get_row_cast(R, C, use_rht)
    fn(x_i32, out, sc, R, C, grid_x, stream)
    torch.cuda.synchronize()
    return out.view(torch.uint8), sc


def check(R, C, use_rht):
    torch.manual_seed(0)
    x = torch.randn((R, C), dtype=torch.bfloat16, device="cuda")
    co, cs = cpp_row(x, use_rht)
    fo, fs = fly_row(x, use_rht)
    # compare valid region only (C++ scale may pad N)
    cs_v = cs[:, : C // 32]
    data_eq = torch.equal(co[:, : C // 2], fo)
    scale_eq = torch.equal(cs_v, fs)
    dmm = (co[:, : C // 2] != fo).sum().item()
    smm = (cs_v != fs).sum().item()
    print(f"[R={R} C={C} rht={use_rht}] data_eq={data_eq} ({dmm} mism) scale_eq={scale_eq} ({smm} mism)")
    return data_eq and scale_eq


def cpp_dual(x, row_rht, col_rht, row_2d=False, col_2d=False):
    rrec = ScalingRecipe(
        use_2d_block=row_2d, use_sr=False, use_rht=row_rht, shuffle_scale=False, shuffle_out=False
    )
    crec = ScalingRecipe(
        use_2d_block=col_2d, use_sr=False, use_rht=col_rht, shuffle_scale=False, shuffle_out=False
    )
    ro, rs, co, cs = torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        torch.float4_e2m1fn_x2,
        MXFP4_PAD,
        rrec.use_2d_block,
        rrec.use_sr,
        rrec.use_rht,
        crec.use_2d_block,
        crec.use_sr,
        crec.use_rht,
        rrec.shuffle_scale,
        rrec.shuffle_out,
        crec.shuffle_scale,
        crec.shuffle_out,
    )
    return (ro.view(torch.uint8), rs.view(torch.uint8), co.view(torch.uint8), cs.view(torch.uint8))


def fly_dual(x, row_rht, col_rht, row_2d=False, col_2d=False):
    R, C = x.shape
    x_i32 = x.view(torch.int32)
    ro = torch.zeros((R, C // 8), dtype=torch.int32, device="cuda")
    rs = torch.zeros((R, C // 32), dtype=torch.uint8, device="cuda")
    co = torch.zeros((C, R // 8), dtype=torch.int32, device="cuda")
    cs = torch.zeros((C, R // 32), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()
    fn, grid_x = get_dual_cast(R, C, row_rht, col_rht, row_2d, col_2d)
    fn(x_i32, ro, rs, co, cs, R, C, grid_x, stream)
    torch.cuda.synchronize()
    return ro.view(torch.uint8), rs, co.view(torch.uint8), cs


def check_dual(R, C, row_rht, col_rht, row_2d=False, col_2d=False):
    torch.manual_seed(0)
    x = torch.randn((R, C), dtype=torch.bfloat16, device="cuda")
    cro, crs, cco, ccs = cpp_dual(x, row_rht, col_rht, row_2d, col_2d)
    fro, frs, fco, fcs = fly_dual(x, row_rht, col_rht, row_2d, col_2d)
    res = {
        "row_data": torch.equal(cro[:, : C // 2], fro),
        "row_scale": torch.equal(crs[:, : C // 32], frs),
        "col_data": torch.equal(cco[:, : R // 2], fco),
        "col_scale": torch.equal(ccs[:, : R // 32], fcs),
    }
    allok = all(res.values())
    print(f"[DUAL R={R} C={C} rrht={row_rht} crht={col_rht} r2d={row_2d} c2d={col_2d}] {res}")
    return allok


def fly_dual2(xa, xb, a_recipes, b_recipes):
    Ra, Ca = xa.shape
    Rb, Cb = xb.shape

    def _alloc(R, C, dev):
        return (
            torch.zeros((R, C // 8), dtype=torch.int32, device=dev),
            torch.zeros((R, C // 32), dtype=torch.uint8, device=dev),
            torch.zeros((C, R // 8), dtype=torch.int32, device=dev),
            torch.zeros((C, R // 32), dtype=torch.uint8, device=dev),
        )

    roa, rsa, coa, csa = _alloc(Ra, Ca, xa.device)
    rob, rsb, cob, csb = _alloc(Rb, Cb, xb.device)
    recipes = tuple(a_recipes) + tuple(b_recipes)
    fn, grid_x, a_blocks = get_dual2_cast(Ra, Ca, Rb, Cb, recipes)
    fn(
        xa.view(torch.int32),
        roa,
        rsa,
        coa,
        csa,
        xb.view(torch.int32),
        rob,
        rsb,
        cob,
        csb,
        Ra,
        Ca,
        Rb,
        Cb,
        a_blocks,
        grid_x,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return (roa.view(torch.uint8), rsa, coa.view(torch.uint8), csa), (
        rob.view(torch.uint8),
        rsb,
        cob.view(torch.uint8),
        csb,
    )


def check_dual2(Ra, Ca, Rb, Cb):
    # A recipe: row rht=F/2d=F, col rht=T/2d=F ; B recipe: row rht=F/2d=T, col rht=T/2d=T
    a_recipes = (False, True, False, False)
    b_recipes = (False, True, True, True)
    torch.manual_seed(0)
    xa = torch.randn((Ra, Ca), dtype=torch.bfloat16, device="cuda")
    xb = torch.randn((Rb, Cb), dtype=torch.bfloat16, device="cuda")
    ca = cpp_dual(xa, False, True, False, False)
    cb = cpp_dual(xb, False, True, True, True)
    (fa), (fb) = fly_dual2(xa, xb, a_recipes, b_recipes)
    ok = True
    for tag, (cro, crs, cco, ccs), (fro, frs, fco, fcs), (R, C) in [
        ("A", ca, fa, (Ra, Ca)),
        ("B", cb, fb, (Rb, Cb)),
    ]:
        res = {
            "row_data": torch.equal(cro[:, : C // 2], fro),
            "row_scale": torch.equal(crs[:, : C // 32], frs),
            "col_data": torch.equal(cco[:, : R // 2], fco),
            "col_scale": torch.equal(ccs[:, : R // 32], fcs),
        }
        print(f"[DUAL2-{tag} R={R} C={C}] {res}")
        ok &= all(res.values())
    return ok


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dual2":
        ok = True
        # (Ra,Ca) = A[M,K], (Rb,Cb) = B[N,K]; same K, different free dim.
        for Ra, Ca, Rb, Cb in [
            (256, 256, 512, 256),
            (4096, 4096, 12288, 4096),
            (4096, 8192, 28672, 8192),
            (128, 512, 256, 512),
        ]:
            ok &= check_dual2(Ra, Ca, Rb, Cb)
        print("ALL_BITEXACT" if ok else "FAIL")
    elif len(sys.argv) > 1 and sys.argv[1] == "row2d":
        ok = True
        # P1 step 1: B-weight row recipe = row 2d=T rht=F, col 2d=F rht=F.
        # M!=K shapes expose transpose + tile-boundary bugs.
        for R, C in [(256, 256), (256, 512), (128, 512), (4096, 4096), (4096, 8192), (8192, 4096)]:
            ok &= check_dual(R, C, False, False, row_2d=True, col_2d=False)
        print("ALL_BITEXACT" if ok else "FAIL")
    elif len(sys.argv) > 1 and sys.argv[1] == "brecipe":
        ok = True
        shapes = [(256, 256), (256, 512), (128, 512), (4096, 4096), (4096, 8192), (8192, 4096)]
        # col-2d alone (rht off then on), then the full B recipe (row 2d, col 2d+rht).
        for R, C in shapes:
            ok &= check_dual(R, C, False, False, row_2d=False, col_2d=True)
        for R, C in shapes:
            ok &= check_dual(R, C, False, True, row_2d=False, col_2d=True)
        for R, C in shapes:
            ok &= check_dual(R, C, False, True, row_2d=True, col_2d=True)
        print("ALL_BITEXACT" if ok else "FAIL")
    elif len(sys.argv) > 1 and sys.argv[1] == "dual":
        ok = True
        # A recipe: row rht=F, col rht=T ; grad_out recipe: both rht=T
        for rr, cr in [(False, False), (False, True), (True, True)]:
            for R, C in [(256, 256), (256, 512), (4096, 4096), (4096, 8192), (8192, 4096)]:
                ok &= check_dual(R, C, rr, cr)
        print("ALL_BITEXACT" if ok else "FAIL")
    else:
        ok = True
        for rht in (False, True):
            for R, C in [(256, 256), (256, 512), (4096, 4096), (4096, 8192), (8192, 4096)]:
                ok &= check(R, C, rht)
        print("ALL_BITEXACT" if ok else "FAIL")
