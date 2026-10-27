###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""emu.py -- CPU functional emulation of the generated regions of a built k_mla_fwd code object (one wave).

    python tools/emu.py --build DIR [--ref-dir tools] [--asmkit-root DIR] [--shape toy|s640|prod|b1|b4|mini]
                        [--waves BX:H:W,...]
                        [--wmma exact|seq] [--exp2 cr|perturb] [--seed N]

It executes the instructions of the INIT / TILE_C / TILE_M / FINAL_i regions exactly as they are in the code object
(llvm-objdump listing DIR/k_mla_fwd.objdump.s; true VGPR numbers from the disassembler's MSB tracking; offsets,
literals, op_sel / neg modifiers from the listing), for one wave of one workgroup, with the shell's part modelled in
Python: the LDS image the TDM ops build (Q staging rows, the K / V ring stage of each tile with the shell's row
strides and zero fill past the tensor), the marker operands as the shell computes them, every shell-owned VGPR and
clobbered SGPR poisoned between regions, and every non-persistent generated VGPR poisoned at every region entry.
The O (bf16 bits) and lse of the wave's 64 rows are compared bit for bit with ref_softmax.ref_head (the numpy model of
FlyDSL's forward) under the SAME primitive models (WMMA rounding, exp2 / log2): equality means the code computes
FlyDSL's operation graph; it does not prove the hardware primitives (WMMA, v_exp_f32, v_log_f32) round like the models.
Unknown instructions / operand forms / special-range division inputs raise (fail closed)."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gen"))

XK, XV, NB = 400, 272, 64
K_B, V_B = NB * XK, NB * XV
STG_B = K_B + V_B
Q_OFF = 4 * STG_B
LDS_B = Q_OFF + 256 * XK
LOG2E32 = np.float32(1.4426950408889634)
SHAPES = {  # (b, s, h, layout) as the gates use them; sbhd with b > 1 folds to [1, s, b h, d]
    "mini": (1, 128, 1, "bshd"),
    "toy": (1, 256, 2, "sbhd"),
    "s640": (2, 640, 128, "sbhd"),
    "prod": (2, 4096, 128, "sbhd"),
    "b1": (1, 4096, 128, "bshd"),
    "b4": (4, 4096, 128, "sbhd"),
}

MNRE = re.compile(r"^[a-z][a-z0-9_]*(::[a-z][a-z0-9_]*)?$")


def kernel_dims(shape):
    b, s, h, lay = SHAPES[shape]
    return (1, s, b * h) if (lay == "sbhd" and b > 1) else (b, s, h)


# ------------------------------------------------------------------------------------------- listing -> regions
def regions_of(build_dir):
    """{region name: [asmkit Ins of the code object listing]} (regions found by instruction ordinal in the intended
    .s, which verify_obj matched 1:1 with the listing)."""
    from asmkit.isa import load_any, parse_line

    s_lines = (build_dir / "k_mla_fwd.s").read_text().split("\n")
    ins_s = [parse_line(i, l) for i, l in enumerate(s_lines)]
    start = next(i for i, x in enumerate(ins_s) if x.label == "k_mla_fwd")
    region_of, n, cur = {}, 0, None
    for i in range(start, len(ins_s)):
        x = ins_s[i]
        m = re.search(r";\s*HKGEN\s+(\w+)", s_lines[i])
        if m:
            cur = m.group(1)
        if ";;#ASMEND" in s_lines[i]:
            cur = None
        if x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and MNRE.match(x.mn) and x.mn != "s_code_end":
            if cur:
                region_of[n] = cur
            n += 1
        if s_lines[i].strip().startswith(".Lfunc_end"):
            break
    o_lines, o_ins, _ = load_any(build_dir / "k_mla_fwd.objdump.s")
    seq = [
        x
        for x in o_ins
        if x.mn and x.cls not in ("LABEL", "DIR", "CLAUSE") and MNRE.match(x.mn) and x.mn != "s_code_end"
    ]
    out = {}
    for k, x in enumerate(seq):
        if k in region_of:
            out.setdefault(region_of[k], []).append(x)
    return out


# ------------------------------------------------------------------------------------------- the machine
def f2u(x):
    return np.asarray(x, np.float32).view(np.uint32)


def u2f(x):
    return np.asarray(x, np.uint32).view(np.float32)


class Wave:
    def __init__(self, P, lds):
        self.v = np.zeros((1024, 32), np.uint32)
        self.s = {}
        self.scc = 0
        self.P = P
        self.lds = lds
        self.counts = {}

    # operands -------------------------------------------------------------------------------------------
    def vregs(self, tok):
        from asmkit.isa import first_tok, regs_of

        return regs_of(first_tok(tok))

    def sget(self, name):
        if name in ("vcc_lo", "vcc"):
            name = "vcc"
        if name not in self.s:
            raise KeyError(f"read of unset scalar {name}")
        return self.s[name]

    def src(self, tok, lane_vec=True):
        """32-lane uint32 value of a 32-bit source operand (VGPR, SGPR, inline constant, literal); neg / abs
        modifiers are applied by the caller (as fp32)."""
        t = tok.strip()
        neg = bool(re.match(r"-(v|s\d|s\[|vcc)", t))  # a neg modifier on a register; "-1" is a constant
        if neg:
            t = t[1:]
        rr = self.vregs(t)
        if rr:
            val = self.v[rr[0]].copy()
        elif re.fullmatch(r"s\d+|vcc_lo|vcc", t.split()[0]):
            val = np.full(32, self.sget(t.split()[0]), np.uint32)
        else:
            val = np.full(32, self.const(t.split()[0]), np.uint32)
        return val, neg

    @staticmethod
    def const(t):
        fl = {
            "0": 0,
            "1.0": 0x3F800000,
            "-1.0": 0xBF800000,
            "0.5": 0x3F000000,
            "2.0": 0x40000000,
            "-1": 0xFFFFFFFF,
        }
        if t in fl:
            return fl[t]
        if re.fullmatch(r"-?\d+", t):
            return int(t) & 0xFFFFFFFF
        if re.fullmatch(r"0x[0-9a-fA-F]+", t):
            return int(t, 16) & 0xFFFFFFFF
        raise ValueError(f"constant {t}")

    def fsrc(self, tok):
        val, neg = self.src(tok)
        f = u2f(val).copy()
        return -f if neg else f

    def setv(self, tok, vals):
        rr = self.vregs(tok)
        if len(rr) != 1:
            raise ValueError(f"32-bit destination expected: {tok}")
        self.v[rr[0]] = np.asarray(vals).view(np.uint32) if np.asarray(vals).dtype != np.uint32 else vals

    # execution ------------------------------------------------------------------------------------------
    def run(self, insts):
        for x in insts:
            self.counts[x.mn] = self.counts.get(x.mn, 0) + 1
            self.step(x, [o for o in x.ops if o != "::"])

    def step(self, x, ops):
        import ref_softmax as R

        mn = x.mn
        P = self.P
        if mn in ("s_set_vgpr_msb", "v_nop", "s_nop", "s_barrier_signal") or mn.startswith(("s_wait_",)):
            return
        if mn == "s_mov_b32":
            self.s[ops[0]] = self.src(ops[1])[0][0]
            return
        if mn == "s_cmp_lg_u32":
            self.scc = int(self.src(ops[0])[0][0] != self.src(ops[1])[0][0])
            return
        if mn == "s_cselect_b32":
            a, b = self.src(ops[1])[0][0], self.src(ops[2])[0][0]
            self.s[ops[0]] = a if self.scc else b
            return
        if mn in ("ds_load_b128", "ds_load_tr16_b128"):
            dst = self.vregs(ops[0])
            addr = self.v[self.vregs(ops[1])[0]].astype(np.int64)
            m = re.search(r"offset:(\d+)", x.text)
            addr = addr + (int(m.group(1)) if m else 0)
            if np.any(addr < 0) or np.any(addr + 16 > len(self.lds)):
                raise IndexError(f"LDS read out of range: {x.text.strip()}")
            if mn == "ds_load_b128":
                data = np.stack([self.lds[a : a + 16].view(np.uint32) for a in addr])  # [32, 4]
            else:  # lane 8 g + i, element j = u16 at addr[8 g + j] + 2 i
                h16 = np.zeros((32, 8), np.uint16)
                for g in range(4):
                    for i in range(8):
                        for j in range(8):
                            a = addr[8 * g + j] + 2 * i
                            h16[8 * g + i, j] = self.lds[a : a + 2].view(np.uint16)[0]
                data = h16.view(np.uint32)  # [32, 4] (element 2 r lo, 2 r + 1 hi)
            for r in range(4):
                self.v[dst[r]] = data[:, r]
            return
        if mn == "v_wmma_f32_16x16x32_bf16":
            d, a, b = (self.vregs(o) for o in ops[:3])
            A = self.frag_ab(a)  # [16 m, 32 k]
            B = self.frag_ab(b).T  # [32 k, 16 n]
            C = np.zeros((16, 16), np.float32) if ops[3].strip() == "0" else self.frag_c(self.vregs(ops[3]))
            D = P.wmma(C, A, B)
            self.put_c(d, D)
            return
        if mn in ("v_max3_num_f32", "v_max_num_f32_e32", "v_max_num_f32"):
            vals = [self.fsrc(o) for o in ops[1:]]
            r = vals[0]
            for y in vals[1:]:
                r = np.fmax(r, y)
            self.setv(ops[0], r.astype(np.float32))
            return
        if mn == "v_permlanex16_b32":
            src = self.src(ops[1])[0]
            s1, s2 = self.src(ops[2])[0][0], self.src(ops[3])[0][0]
            out = np.zeros(32, np.uint32)
            for l in range(32):
                i = l % 16
                sel = ((s1 >> (4 * i)) & 0xF) if i < 8 else ((s2 >> (4 * (i - 8))) & 0xF)
                other = 16 if l < 16 else 0
                out[l] = src[other + sel]
            self.setv(ops[0], out)
            return
        if mn in ("v_fma_f32",):
            a, b, c = (self.fsrc(o) for o in ops[1:4])
            self.setv(ops[0], R.fma_f32(a, b, c))
            return
        if mn == "v_fmac_f32_e32":
            a, b = self.fsrc(ops[1]), self.fsrc(ops[2])
            c = u2f(self.v[self.vregs(ops[0])[0]])
            self.setv(ops[0], R.fma_f32(a, b, c))
            return
        if mn in ("v_mul_f32_e32", "v_mul_f32"):
            self.setv(ops[0], R.mul_f32(self.fsrc(ops[1]), self.fsrc(ops[2])))
            return
        if mn in ("v_add_f32_e32", "v_add_f32"):
            self.setv(ops[0], R.add_f32(self.fsrc(ops[1]), self.fsrc(ops[2])))
            return
        if mn in ("v_sub_f32_e32", "v_sub_f32"):
            self.setv(ops[0], R.add_f32(self.fsrc(ops[1]), -self.fsrc(ops[2])))
            return
        if mn in ("v_exp_f32_e32", "v_exp_f32"):
            self.setv(ops[0], P.exp2(self.fsrc(ops[1])).astype(np.float32))
            return
        if mn in ("v_log_f32_e32", "v_log_f32"):
            self.setv(ops[0], P.vlog(self.fsrc(ops[1])).astype(np.float32))
            return
        if mn in ("v_rcp_f32_e32", "v_rcp_f32"):
            a = self.fsrc(ops[1]).astype(np.float64)
            with np.errstate(divide="ignore"):
                self.setv(ops[0], (1.0 / a).astype(np.float32))
            return
        if mn in ("v_mov_b32_e32", "v_mov_b32"):
            self.setv(ops[0], self.src(ops[1])[0])
            return
        if mn in ("v_add_nc_u32_e32", "v_add_nc_u32"):
            a, b = self.src(ops[1])[0].astype(np.uint64), self.src(ops[2])[0].astype(np.uint64)
            self.setv(ops[0], ((a + b) & 0xFFFFFFFF).astype(np.uint32))
            return
        if mn.startswith("v_cmp_") and mn.endswith("_e64"):
            kind = mn[len("v_cmp_") : -len("_e64")]
            op, ty = kind.rsplit("_", 1)
            if ty == "f32":
                a, b = self.fsrc(ops[1]), self.fsrc(ops[2])
            elif ty == "i32":
                a, b = self.src(ops[1])[0].view(np.int32), self.src(ops[2])[0].view(np.int32)
            else:
                raise ValueError(x.text)
            r = {"lt": a < b, "gt": a > b}[op]
            self.s[ops[0]] = int(sum(int(bool(r[l])) << l for l in range(32)))
            return
        if mn == "v_cndmask_b32_e64":
            a, b = self.src(ops[1])[0], self.src(ops[2])[0]
            msk = self.sget(ops[3].strip())
            bits = np.array([(msk >> l) & 1 for l in range(32)], bool)
            self.setv(ops[0], np.where(bits, b, a))
            return
        if mn in ("v_pk_fma_f32", "v_pk_mul_f32", "v_pk_add_f32"):
            self.packed(mn, x, ops)
            return
        if mn == "v_cvt_pk_bf16_f32":
            lo = R.bf16_rne_bits(self.fsrc(ops[1])).astype(np.uint32)
            hi = R.bf16_rne_bits(self.fsrc(ops[2])).astype(np.uint32)
            self.setv(ops[0], (lo | (hi << 16)).astype(np.uint32))
            return
        if mn == "v_div_scale_f32":
            a, b, c = (self.fsrc(o) for o in ops[2:5])  # D = S0 (no scaling in the normal range); flag 0
            self.check_div_normal(b, c, x)
            self.setv(ops[0], a)
            if ops[1].strip() != "null":
                self.s["vcc" if ops[1].startswith("vcc") else ops[1]] = 0
            return
        if mn == "v_div_fmas_f32":
            if self.s.get("vcc", None) != 0:
                raise ValueError("v_div_fmas with VCC != 0 (scaled division) is not modelled")
            a, b, c = (self.fsrc(o) for o in ops[1:4])
            self.setv(ops[0], R.fma_f32(a, b, c))
            return
        if mn == "v_div_fixup_f32":
            q, den, num = (self.fsrc(o) for o in ops[1:4])
            self.check_div_normal(den, num, x)
            self.setv(ops[0], q)
            return
        if mn in ("v_ldexp_f32",):
            a = self.fsrc(ops[1]).astype(np.float64)
            e = self.src(ops[2])[0].view(np.int32)
            self.setv(ops[0], np.ldexp(a, e).astype(np.float32))
            return
        raise ValueError(f"unmodelled instruction: {x.text.strip()}")

    @staticmethod
    def check_div_normal(den, num, x):
        den = np.asarray(den, np.float64)
        num = np.asarray(num, np.float64)
        ok = np.isfinite(den) & (np.abs(den) >= 2.0**-100) & (np.abs(den) <= 2.0**100) & (num == 1.0)
        if not np.all(ok):
            raise ValueError(f"division outside the modelled range: {x.text.strip()} den {den[~ok][:4]}")

    def packed(self, mn, x, ops):
        import ref_softmax as R

        def mods(name, n, dflt):
            m = re.search(name + r":\[([\d,]+)\]", x.text)
            return [int(t) for t in m.group(1).split(",")] if m else [dflt] * n

        nsrc = 3 if mn == "v_pk_fma_f32" else 2
        op_sel, op_sel_hi = mods("op_sel", nsrc, 0), mods("op_sel_hi", nsrc, 1)
        neg_lo, neg_hi = mods("neg_lo", nsrc, 0), mods("neg_hi", nsrc, 0)
        srcs = []
        for k in range(nsrc):
            rr = self.vregs(ops[1 + k])
            if len(rr) != 2:
                raise ValueError(f"packed source must be a VGPR pair: {x.text.strip()}")
            pair = [u2f(self.v[rr[0]]), u2f(self.v[rr[1]])]
            lo = pair[op_sel[k]] * (-1 if neg_lo[k] else 1)
            hi = pair[op_sel_hi[k]] * (-1 if neg_hi[k] else 1)
            srcs.append((lo.astype(np.float32), hi.astype(np.float32)))
        d = self.vregs(ops[0])
        for half in range(2):
            a = [s[half] for s in srcs]
            if mn == "v_pk_fma_f32":
                r = R.fma_f32(a[0], a[1], a[2])
            elif mn == "v_pk_mul_f32":
                r = R.mul_f32(a[0], a[1])
            else:
                r = R.add_f32(a[0], a[1])
            self.v[d[half]] = f2u(r)

    # WMMA fragments (wave32 16x16x32 bf16): A / B lane l holds row (col) l % 16, k = 8 (l // 16) + [0, 8) in VGPRs
    # 0..3 and 16 + 8 (l // 16) + [0, 8) in VGPRs 4..7 (element 2 r lo, 2 r + 1 hi); C / D lane l holds column
    # l % 16, VGPR r row 8 (l // 16) + r.
    def frag_ab(self, regs):
        import ref_softmax as R

        out = np.zeros((16, 32), np.float32)
        for l in range(32):
            h, row = l // 16, l % 16
            for r in range(8):
                w = int(self.v[regs[r], l])
                kb = (8 * h if r < 4 else 16 + 8 * h) + 2 * (r % 4)
                out[row, kb] = R.bf16_to_f32(np.uint16(w & 0xFFFF))
                out[row, kb + 1] = R.bf16_to_f32(np.uint16(w >> 16))
        return out

    def frag_c(self, regs):
        out = np.zeros((16, 16), np.float32)
        for l in range(32):
            h, col = l // 16, l % 16
            for r in range(8):
                out[8 * h + r, col] = u2f(self.v[regs[r], l])
        return out

    def put_c(self, regs, D):
        for l in range(32):
            h, col = l // 16, l % 16
            for r in range(8):
                self.v[regs[r], l] = f2u(np.float32(D[8 * h + r, col]))


# ------------------------------------------------------------------------------------------- the shell's model
def bf16_bytes(x_f32):
    import ref_softmax as R

    return R.bf16_rne_bits(x_f32).astype(np.uint16)


def stage_rows(lds, off, rows_u16, row_stride, ncols):
    """TDM model: rows of ncols bf16 at lds[off + r * row_stride]."""
    for r in range(rows_u16.shape[0]):
        lds[off + r * row_stride : off + r * row_stride + 2 * ncols] = rows_u16[r].view(np.uint8)


def run_wave(rg, P, q, k, v, scale, bx, w, sq, skv, persistent, poison_seed=1):
    """Emulate wave w of the workgroup of q block bx for one head: q [sq, 192], k / v [skv, d] as bf16-valued
    float32. Returns (out bf16 bits [64, 128] by slot-major rows, lse [64], rows)."""
    lds = np.zeros(327680, np.uint8)
    W = Wave(P, lds)
    rng = np.random.default_rng(poison_seed)
    q0 = 256 * bx
    cshift = skv - sq
    qv = min(q0 + 256, sq)
    n = (qv + cshift + NB - 1) // NB
    ncc = min((q0 + cshift + 1) // NB, n)
    lane = np.arange(32)
    # Q staging (rows past sq zero)
    qrows = np.zeros((256, 192), np.uint16)
    nq = max(0, min(256, sq - q0))
    qrows[:nq] = bf16_bytes(q[q0 : q0 + nq])
    stage_rows(lds, Q_OFF, qrows, XK, 192)
    c2 = np.float32(np.float32(scale) * LOG2E32)
    gen_regs = set(range(rg["base"], rg["end0"])) | set(range(256, 1024))
    pers = rg["persistent"]

    def poison_shell():
        for r in range(rg["base"]):
            W.v[r] = rng.integers(0, 2**32, 32, dtype=np.uint64).astype(np.uint32)
        for i in range(90, 104):
            W.s[f"s{i}"] = int(rng.integers(0, 2**32))
        W.s["vcc"] = int(rng.integers(0, 2**32))
        W.scc = int(rng.integers(0, 2))

    def poison_gen():
        for r in sorted(gen_regs - pers):
            W.v[r] = 0x7FC00001  # a NaN: any read before a write shows up in the result

    def setv(name, vals):
        W.v[rg["ops"][name]] = np.asarray(vals, np.int64).astype(np.uint32)

    poison_shell()
    poison_gen()
    for r in sorted(pers):
        W.v[r] = 0x7FC00002
    qa = Q_OFF + (16 * w + (lane & 15)) * XK + (lane >> 4) * 16
    setv("INIT.qa", qa)
    setv("INIT.qb", qa + 2 * 64 * XK)
    W.s[rg["ops"]["INIT.c2"]] = int(f2u(c2))
    W.run(rg["regions"]["INIT"])
    k_lane = (lane & 15) * XK + (lane >> 4) * 16
    v_lane = ((lane >> 4) * 8 + (lane & 7)) * XV + ((lane >> 3) & 1) * 16
    dl0 = q0 + 16 * w + (lane & 15) + cshift - (lane >> 4) * 8
    for t in range(n):
        st = (t & 3) * STG_B
        kv0 = t * NB
        krows = np.zeros((NB, 192), np.uint16)
        vrows = np.zeros((NB, 128), np.uint16)
        nk = max(0, min(NB, skv - kv0))
        krows[:nk] = bf16_bytes(k[kv0 : kv0 + nk])
        vrows[:nk] = bf16_bytes(v[kv0 : kv0 + nk])
        stage_rows(lds, st, krows, XK, 192)
        stage_rows(lds, st + K_B, vrows, XV, 128)
        poison_shell()
        poison_gen()
        if t < ncc:
            name = "TILE_C"
        elif "TILE_D0" in rg["regions"]:  # the diagonal variant (Sq == Skv)
            if skv != sq or t - ncc > 3:
                raise ValueError("the TILE_D variant needs Sq == Skv")
            name = f"TILE_D{t - ncc}"
        else:
            name = "TILE_M"
        setv(f"{name}.kb", st + k_lane)
        setv(f"{name}.vb", st + K_B + v_lane)
        if name != "TILE_C":
            setv(f"{name}.dl", dl0 - NB * t)
        W.run(rg["regions"][name])
    # the fp32 state before the epilogue: m, d (lanes 0..15 = rows of the q tile) and O^T (lane l = row l % 16,
    # VGPR r of O[i][dt] = column 16 dt + 8 (l // 16) + r)
    L = rg["L"]
    state = {
        "m": np.zeros(64, np.float32),
        "d": np.zeros(64, np.float32),
        "O": np.zeros((64, 128), np.float32),
    }
    for i in range(4):
        state["m"][16 * i : 16 * i + 16] = u2f(W.v[L.m[i]])[:16]
        state["d"][16 * i : 16 * i + 16] = u2f(W.v[L.d[i]])[:16]
        for dt in range(8):
            for l in range(32):
                h, col = l // 16, l % 16
                for r in range(8):
                    state["O"][16 * i + col, 16 * dt + 8 * h + r] = u2f(W.v[L.O(i, dt) + r, l])
    outs, lses, rows = [], [], []
    for i in range(4):
        poison_shell()
        poison_gen()
        W.run(rg["regions"][f"FINAL_{i}"])
        o0 = rg["ops"][f"FINAL_{i}.o"]
        words = W.v[
            o0 : o0 + 32
        ]  # [32 regs, 32 lanes]: reg 4 dt + r = bf16 pair of d = 16 dt + 8 h + 2 r, +1
        out = np.zeros((16, 128), np.uint16)
        for l in range(32):
            h, col = l // 16, l % 16
            for dt in range(8):
                for r in range(4):
                    wv = int(words[4 * dt + r, l])
                    out[col, 16 * dt + 8 * h + 2 * r] = wv & 0xFFFF
                    out[col, 16 * dt + 8 * h + 2 * r + 1] = wv >> 16
        lse = u2f(W.v[rg["ops"][f"FINAL_{i}.lse"]])[:16].copy()
        outs.append(out)
        lses.append(lse)
        rows.append(q0 + 16 * (w + 4 * i) + np.arange(16))
    return np.concatenate(outs), np.concatenate(lses), np.concatenate(rows), W.counts, state


def load_build(build_dir):
    import json

    sys.path.insert(0, str(HERE.parent / "gen"))
    import fwdgen as G

    bj = json.loads((build_dir / "build.json").read_text())
    base = bj["layout"]["base"]
    L = G.Layout(base)
    ops = {}

    def vfirst(t):
        m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", t)
        return int(m.group(1) or m.group(2))

    for name, d in bj["markers"].items():
        for kk, vv in d.items():
            ops[f"{name}.{kk}"] = vfirst(vv) if vv.startswith("v") else vv
    return {
        "regions": regions_of(build_dir),
        "ops": ops,
        "base": base,
        "end0": L.end0,
        "persistent": L.persistent(),
        "L": L,
    }


class PerturbPrims:
    """Prims with exp2 / log2 = correctly rounded, then the last bit flipped for inputs with an odd hash: a model
    that differs from every real one, so a match under it is about the operation graph, not the primitive."""

    def __init__(self, base):
        self.b = base
        self.wmma = base.wmma

    def _flip(self, x, y):
        u = np.asarray(x, np.float32).view(np.uint32)
        h = (u * np.uint32(2654435761)) >> np.uint32(31)
        yy = np.asarray(y, np.float32).copy()
        fin = np.isfinite(yy) & (yy != 0)
        bits = yy.view(np.uint32)
        bits[fin] ^= h[fin].astype(np.uint32)
        return yy

    def exp2(self, x):
        return self._flip(x, self.b.exp2(x))

    def vlog(self, x):
        return self._flip(x, self.b.vlog(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--ref-dir", default=str(HERE), help="directory of ref_softmax.py (default: tools/)")
    ap.add_argument(
        "--asmkit-root", default=str(HERE.parent.parent), help="directory holding the asmkit package"
    )
    ap.add_argument("--shape", default="toy", choices=sorted(SHAPES))
    ap.add_argument(
        "--waves", default="", help="bx:head:wave,... (default: first and last block, waves 0 and 3)"
    )
    ap.add_argument("--wmma", default="exact", choices=("exact", "seq"))
    ap.add_argument("--exp2", default="cr", choices=("cr", "perturb"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.1352337788608801)
    a = ap.parse_args()
    sys.path.insert(0, a.ref_dir)
    sys.path.insert(0, a.asmkit_root)
    import ref_softmax as R

    rg = load_build(Path(a.build))
    B, S, H = kernel_dims(a.shape)
    nqb = (S + 255) // 256
    P = R.Prims(wmma=a.wmma)
    if a.exp2 == "perturb":
        P = PerturbPrims(P)
    sel = []
    if a.waves:
        for t in a.waves.split(","):
            bx, hd, w = (int(z) for z in t.split(":"))
            sel.append((bx, hd, w))
    else:
        sel = [(0, 0, 0), (nqb - 1, H - 1, 3), (nqb - 1, 0, 0)]
    heads = {}
    ok_all = True
    for bx, hd, w in sel:
        if hd not in heads:
            g = np.random.default_rng(a.seed * 1000 + hd)
            qh = R.to_bf16_f32(g.standard_normal((S, 192)).astype(np.float32))
            kh = R.to_bf16_f32(g.standard_normal((S, 192)).astype(np.float32))
            vh = R.to_bf16_f32(g.standard_normal((S, 128)).astype(np.float32))
            tr = {}
            ro, rl = R.ref_head(qh, kh, vh, a.scale, P, trace=tr)
            heads[hd] = (qh, kh, vh, ro, rl, tr)
        qh, kh, vh, ro, rl, tr = heads[hd]
        out, lse, rows, counts, st = run_wave(rg, P, qh, kh, vh, a.scale, bx, w, S, S, False, a.seed + 7)
        valid = rows < S
        rv = rows[valid]
        eo = int(np.sum(out[valid] != ro[rv]))
        el = int(np.sum(lse[valid].view(np.uint32) != rl[rv].view(np.uint32)))
        em = int(np.sum(st["m"][valid].view(np.uint32) != tr["m"][rv].view(np.uint32)))
        ed = int(np.sum(st["d"][valid].view(np.uint32) != tr["d"][rv].view(np.uint32)))
        eO = int(np.sum(st["O"][valid].view(np.uint32) != tr["O"][rv].view(np.uint32)))
        ok = eo == 0 and el == 0 and em == 0 and ed == 0 and eO == 0 and bool(np.all(np.isfinite(lse[valid])))
        ok_all &= ok
        print(
            f"EMU shape={a.shape} wmma={a.wmma} exp2={a.exp2} block={bx} head={hd} wave={w} rows={int(valid.sum())} "
            f"out_bits_diff={eo}/{out[valid].size} lse_bits_diff={el}/{int(valid.sum())} "
            f"state_diff(m,d,O_fp32)={em},{ed},{eO} {'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
    print(f"EMU_RESULT {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
