"""Build M8 ablation arms from the fwd champion r13ns (timing-only arms; outputs are wrong by design).
Every patch is an exact, asserted-unique string replacement in flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py.
No index/address expression is touched."""
import shutil, pathlib, sys
SRC = pathlib.Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/champions/fwd_r16_r13ns")  # same tree as 0928__a0_repro/arms/fwd_r13ns
OUT = pathlib.Path(__file__).resolve().parent / "arms"
K = "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"

FLAGS_AT = 'O_VARIANT = "v3"\n'
P_EXP = ("    def exp2(x):\n        return fx.Float32(rocdl.exp2(f32, _raw(x)))\n",
         "    def exp2(x):\n        if ABL_NOEXP:  # M8: trans op replaced by one VALU mul\n            return fmul(x, fx.Float32(0.5))\n        return fx.Float32(rocdl.exp2(f32, _raw(x)))\n")
P_SM = ("    # ---- Row max: the R rows' balanced max-trees emitted INTERLEAVED",
        "    if ABL_NOSM:  # M8: no max / exp / sum / rescale; P = cvt(masked S)\n"
        "        return ([[fx.Vector.from_elements(s_masked_list[r][k * 8:(k + 1) * 8], fx.Float32).to(elem_dtype)\n"
        "                  for k in range(NKV)] for r in range(R)],\n"
        "                list(m_prev_list), list(d_prev_list), [fx.Float32(1.0) for _ in range(R)],\n"
        "                [fx.Int32(0) != fx.Int32(0) for _ in range(R)])\n\n"
        "    # ---- Row max: the R rows' balanced max-trees emitted INTERLEAVED")
P_MASK = ("    def main_loop(t, state, *, mask_left, mask_right, kv_len):\n",
          "    def main_loop(t, state, *, mask_left, mask_right, kv_len):\n"
          "        if ABL_NOMASK:  # M8: every tile runs the mask-free body (same tile count)\n"
          "            mask_left = mask_right = kv_len = None\n")
P_BAR = ("            rocdl.sched_barrier(0)\n            gpu.barrier()\n            rocdl.sched_barrier(0)\n",
         "            rocdl.sched_barrier(0)\n            if not ABL_NOBAR:  # M8: per-tile WG barrier removed (TDM wait kept)\n"
         "                gpu.barrier()\n            rocdl.sched_barrier(0)\n")
ARMS = {"base": {}, "noexp": {"ABL_NOEXP"}, "nosm": {"ABL_NOSM"}, "nomask": {"ABL_NOMASK"},
        "nobar": {"ABL_NOBAR"}, "wl": {"ABL_NOSM", "ABL_NOMASK"}}
ALL = ["ABL_NOEXP", "ABL_NOSM", "ABL_NOMASK", "ABL_NOBAR"]
for name, on in ARMS.items():
    d = OUT / name
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__"))
    f = d / K; s = f.read_text()
    for old, new in (P_EXP, P_SM, P_MASK, P_BAR):
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    assert s.count(FLAGS_AT) == 1
    s = s.replace(FLAGS_AT, FLAGS_AT + "".join(f"{a} = {a in on}  # M8 ablation flag\n" for a in ALL))
    f.write_text(s)
    print(name, sorted(on))
