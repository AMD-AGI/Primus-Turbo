"""Make a compile-only variant of the fwd champion. Usage: mkvariant.py <label> <waves> <n_block> <wpe> [min_kv_kb]
Copies the champion (never edits it) and patches: NUM_WAVES, DEFAULT_N_BLOCK, waves_per_eu,
MIN_KV_BLK_BYTES, the 5 num_waves!=8 guards, and threads num_waves into the collective K/V TDM atom."""
import pathlib, re, shutil, sys
CHAMP = pathlib.Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/job_context/op/current")
label, waves, nb, wpe = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
minkv = int(sys.argv[5]) if len(sys.argv) > 5 else 64
dst = pathlib.Path(__file__).parent / "work" / label
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(CHAMP, dst, ignore=shutil.ignore_patterns("__pycache__"))
k = dst / "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"
m = dst / "flydsl_fwd/fmha_b16_buffer_managers.py"
ks = k.read_text(); ms = m.read_text()
def sub(pat, rep, s, n=1):
    s2, c = re.subn(pat, rep, s)
    assert c == n, (pat, c)
    return s2
ks = sub(r"\nNUM_WAVES = 8 ", f"\nNUM_WAVES = {waves} ", ks)
ks = sub(r"\nDEFAULT_N_BLOCK = 64\n", f"\nDEFAULT_N_BLOCK = {nb}\n", ks)
ks = sub(r"\nMIN_KV_BLK_BYTES = 64 \* 1024\n", f"\nMIN_KV_BLK_BYTES = {minkv} * 1024\n", ks)
ks = sub(r'compile_hints\["waves_per_eu"\] = 2', f'compile_hints["waves_per_eu"] = {wpe}', ks, 2)
if waves != 8:
    ms = sub(r'raise NotImplementedError\("V2 TDM loader assumes 8 waves"\)', "pass  # asm-structure: guard lifted", ms, 4)
    ms = sub(r'raise NotImplementedError\("V3 assumes 8 waves"\)', "pass  # asm-structure: guard lifted", ms, 1)
    # the one real dependency: collective K/V TDM atom split across num_warps wave ids
    ms = sub(r"def _tdm_load_views\(\n    \*,\n", "def _tdm_load_views(\n    *,\n    num_warps=_DEFAULT_NUM_WAVES,\n", ms)
    ms = sub(r"num_warps=_DEFAULT_NUM_WAVES,\n            pad_interval=w,", "num_warps=num_warps,\n            pad_interval=w,", ms)
    ms = sub(r"return _tdm_load_views\(\n", "return _tdm_load_views(\n            num_warps=self.num_waves,\n", ms, 2)
k.write_text(ks); m.write_text(ms)
print("variant", dst)
