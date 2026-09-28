"""Create one arm directory per compile-option setting (L24). Kernel source identical to the
champion except the llvm_options dict of the two _launch compile_hints (bshd + thd)."""
import json, pathlib, shutil, sys
SRC = pathlib.Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op/current")
L = pathlib.Path(__file__).resolve().parent.parent
KF = "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py"
OLD = '        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,\n'
# arm -> llvm_options entries added to (or replacing) the champion's dict
ARMS = {
    "base":      {},                                                   # control, byte-identical copy
    "mmc":       {"amdgpu-sched-strategy": "max-memory-clause"},
    "nopm":      {"enable-post-misched": False},
    "mmc_nopm":  {"amdgpu-sched-strategy": "max-memory-clause", "enable-post-misched": False},
    "ilp":       {"amdgpu-sched-strategy": "max-ilp"},
    "itilp":     {"amdgpu-sched-strategy": "iterative-ilp"},
    "coexec":    {"amdgpu-sched-strategy": "coexec"},
    "clause32":  {"amdgpu-max-memory-clause": 32},
    "clause4":   {"amdgpu-max-memory-clause": 4},
    "nosm2":     {"amdgpu-expert-scheduling-mode": False},
    "nosink":    {"disable-machine-sink": True},
    "postra_bu": {"misched-postra-direction": "bottomup"},
    "relaxocc":  {"amdgpu-schedule-relaxed-occupancy": True},
}
if __name__ == "__main__":
    for arm, opts in ARMS.items():
        dst = L / arm
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
        if not opts:
            continue
        kf = dst / KF
        s = kf.read_text()
        assert s.count(OLD) == 2, s.count(OLD)
        new = ""
        if "amdgpu-expert-scheduling-mode" not in opts:
            new += OLD
        for k, v in opts.items():
            new += f"        {json.dumps(k)}: {v!r},  # L24 arm {arm}\n"
        kf.write_text(s.replace(OLD, new))
        (dst / "L24_ARM.json").write_text(json.dumps({"arm": arm, "llvm_options_delta": opts}, indent=1) + "\n")
    print("arms:", " ".join(ARMS))
