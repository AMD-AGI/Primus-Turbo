"""ISA dump of every config of a sweep file: python isa_sweep.py <configs.py> <out_dir>.
One process per config (isa_dump_bwd.py); prints the loop counts of each dkdv ISA."""
import glob
import os
import runpy
import subprocess
import sys

cfgs = runpy.run_path(sys.argv[1])["CONFIGS"]
out = sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
counts = os.path.join(here, "..", "rounds", "round-10", "artifacts", "isa_loop_counts.py")
for i, (label, over) in enumerate(cfgs):
    d = os.path.join(out, "c%02d" % i)
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d)
    args = [sys.executable, os.path.join(here, "isa_dump_bwd.py")] + ["%s=%r" % kv for kv in over.items()]
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1200)
    f = glob.glob(os.path.join(d, "*mla_dkdv_kernel*", "22_final_isa.s"))
    if p.returncode or not f:
        print("%-58s FAIL %s" % (label, (p.stderr or p.stdout).strip().splitlines()[-1:]), flush=True)
        continue
    r = subprocess.run([sys.executable, counts, f[0]], capture_output=True, text=True).stdout.strip()
    print("%-58s %s" % (label, r.split(" spill ", 1)[-1]), flush=True)
