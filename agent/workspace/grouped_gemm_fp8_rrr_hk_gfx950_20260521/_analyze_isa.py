import sys, re
fn = sys.argv[1]
lines = open(fn).readlines()
n = len(lines)
# Find first and last mfma
first_mfma = next((i for i, l in enumerate(lines) if "v_mfma_f32_16x16x128" in l), None)
last_mfma = next((n - 1 - i for i, l in enumerate(reversed(lines)) if "v_mfma_f32_16x16x128" in l), None)
print(f"file: {fn}  lines={n}")
print(f"first mfma line: {first_mfma}  last mfma line: {last_mfma}")

# Count instructions per region
def cnt(pred, start, end):
    return sum(1 for l in lines[start:end] if pred(l))

regions = [("init", 0, first_mfma),
           ("main+epilog", first_mfma, last_mfma + 1),
           ("post-mma", last_mfma + 1, n)]

for name, s, e in regions:
    wait = cnt(lambda l: "s_waitcnt" in l, s, e)
    barr = cnt(lambda l: "s_barrier" in l, s, e)
    scratch = cnt(lambda l: "scratch_" in l, s, e)
    mfma = cnt(lambda l: "v_mfma_f32_16x16x128" in l, s, e)
    dsr64 = cnt(lambda l: "ds_read_b64_tr_b8" in l, s, e)
    dsr128 = cnt(lambda l: "ds_read_b128" in l, s, e)
    print(f"  {name:15s} [{s}-{e}]: waitcnt={wait} barrier={barr} scratch={scratch} mfma={mfma} ds_read_b64={dsr64} ds_read_b128={dsr128}")
