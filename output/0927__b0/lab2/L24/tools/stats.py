"""Per arm/config: resources from 22_final_isa.s, ISA hash (comments/labels stripped), op-class counts, diff vs base."""
import collections, glob, hashlib, json, pathlib, re, sys
L = pathlib.Path(__file__).resolve().parent.parent
KEYS = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size", "group_segment_fixed_size")
def load(arm, cfg):
    f = glob.glob(str(L / "isa" / arm / cfg / "*" / "22_final_isa.s"))
    if not f: return None
    txt = open(f[0]).read()
    res = {k: int(m.group(1)) for k in KEYS for m in [re.search(rf"\.{k}:\s+(\d+)", txt)] if m}
    ops = []
    for ln in txt.split("\n"):
        s = ln.strip()
        if not s or s.startswith((".", ";", "//")) or s.endswith(":"): continue
        ops.append(s.split(";")[0].split("//")[0].strip())
    body = "\n".join(ops)
    res["n_inst"] = len(ops)
    res["hash"] = hashlib.sha256(body.encode()).hexdigest()[:12]
    op = collections.Counter(o.split()[0] for o in ops)
    res["mem"] = {k: sum(v for o, v in op.items() if o.startswith(k)) for k in
                  ("buffer_load", "buffer_store", "global_", "tensor_", "ds_load", "ds_store", "v_wmma", "s_wait", "s_barrier", "s_clause", "s_delay_alu", "v_nop", "s_nop")}
    res["ops"] = op
    return res
arms = sys.argv[1:]
cfgs = ["prod", "proxy", "fast", "nc_g4", "c_g1", "nc_g1"]
out = {}
for a in arms:
    for c in cfgs:
        r = load(a, c)
        if r is None: continue
        b = load("base", c)
        # multiset of memory/sync op mnemonics must equal base (the option may only reorder, never add/drop memory ops)
        memkeys = lambda o: {k: v for k, v in o.items() if k.startswith(("buffer_", "global_", "tensor_", "ds_", "s_barrier", "v_wmma"))}
        r["memops_equal_base"] = memkeys(r["ops"]) == memkeys(b["ops"])
        r["same_as_base"] = r["hash"] == b["hash"]
        del r["ops"]
        out[f"{a}/{c}"] = r
        print(f"{a:10s} {c:6s} vgpr={r.get('vgpr_count')} sgpr={r.get('sgpr_count')} vspill={r.get('vgpr_spill_count')} sspill={r.get('sgpr_spill_count')} scratch={r.get('private_segment_fixed_size')} lds={r.get('group_segment_fixed_size')} n={r['n_inst']} hash={r['hash']} same_base={r['same_as_base']} memops_eq={r['memops_equal_base']} wait={r['mem']['s_wait']} clause={r['mem']['s_clause']} vnop={r['mem']['v_nop']}")
json.dump(out, open(L / "isa" / "stats.json", "w"), indent=1)
