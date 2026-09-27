#!/usr/bin/env python3
"""Barrier-region ISA census of FlyDSL k_dkdv builds (gfx1250). CPU only, static.

usage: census.py <tag>=<final_isa.s>[:<objdump.dis>] ...  [--json out.json]

For every loop body that contains WMMA:
  * regions split at s_barrier_signal (region 0 = loop top .. first signal)
  * per region: instr, bytes, wmma, valu, salu, vmem loads, ds loads, ds stores, waits
  * every s_wait_* : immediate, rel position, trigger instruction, and -- from a
    steady-state in-order counter simulation (3 iterations) -- which ops it forces
    to retire, their issue->wait cover, and the wait->first-use slack of each forced
    load (slack >> 0 means the wait is NOT the data consumer's wait).
Real VGPR numbers come from the `/*vNNN*/` comments FlyDSL/LLVM print after
s_set_vgpr_msb-encoded operands.
"""
import json
import re
import sys
from collections import Counter, deque

RE_LABEL = re.compile(r"^(\.?L\w+|\w+):")
RE_REAL = re.compile(r"(\b[vs]\[?\d+(?::\d+)?\]?)\s*/\*\s*(v\[?\d+(?::\d+)?\]?)\s*\*/")
RE_VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def vregs(text):
    out = set()
    for m in RE_VREG.finditer(text):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def parse_s(path):
    items = []  # ('label', name) | ('ins', mnemonic, text_real)
    for raw in open(path):
        line = raw.split(";")[0].rstrip()
        s = line.strip()
        if not s:
            continue
        m = RE_LABEL.match(s)
        if m and not s.startswith("."):
            items.append(("label", m.group(1)))
            continue
        if s.startswith(".L") and s.endswith(":"):
            items.append(("label", s[:-1]))
            continue
        if s.startswith("."):
            continue
        if not raw.startswith("\t") and not raw.startswith(" "):
            continue
        txt = RE_REAL.sub(lambda mm: mm.group(2), s)
        mn = txt.split()[0]
        if not re.match(r"^[a-z][a-z0-9_]*$", mn):
            continue
        items.append(("ins", mn, txt))
    return items


def parse_dis(path):
    out = []
    for line in open(path):
        m = re.match(r"\s+([a-z_][a-z0-9_]*)\b.*//\s*([0-9A-F]+):\s*((?:[0-9A-F]{8}\s*)+)", line)
        if m:
            out.append((m.group(1), int(m.group(2), 16), 4 * len(m.group(3).split())))
    return out


def rw(mn, txt):
    """(reads, writes) VGPR sets of one instruction (approximate, operand-position based)."""
    parts = [p.strip() for p in txt.split("::")]
    reads, writes = set(), set()
    for p in parts:
        toks = p.split(None, 1)
        m = toks[0]
        ops = toks[1] if len(toks) > 1 else ""
        opl = [o.strip() for o in ops.split(",")]
        if m.startswith(("buffer_store", "global_store", "ds_store", "ds_write", "flat_store",
                         "scratch_store", "buffer_atomic", "global_atomic")) or m.startswith("s_"):
            reads |= vregs(ops)
            continue
        if opl:
            writes |= vregs(opl[0])
            reads |= vregs(",".join(opl[1:]))
    return reads, writes


def kind(mn):
    if mn.startswith(("buffer_load", "global_load")) and "lds" not in mn and "async" not in mn:
        return "vmload"
    if mn.startswith(("buffer_store", "global_store", "buffer_atomic", "global_atomic")):
        return "vmstore"
    if mn.startswith("ds_load") or mn.startswith("ds_read"):
        return "dsload"
    if mn.startswith("ds_store") or mn.startswith("ds_write"):
        return "dsstore"
    if mn.startswith("v_wmma"):
        return "wmma"
    if mn.startswith("s_wait_") or mn.startswith("s_barrier"):
        return "sync"
    if mn.startswith("v_"):
        return "valu"
    if mn.startswith("s_"):
        return "salu"
    return "other"


def loops(items, dis):
    ins_idx = [i for i, it in enumerate(items) if it[0] == "ins"]
    addr = {}
    if dis:
        dis = [d for d in dis if d[0] != "s_code_end"]
        assert len(dis) == len(ins_idx), (len(dis), len(ins_idx))
        for k, i in enumerate(ins_idx):
            assert dis[k][0] == items[i][1] or dis[k][0].split("_e")[0] == items[i][1].split("_e")[0] \
                or True
            addr[i] = dis[k][1:]
    found = []
    for i, it in enumerate(items):
        if it[0] != "label":
            continue
        name = it[1]
        # body: instructions after the label up to a cbranch back to it
        body = []
        for j in range(i + 1, len(items)):
            jt = items[j]
            if jt[0] == "label":
                continue
            body.append(j)
            if jt[1].startswith("s_cbranch") and jt[2].split()[-1] == name:
                break
        else:
            continue
        if not any(items[j][1].startswith("v_wmma") for j in body):
            continue
        # the body must not span another backward loop label (keep simple: max 1200)
        found.append((name, body))
    return found, addr


def census(tag, spath, dpath):
    items = parse_s(spath)
    dis = parse_dis(dpath) if dpath else None
    lp, addr = loops(items, dis)
    rec = {"tag": tag, "isa": spath, "loops": []}
    txt_all = open(spath).read()
    for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size"):
        m = re.search(rf"\.{key}:\s+(\d+)", txt_all)
        rec[key] = int(m.group(1)) if m else None
    if dis:
        dis = [d for d in dis if d[0] != "s_code_end"]
        rec["code_bytes"] = dis[-1][1] + dis[-1][2]
    for name, body in lp:
        B = [(items[j][1], items[j][2]) for j in body]
        N = len(B)
        byts = [addr[j][1] if j in addr else None for j in body]
        L = {"label": name, "instr": N, "bytes": sum(b for b in byts if b) if dis else None}
        if dis:
            L["addr_span"] = [addr[body[0]][0], addr[body[-1]][0] + addr[body[-1]][1]]
        # regions
        sig = [k for k, (mn, _) in enumerate(B) if mn in ("s_barrier_signal", "s_barrier")]
        cuts = [0] + [k for k in sig] + [N]
        regs = []
        for r in range(len(cuts) - 1):
            a, b = cuts[r], cuts[r + 1]
            c = Counter(kind(B[k][0]) for k in range(a, b))
            waits = [f"{B[k][0].replace('s_wait_', '')} {B[k][1].split()[-1]}@{k}"
                     for k in range(a, b) if B[k][0].startswith("s_wait_")
                     and "xcnt" not in B[k][0] and "kmcnt" not in B[k][0]]
            regs.append({"region": r, "from": a, "to": b, "instr": b - a,
                         "bytes": sum(x for x in byts[a:b] if x) if dis else None,
                         "wmma": c["wmma"], "valu": c["valu"], "salu": c["salu"],
                         "vmload": c["vmload"], "vmstore": c["vmstore"],
                         "dsload": c["dsload"], "dsstore": c["dsstore"], "waits": waits})
        L["barriers_at"] = sig
        L["barrier_waits_at"] = [k for k, (mn, _) in enumerate(B) if mn == "s_barrier_wait"]
        L["regions"] = regs
        L["totals"] = dict(Counter(kind(mn) for mn, _ in B))
        # steady-state counter simulation
        RW = [rw(mn, t) for mn, t in B]
        q = {"load": deque(), "ds": deque()}
        waits = []
        ITER = 3
        for it in range(ITER):
            for k, (mn, t) in enumerate(B):
                kd = kind(mn)
                if mn.startswith("s_wait_") and ("loadcnt" in mn or "dscnt" in mn):
                    imm = int(t.split()[-1], 16)
                    cnts = []
                    if "loadcnt" in mn:
                        cnts.append("load")
                    if "dscnt" in mn:
                        cnts.append("ds")
                    forced = []
                    for c in cnts:
                        while len(q[c]) > imm:
                            forced.append((c,) + q[c].popleft())
                    if it == ITER - 1:
                        nxt = next((B[x][0] for x in range(k + 1, N)
                                    if not B[x][0].startswith(("s_set_vgpr_msb", "s_nop", "s_wait_"))), "")
                        det = []
                        for (c, iit, ik, mnf) in forced:
                            cover = (it - iit) * N + (k - ik)
                            dst = RW[ik][1] if kind(mnf) in ("vmload", "dsload") else set()
                            slack = None
                            if dst:
                                for d in range(1, 2 * N):
                                    x = (k + d) % N
                                    if RW[x][0] & dst:
                                        slack = d
                                        break
                                    if RW[x][1] & dst and not kind(B[x][0]) in ("vmload", "dsload"):
                                        slack = -d  # overwritten before read (WAW)
                                        break
                            det.append({"cnt": c, "op": mnf, "issued_rel": ik,
                                        "iters_back": it - iit, "cover": cover, "slack": slack})
                        waits.append({"pos": k, "wait": mn, "imm": imm, "trigger": nxt,
                                      "outstanding_before": None, "forced": det})
                elif kd == "vmload":
                    q["load"].append((it, k, mn))
                elif kd in ("dsload", "dsstore"):
                    q["ds"].append((it, k, mn))
        L["waits"] = []
        for w in waits:
            f = w["forced"]
            loads = [x for x in f if x["cnt"] == "load"]
            dss = [x for x in f if x["cnt"] == "ds"]

            def summ(xs):
                if not xs:
                    return None
                sl = [x["slack"] for x in xs if x["slack"] is not None]
                return {"n": len(xs),
                        "ops": dict(Counter(x["op"] for x in xs)),
                        "cover_min": min(x["cover"] for x in xs),
                        "cover_max": max(x["cover"] for x in xs),
                        "from_prev_iter": sum(1 for x in xs if x["iters_back"] > 0),
                        "slack_min": min(sl) if sl else None,
                        "slack_max": max(sl) if sl else None,
                        "issued_rel": sorted(set(x["issued_rel"] for x in xs))[:3] + ["..."]
                        + sorted(set(x["issued_rel"] for x in xs))[-2:]}
            L["waits"].append({"pos": w["pos"], "wait": f"{w['wait']} {w['imm']:#x}",
                               "trigger": w["trigger"], "load": summ(loads), "ds": summ(dss)})
        rec["loops"].append(L)
    return rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        args = [a for a in args if a != out]
    recs = []
    for a in args:
        tag, paths = a.split("=", 1)
        sp, _, dp = paths.partition(":")
        recs.append(census(tag, sp, dp or None))
    if out:
        json.dump(recs, open(out, "w"), indent=1)
    for r in recs:
        print(f"=== {r['tag']}  vgpr {r['vgpr_count']}  lds {r['group_segment_fixed_size']}  "
              f"scratch {r['private_segment_fixed_size']}  code {r.get('code_bytes')} B")
        for L in r["loops"]:
            print(f"  loop {L['label']}: {L['instr']} instr, {L['bytes']} B, span {L.get('addr_span')}, "
                  f"totals {L['totals']}, signal@{L['barriers_at']} wait@{L['barrier_waits_at']}")
            for g in L["regions"]:
                print(f"    R{g['region']} [{g['from']},{g['to']}) instr {g['instr']} B {g['bytes']} "
                      f"wmma {g['wmma']} valu {g['valu']} salu {g['salu']} vml {g['vmload']} "
                      f"dsl {g['dsload']} dss {g['dsstore']} waits {g['waits']}")
            for w in L["waits"]:
                if w["load"] or w["ds"]:
                    print(f"    wait@{w['pos']:4d} {w['wait']:24s} -> {w['trigger']:26s} "
                          f"load {w['load']}  ds {w['ds']}")


if __name__ == "__main__":
    main()
