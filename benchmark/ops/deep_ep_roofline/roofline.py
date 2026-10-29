###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Roofline of the flydsl_v2 dispatch and combine from the JSON of bench_ops.py and bench_ceiling.py; no GPU,
# standard library only. Prints Markdown tables (docs/flydsl_deepep_mi455_roofline.md explains them).
#
#   python3 roofline.py --ops ops.json [--tsweep tsweep.json] [--ceiling ceiling.json] [--ref-rows 8192]
#
# Bytes per rank and kernel (bf16 rows of RB = 2 * H bytes; per-row metadata, under 1%, not counted):
#   link     the larger direction: rows to peers (rows_out) or from peers (rows_in)
#   dispatch HBM: read x once per token (T); this rank's own rows written straight into recv_x (rows_self);
#            the peers' rows written into the heap, then read from it and written into recv_x (3 x rows_in)
#   combine  HBM: read the rows pushed back (rows_in), the heap writes of the rows coming back (rows_out),
#            the reduce reading this rank's own rows and the returned ones (rows_self + rows_out), the output (T)
#   FLOPs    dispatch none; combine one fp32 add per element beyond the first row of a token
# rows_self (rows a rank keeps) is in the layout record of bench_ops.py; older JSON without it gets
# rows_in / (world - 1), which holds for routing that treats the ranks alike.
import argparse
import json
from collections import defaultdict

# AMD Instinct MI455X, peak (amd.com product page and brochure): 23.3 TB/s HBM4; 3.6 TB/s bidirectional
# scale-up per GPU (36 links x 400 Gbit/s each way, so 1.8 TB/s each way). FP32: the combine's plain v_add_f32
# peaks at 256 CUs x 2.4 GHz x 4 SIMDs x 32 lanes = 78.6 TFLOP/s (157 with FMA, 315 packed, the spec value).
PEAK_HBM = 23.3e12
PEAK_LINK = 1.8e12
PEAK_VALU = 78.6e12

DISPATCH_OPS = ("dispatch_sync", "dispatch_cached", "dispatch_worst")


def load(path):
    with open(path) as f:
        return json.load(f)


def account(op, T, rows_in, rows_out, rows_self, H):
    rb = 2 * H
    link = max(rows_in, rows_out) * rb
    if op == "combine":
        hbm = (rows_in + rows_out + rows_self + rows_out + T) * rb
        flops = max(rows_self + rows_out - T, 0) * H
    else:
        hbm = (T + rows_self + 3 * rows_in) * rb
        flops = 0
    return link, hbm, flops


class Ceilings:
    def __init__(self, data, ref_rows):
        rows = [r for r in data["rows"] if r.get("op") != "check" and "gbps" in r]
        self.link = {
            r["blocks"]: r["gbps"] * 1e9
            for r in rows
            if r["method"] == "tdm_uncached" and r["direction"] == "bi" and r["rows"] == ref_rows
        }
        self.copy = {
            r["blocks"]: r["gbps"] * 1e9
            for r in rows
            if r["method"] == "tdm_self" and r["direction"] == "bi" and r["rows"] == ref_rows
        }
        refs = [
            r for r in rows if r["method"] in ("tdm_uncached", "sdma", "hipmemcpy") and r["direction"] == "bi"
        ]
        self.link_best = max(r["gbps"] for r in refs) * 1e9
        self.copy_best = max(r["gbps"] for r in rows if r["method"] == "tdm_self") * 1e9

    def at(self, table, blocks):
        return table.get(blocks)


def layouts(data):
    out = {}
    world = data["world"]
    for r in data["rows"]:
        if r.get("op") == "layout" and r.get("impl") == "flydsl_v2":
            rows_self = r.get("rows_self", r["rows_in"] // max(world - 1, 1))
            out[(r["T"], r["sms"])] = (r["rows_in"], r["rows_out"], rows_self)
    return out


def times(data, impl="flydsl_v2"):
    out = {}
    for r in data["rows"]:
        if r.get("impl") == impl and "us" in r and r.get("op") not in ("layout", "check"):
            out[(r["op"], r["T"], r["sms"])] = r["us"]
    return out


def pct(x):
    return f"{100 * x:.0f}%"


def roofline_table(data, ceil):
    H = data["H"]
    lay, t = layouts(data), times(data)
    print(f"### Roofline per kernel: EP{data['world']}, hidden {H} bf16, top-{data['K']} of {data['E']}\n")
    print(
        "Bounds at the MI455X peaks: link 1.8 TB/s each way, HBM 23.3 TB/s, FP32 add 78.6 TFLOP/s. "
        '"measured link": the best copy (copy engines or the TDM copy kernel) over the link.\n'
    )
    print(
        "| T | CUs | op | µs | link MB | HBM MB | FLOP/link B | link GB/s | of link peak | of measured link | "
        "link-peak µs | HBM-peak µs | FLOP-peak µs |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (op, T, n), us in sorted(t.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        if (T, n) not in lay:
            continue
        link, hbm, flops = account(op, T, *lay[(T, n)], H)
        gbps = link / (us * 1e-6)
        meas = f" {pct(gbps / ceil.link_best)} |" if ceil else " – |"
        print(
            f"| {T} | {n} | {op} | {us:.1f} | {link / 1e6:.1f} | {hbm / 1e6:.1f} | {flops / link:.2f} | "
            f"{gbps / 1e9:.0f} | {pct(gbps / PEAK_LINK)} |{meas} {link / PEAK_LINK * 1e6:.1f} | "
            f"{hbm / PEAK_HBM * 1e6:.1f} | {flops / PEAK_VALU * 1e6:.2f} |"
        )
    print()


def phase_table(data, ceil, ref_rows):
    """Dispatch as the send at the link ceiling plus its local writes at the local copy rate (its own rows, stored
    by the same warps during the send, and the peers' rows, copied out of the heap after it), both
    measured with the same block count; the combine against the send alone."""
    H = data["H"]
    rb = 2 * H
    lay, t = layouts(data), times(data)
    print(f"### Phases at the measured ceilings (same block count; ceilings at {ref_rows} rows per peer)\n")
    print(
        "| T | CUs | op | µs | send at link ceiling µs | own rows + receive copy at local copy rate µs | model µs | "
        "measured − model µs |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|")
    for (op, T, n), us in sorted(t.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        if op not in ("dispatch_cached", "combine") or (T, n) not in lay:
            continue
        link_bw, copy_bw = ceil.at(ceil.link, n), ceil.at(ceil.copy, n)
        if not link_bw or not copy_bw:
            continue
        rows_in, rows_out, rows_self = lay[(T, n)]
        send = max(rows_in, rows_out) * rb / link_bw * 1e6
        copy = (rows_in + rows_self) * rb / copy_bw * 1e6 if op == "dispatch_cached" else 0.0
        copy_s = f"{copy:.1f}" if op == "dispatch_cached" else "– (overlapped reduce)"
        print(
            f"| {T} | {n} | {op} | {us:.1f} | {send:.1f} | {copy_s} | {send + copy:.1f} | {us - send - copy:.1f} |"
        )
    print()


def fit(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - slope * mx, slope


def latency_table(data):
    """time = fixed + link bytes / bandwidth over the token sweep; the crossover is where both halves are equal."""
    H = data["H"]
    rb = 2 * H
    lay, t = layouts(data), times(data)
    series = defaultdict(list)
    for (op, T, n), us in t.items():
        if (T, n) in lay:
            rows_in, rows_out, _ = lay[(T, n)]
            series[(op, n)].append((T, max(rows_in, rows_out) * rb, us))
    print("### Fixed cost and streaming rate (fit of time = fixed + link bytes / bandwidth over T)\n")
    print("| op | CUs | fixed µs | streaming GB/s | of link peak | half the time is fixed at T ≈ | points |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for (op, n), pts in sorted(series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len(pts) < 3:
            continue
        fixed, slope = fit([(b, us) for _, b, us in pts])
        bw = 1 / slope * 1e6
        bytes_per_token = sum(b / T for T, b, _ in pts) / len(pts)
        crossover = fixed * 1e-6 * bw / bytes_per_token
        print(
            f"| {op} | {n} | {fixed:.1f} | {bw / 1e9:.0f} | {pct(bw / PEAK_LINK)} | {crossover:,.0f} | {len(pts)} |"
        )
    print()


def ceiling_table(ceil):
    print("### Measured ceilings (`bench_ceiling.py`, both directions busy, per GPU and direction)\n")
    print(
        f"Best link copy: {ceil.link_best / 1e9:.0f} GB/s ({pct(ceil.link_best / PEAK_LINK)} of 1.8 TB/s). "
        f"Best local copy (TDM): {ceil.copy_best / 1e9:.0f} GB/s copied = {2 * ceil.copy_best / 1e12:.1f} TB/s "
        f"of HBM traffic ({pct(2 * ceil.copy_best / PEAK_HBM)} of 23.3 TB/s).\n"
    )
    print("| blocks | TDM copy over the link GB/s | per block GB/s | local TDM copy GB/s |")
    print("|---:|---:|---:|---:|")
    for b in sorted(ceil.link):
        c = ceil.copy.get(b)
        print(
            f"| {b} | {ceil.link[b] / 1e9:.0f} | {ceil.link[b] / b / 1e9:.1f} | {c / 1e9:.0f} |"
            if c
            else f"| {b} | {ceil.link[b] / 1e9:.0f} | {ceil.link[b] / b / 1e9:.1f} | – |"
        )
    print()


def rccl_rows(paths):
    """Timed rows of bench_rccl_tune.py JSON files, each labelled with its variant (buffers, launch, the
    process-wide NCCL_CTA_POLICY and RCCL_FORCE_CE) and its link bytes (the larger direction)."""
    out = []
    for path in paths:
        d = load(path)
        env = d.get("rccl_env", {})
        tag = f"CTA {env.get('NCCL_CTA_POLICY', '0')}"
        if "RCCL_FORCE_CE" in env:
            tag += f", CE {env['RCCL_FORCE_CE']}"
        if d.get("flush_mb") == 0:
            tag += ", no L2 flush"
        for r in d["rows"]:
            if "us" in r:
                out.append(
                    dict(
                        r,
                        world=d["world"],
                        routing=d["routing"],
                        E=d["E"],
                        H=d["H"],
                        variant=f"{r['impl'].removeprefix('rccl_')}, {r['launch']}, {tag}",
                        link=max(r["xgmi_in"], r["xgmi_out"]),
                    )
                )
    return out


# flydsl_v2 op -> the RCCL op that moves the same rows ("dedup": one row per token and destination rank)
RCCL_OF = {
    "dispatch_cached": "dedup_dispatch",
    "dispatch_sync": "dedup_dispatch+counts",
    "combine": "dedup_combine",
}


def rccl_table(data, rccl, cus):
    """flydsl_v2 at `cus` CUs against the default and the fastest RCCL variant moving the same rows."""
    world = data["world"]
    lay, t = layouts(data), times(data)
    same = [
        r for r in rccl if r["world"] == world and r["routing"] == data["routing"] and r["E"] == data["E"]
    ]
    print(
        f"### flydsl_v2 against tuned RCCL: EP{world}, {data['routing']} routing, top-{data['K']} of {data['E']}\n"
    )
    print(
        "RCCL: every buffer kind × launch × NCCL_CTA_POLICY / RCCL_FORCE_CE measured; the default is "
        "`dist.all_to_all_single` (plain_torch, eager, CTA 0). Link GB/s = link bytes (the larger direction) / time "
        "(nccl-tests busbw).\n"
    )
    print(
        "| T | op | flydsl_v2 µs | link GB/s | of 1.8 TB/s | RCCL default µs | fastest RCCL µs | variant | "
        "link GB/s | of 1.8 TB/s | RCCL / flydsl_v2 |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for (op, T, n), us in sorted(t.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if n != cus or op not in RCCL_OF or (T, n) not in lay:
            continue
        rows_in, rows_out, _ = lay[(T, n)]
        link = max(rows_in, rows_out) * 2 * data["H"]
        cand = [r for r in same if r["op"] == RCCL_OF[op] and r["T"] == T]
        if not cand:
            continue
        best = min(cand, key=lambda r: r["us"])
        dflt = [r["us"] for r in cand if r["variant"] == "plain_torch, eager, CTA 0"]
        dflt_s = f"{min(dflt):.1f}" if dflt else "–"
        g, gb = link / (us * 1e-6), best["link"] / (best["us"] * 1e-6)
        print(
            f"| {T} | {op} | {us:.1f} | {g / 1e9:.0f} | {pct(g / PEAK_LINK)} | {dflt_s} | {best['us']:.1f} | "
            f"{best['variant']} | {gb / 1e9:.0f} | {pct(gb / PEAK_LINK)} | {best['us'] / us:.2f}× |"
        )
    eq = [r for r in same if r["op"] == "equal_alltoall"]
    if eq:
        print(
            "\nEqual-split `ncclAllToAll` (the collective of 06-rccl's busbw roofline), fastest measured variant per "
            "volume:\n"
        )
        print("| T | link MB | µs | variant | link GB/s | of 1.8 TB/s |")
        print("|---:|---:|---:|---|---:|---:|")
        by = defaultdict(list)
        for r in eq:
            by[(r["T"], round(r["link"] / 1e6))].append(r)
        for (T, mb), rs in sorted(by.items()):
            b = min(rs, key=lambda r: r["us"])
            g = b["link"] / (b["us"] * 1e-6)
            print(f"| {T} | {mb} | {b['us']:.1f} | {b['variant']} | {g / 1e9:.0f} | {pct(g / PEAK_LINK)} |")
    print()


def rccl_latency_table(rccl):
    """time = fixed + link bytes / bandwidth per RCCL variant and op, over the files' token counts."""
    series = defaultdict(lambda: defaultdict(list))
    for r in rccl:
        series[(r["world"], r["routing"], r["op"], r["variant"])][r["link"]].append(r["us"])
    print("### RCCL: fixed cost and streaming rate (fit of time = fixed + link bytes / bandwidth)\n")
    print("| EP | routing | op | variant | fixed µs | streaming GB/s | of link peak | points |")
    print("|---:|---|---|---|---:|---:|---:|---:|")
    for (world, routing, op, variant), pts in sorted(series.items()):
        # RCCL's time is not monotonic in the size everywhere (EP4: 256 MB per rank took longer than 352 MB)
        if len(pts) < 5:
            continue
        fixed, slope = fit([(b, sum(v) / len(v)) for b, v in pts.items()])
        bw = 1 / slope * 1e6
        print(
            f"| {world} | {routing} | {op} | {variant} | {fixed:.1f} | {bw / 1e9:.0f} | {pct(bw / PEAK_LINK)} | "
            f"{len(pts)} |"
        )
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", required=True, help="bench_ops.py JSON (T = 8192 CU sweep)")
    p.add_argument("--tsweep", help="bench_ops.py JSON with a TOKENS_LIST sweep")
    p.add_argument("--ceiling", help="bench_ceiling.py JSON")
    p.add_argument(
        "--ref-rows", type=int, default=8192, help="rows per peer of the ceiling sweep to compare with"
    )
    p.add_argument("--rccl", nargs="+", help="bench_rccl_tune.py JSON files (variants, token sweeps)")
    p.add_argument("--rccl-cus", type=int, default=160, help="flydsl_v2 CU count to set against RCCL")
    a = p.parse_args()
    ops = load(a.ops)
    ceil = Ceilings(load(a.ceiling), a.ref_rows) if a.ceiling else None
    if ceil:
        ceiling_table(ceil)
    roofline_table(ops, ceil)
    if ceil:
        phase_table(ops, ceil, a.ref_rows)
    if a.tsweep:
        latency_table(load(a.tsweep))
    if a.rccl:
        rccl = rccl_rows(a.rccl)
        rccl_table(ops, rccl, a.rccl_cus)
        rccl_latency_table(rccl)


if __name__ == "__main__":
    main()
