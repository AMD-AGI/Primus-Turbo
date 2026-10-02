#!/usr/bin/env python3
"""Final tables (Chinese, markdown) for the 2026-10-02 A0 e2e from the per-process JSON files.

  make_table.py <runs dir> <tag> [<tag> ...] [--op "asm_bwd=5.51,s6=5.30,r29=6.57,asm_fwd=1.26,r16=1.36"]
                [--op-label "real dumps after GEMM burst, A0 10-02"] [--layers 32]

Inputs per tag (written by analyze.sh): <tag>.steady.json (steady_arms3.py), <tag>.events.json
(attn_events.py), <tag>.trace.json (trace_breakdown2.py, optional), <tag>.clk.json, <tag>.post.txt.
--op: op-level per-CALL ms of the arms at the training operating point (from the real-dump +
GEMM-burst A/B of today's op job); used for the "expected vs measured" table:
expected per-step delta = layers x (per-call delta). bwd arms: asm_bwd, s6, r29; fwd: asm_fwd, r16.
"""
import argparse
import json
import os
import statistics as st

B0 = {"fly/asm": "1.0323 / 1.0328（B0 09-28，fwd r16 + bwd r29）"}
ARM_BWD = {"asm": "asm_bwd", "fly": "s6", "flyr29": "r29"}
ARM_FWD = {"asm": "asm_fwd", "fly": "r16", "flyr29": "r16"}


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def post(p):
    d = {}
    try:
        for line in open(p):
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    except Exception:
        pass
    return d


def f(x, n=1, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--op", default="")
    ap.add_argument("--op-label", default="")
    ap.add_argument("--layers", type=int, default=32)
    a = ap.parse_args()
    op = {k: float(v) for k, v in (x.split("=") for x in a.op.split(",") if x)} if a.op else {}
    R = {}
    for t in a.tags:
        R[t] = {k: jload(os.path.join(a.runs, f"{t}.{k}.json")) for k in ("steady", "events", "trace", "clk")}
        R[t]["post"] = post(os.path.join(a.runs, f"{t}.post.txt"))
    print("## 1. 每个进程、每个 arm 的稳态数字\n")
    print("| 进程 | arm | n | 单步 ms 中位数（IQR） | tps | 对 asm 单步比：相邻配对 / 周期 | attn fwd ms/步 | attn bwd ms/步 | FA ms/步 | FA 占单步 | 峰值显存 |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for t, r in R.items():
        s, e = r["steady"] or {}, r["events"] or {}
        for arm, x in (s.get("arms") or {}).items():
            ev = (e.get("arms") or {}).get(arm, {})
            pr = (s.get("pairs") or {}).get(f"{arm}/asm", {})
            ratio = "1" if arm == "asm" else f"{f(pr.get('adjacent_median'), 4)}（n={pr.get('adjacent_n')}）/ {f(pr.get('cycles_median'), 4)}（n={pr.get('cycles_n')}）"
            print(f"| {t} | {arm} | {x['n']} | {f(x['ms_median'])}（{f(x['ms_iqr'][0])}–{f(x['ms_iqr'][1])}） | "
                  f"{x['tps_median']:,.0f} | {ratio} | {f(ev.get('fwd_ms'), 2)} | {f(ev.get('bwd_ms'), 2)} | "
                  f"{f(ev.get('fa_ms'), 2)} | {f(100 * ev['fa_share_of_step'], 2) + '%' if ev.get('fa_share_of_step') else '—'} | "
                  f"{x.get('peak_mem_gib')} GiB（{x.get('peak_mem_pct')}%） |")
    print(f"\nB0 对照：fly/asm 单步比 {B0['fly/asm']}；B0 fly fwd 48.0–49.3 / bwd 285.4–290.0，asm fwd 38.8–39.3 / bwd 233.1–236.7 ms/步（trace）。\n")
    print("## 2. 两两对比（相邻两步、arm 不同、都在稳态窗口内；差值 = 前者 − 后者）\n")
    print("| 进程 | 对比 | 相邻 n | 单步比（相邻 / 周期） | Δ单步 ms | ΔFA ms（events） | 其中 Δfwd | 其中 Δbwd | Δ单步 − ΔFA |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for t, r in R.items():
        s, e = r["steady"] or {}, r["events"] or {}
        for k, pr in (s.get("pairs") or {}).items():
            b_, a_ = k.split("/")
            d = (e.get("pairs") or {}).get(f"{b_}/{a_}", {})
            print(f"| {t} | {b_} vs {a_} | {pr.get('adjacent_n')} | {f(pr.get('adjacent_median'), 4)} / {f(pr.get('cycles_median'), 4)} | "
                  f"{f(pr.get('adjacent_diff_ms_median'), 1, True)} | {f(d.get('fa'), 2, True)} | {f(d.get('fwd'), 2, True)} | "
                  f"{f(d.get('bwd'), 2, True)} | {f(d.get('unexplained_ms'), 1, True)} |")
    if op:
        L = a.layers
        print(f"\n## 3. 预期（op 级 × {L} 层）对实测（e2e，events）\n")
        print(f"op 级来源：{a.op_label or '（未注明）'}；每次调用 ms：" + ", ".join(f"{k} {v}" for k, v in op.items()) + "\n")
        print("| 进程 | 对比 | 预期 Δbwd ms/步 | 实测 Δbwd | 预期 Δfwd ms/步 | 实测 Δfwd | 实测 Δ单步 | 判定 |")
        print("|---|---|--:|--:|--:|--:|--:|---|")
        for t, r in R.items():
            e, s = r["events"] or {}, r["steady"] or {}
            for k, d in (e.get("pairs") or {}).items():
                b_, a_ = k.split("/")
                pb = (op.get(ARM_BWD.get(b_, "")), op.get(ARM_BWD.get(a_, "")))
                pf = (op.get(ARM_FWD.get(b_, "")), op.get(ARM_FWD.get(a_, "")))
                eb = L * (pb[0] - pb[1]) if None not in pb else None
                ef = L * (pf[0] - pf[1]) if None not in pf else None
                st_ = (s.get("pairs") or {}).get(k, {}).get("adjacent_diff_ms_median")
                verdict = "—"
                if eb is not None and d.get("bwd") is not None:
                    tol = max(5.0, 0.2 * abs(eb))
                    ok_b = abs(d["bwd"] - eb) <= tol
                    ok_s = st_ is not None and d.get("fa") is not None and abs(st_ - d["fa"]) <= max(5.0, 0.15 * abs(d["fa"]))
                    verdict = ("正常" if ok_b and ok_s else
                               ("bwd 与 op 级不符" if not ok_b else "单步差 ≠ attention 差（别处有代价）"))
                print(f"| {t} | {b_} vs {a_} | {f(eb, 1, True)} | {f(d.get('bwd'), 1, True)} | {f(ef, 1, True)} | "
                      f"{f(d.get('fwd'), 1, True)} | {f(st_, 1, True)} | {verdict} |")
    print("\n## 4. trace（kineto；A0 09-28 曾只记录 ~4 个 kernel/步，INVALID 的不用）\n")
    print("| 进程 | trace | 有效 | arm | GPU kernels | FA sum / wall ms | attn_fwd ms | attn_bwd sum / wall / span ms | gemm | gemm_aux | elementwise | idle |")
    print("|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for t, r in R.items():
        for x in (r["trace"] or []):
            b, w = x["buckets"], x["wall"]
            print(f"| {t} | {os.path.basename(os.path.dirname(x['trace']))} | {'是' if x['valid'] else '否：' + x['invalid_reason']} | "
                  f"{','.join(x['arms'])} | {x['gpu_kernels']} | {f(x['fa_path_ms'])} / {f(x['fa_path_wall_ms'])} | "
                  f"{f(b.get('attn_fwd'))} | {f(b.get('attn_bwd'))} / {f(w.get('attn_bwd'))} / "
                  f"{f(x['attn_call_span_ms'].get('attn_bwd_span'))} | {f(b.get('gemm'))} | {f(b.get('gemm_aux'))} | "
                  f"{f(b.get('elementwise'))} | {f(x['idle_ms'])} |")
    print("\n## 5. 运行健康\n")
    print("| 进程 | rc | 步数 | 非有限步 | nkfix 事件 | BLAS 改写 | watchdog | memguard | dmesg 故障 / INFO | 新 CPU MCE | sclk 负载中位数 MHz | 功耗中位数 W | 墙钟 s | loss 首 / 末 |")
    print("|---|--:|--:|--:|---|--:|---|---|--:|--:|--:|--:|--:|---|")
    for t, r in R.items():
        p, c, s = r["post"], (r["clk"] or {}).get("load", {}), r["steady"] or {}
        print(f"| {t} | {p.get('rc')} | {p.get('steps_logged')} | {p.get('nonfinite_lines')} | {p.get('nkfix_events') or '—'} | "
              f"{p.get('blas_repoints')} | {p.get('watchdog') or '—'} | {p.get('memguard') or '—'} | "
              f"{p.get('dmesg_fault_lines')} / {p.get('dmesg_info_lines')} | {p.get('cpu_mce_new')} | "
              f"{c.get('sclk_median')} | {c.get('power_w_median')} | {p.get('wall_s')} | {s.get('loss_first')} / {s.get('loss_last')} |")


if __name__ == "__main__":
    main()
