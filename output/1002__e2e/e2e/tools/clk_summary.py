#!/usr/bin/env python3
"""Summarise a launcher clock CSV (t,busy_pct,sclk_mhz,fclk_mhz,power_uw,temp_mc; 5 s samples, card1).

  clk_summary.py <clk.csv> [--busy 50] [--json out.json]

Prints the median / p10 / p90 sclk and board power over samples with busy > --busy (A0 idles at
busy 38 % / sclk 2355 MHz / ~1.13 kW, so the default 50 % keeps only loaded samples), plus the
idle-sample medians. B0's training ran at sclk ~1340-1370 MHz under a ~2.13 kW ceiling.
"""
import argparse
import csv
import json
import statistics as st


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--busy", type=float, default=50.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    load, idle = [], []
    for r in csv.DictReader(open(a.csv)):
        try:
            b = float(r.get("busy_pct") or "nan")
            s = float(r.get("sclk_mhz") or "nan")
        except ValueError:
            continue
        if s != s or b != b:
            continue
        p = r.get("power_uw") or ""
        rec = {"busy": b, "sclk": s, "w": float(p) / 1e6 if p.strip() else None}
        (load if b > a.busy else idle).append(rec)
    out = {"csv": a.csv, "n_load": len(load), "n_idle": len(idle)}
    for name, rs in (("load", load), ("idle", idle)):
        sc = [r["sclk"] for r in rs]
        pw = [r["w"] for r in rs if r["w"] is not None]
        out[name] = {"sclk_median": st.median(sc) if sc else None, "sclk_p10": pct(sc, .1),
                     "sclk_p90": pct(sc, .9), "power_w_median": st.median(pw) if pw else None,
                     "power_w_p90": pct(pw, .9)}
        print(f"{name:5s} n={len(rs):4d} sclk median {out[name]['sclk_median']} MHz "
              f"(p10 {out[name]['sclk_p10']}, p90 {out[name]['sclk_p90']})  power median "
              f"{out[name]['power_w_median']} W (p90 {out[name]['power_w_p90']})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
