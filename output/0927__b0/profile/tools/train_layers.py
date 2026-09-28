"""CSV + per-layer summary from timeline.py JSONs of prof_t1 (steps 10/30/50 fly, 20/40/60 asm)."""
import json, sys
P = sys.argv[1]
out = open(P + "/data/train_layer_timeline.csv", "w")
out.write("step,arm,layer,kernel,start_ms,dur_ms,gap_prev_us,prev_kernel,gap_next_us,next_kernel,overlap_us,sclk_mhz,power_w\n")
for it, arm in ((10, 'fly'), (20, 'asm'), (30, 'fly'), (40, 'asm'), (50, 'fly'), (60, 'asm')):
    d = json.load(open(f"{P}/runs/tl_t1_{it}.json"))
    for r in d['rows']:
        out.write(f"{it},{arm},{r['layer']},{r['name']},{r['start_ms']:.3f},{r['dur_ms']:.4f},{r['gap_prev_us']:.1f},{r['prev']},"
                  f"{r['gap_next_us']:.1f},{r['next']},{r['overlap_us']:.0f},{r.get('sclk_mean', 0):.0f},{r.get('pwr_mean_w', 0):.0f}\n")
out.close()
fs, as_ = int(sys.argv[2]), int(sys.argv[3])
f = json.load(open(f"{P}/runs/tl_t1_{fs}.json"))['rows']; a = json.load(open(f"{P}/runs/tl_t1_{as_}.json"))['rows']
get = lambda rows, p: {r['layer']: r for r in rows if r['name'].startswith(p)}
ff, fa, dk, dq, de = get(f, 'kn_fmha'), get(a, 'fmha_bf16'), get(f, 'k_dkdv'), get(f, 'k_dq'), get(f, 'k_delta')
ab, od, dc = get(a, 'fmha_bwd_hd128_bf16'), get(a, 'fmha_bwd_hd128_odo'), get(a, 'fmha_bwd_hd128_dq_conv')
print(f"step fly={fs} asm={as_}")
print("L | fly_fwd(sclk) asm_fwd(sclk) f/a | fly_bwd(sclk) asm_bwd(sclk) f/a | start ms fwd fly/asm, bwd fly/asm")
tf = ta = tbf = tba = 0
for L in range(32):
    fb = dk[L]['dur_ms'] + dq[L]['dur_ms'] + de[L]['dur_ms']; abw = ab[L]['dur_ms'] + od[L]['dur_ms'] + dc[L]['dur_ms']
    tf += ff[L]['dur_ms']; ta += fa[L]['dur_ms']; tbf += fb; tba += abw
    print(f"{L:2d} | {ff[L]['dur_ms']:.3f}({ff[L]['sclk_mean']:.0f}) {fa[L]['dur_ms']:.3f}({fa[L]['sclk_mean']:.0f}) {ff[L]['dur_ms']/fa[L]['dur_ms']:.2f} | "
          f"{fb:.3f}({dk[L]['sclk_mean']:.0f}) {abw:.3f}({ab[L]['sclk_mean']:.0f}) {fb/abw:.3f} | {ff[L]['start_ms']:.1f}/{fa[L]['start_ms']:.1f} {dk[L]['start_ms']:.1f}/{ab[L]['start_ms']:.1f}")
print(f"sum fwd fly {tf:.1f} asm {ta:.1f} (+{tf-ta:.1f})  bwd kernels fly {tbf:.1f} asm {tba:.1f} (+{tbf-tba:.1f})")
