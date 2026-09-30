import re,sys
for l in open(sys.argv[1]):
    if not l.startswith('MB '): continue
    kv=dict(x.split('=') for x in l.split()[2:]); name=l.split()[1]
    ev=float(kv['ev_ms']); cyc=int(kv['cyc_med']); rt=int(kv['rt_med'])
    print(f"{name:9s} wps={kv['wps']} ev_ms={ev:7.4f} cyc={cyc:8d} f_rt={cyc/rt*100:6.0f}MHz f_ev={cyc/ev/1e3:6.0f}MHz cyc/op/simd={float(kv['cyc_per_op_simd']):6.3f}")
