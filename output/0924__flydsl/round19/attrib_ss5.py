"""r19.i0.attrib_async -- attrib_ss.py with a FIFTH queue: ASYNCcnt.

Route row 5 (h18), owed since round 18. `attrib_ss.py` files
`global_load_async_to_lds_*` into the LOADcnt queue, which is wrong twice over:
  * those ops do NOT increment LOADcnt on gfx1250, they increment ASYNCcnt, so
    modelling them in LOADcnt makes every real `s_wait_loadcnt N` pop the wrong op
    and report a fiction (round 18: "797 cycles blocked on
    global_load_async_to_lds_b32" on the g56 ISA);
  * their consumer is `s_wait_asynccnt`, which the old model ignored entirely, so
    the real LDS-arrival wait was never charged at all.

Queues modelled here, each with its own in-order FIFO and its own latency:
    LOADcnt   buffer_load / global_load / flat_load / scratch_load    L_VM
    DScnt     ds_load / ds_store                                     L_LDS
    ASYNCcnt  global_load_async_to_lds_* / cluster_load_async_to_lds_* L_ASYNC
    STOREcnt  buffer_store / global_store  (drained, never blocking here)
    KMcnt     s_load_*                                               L_SMEM

L_ASYNC is the global->LDS trip; it is NOT the same as L_VM, because the data never
comes back to the wave. Reported for a RANGE of L_ASYNC so a candidate is priced by a
band, not by one invented constant.
"""
import re, sys, collections
sys.path.insert(0, '/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/'
                   'gfx1250-flydsl-attn-bwd-20260917-115934/rounds/015/_scratch/work')
from chain import bodies, real

ASYNCQ = ('global_load_async_to_lds', 'cluster_load_async_to_lds',
          'global_load_async', 'cluster_load_async')
LOADQ  = ('buffer_load', 'global_load', 'flat_load', 'scratch_load')
DSQ    = ('ds_load', 'ds_read', 'ds_store', 'ds_write')
SMEMQ  = ('s_load',)


def qof(op):
    if op.startswith(ASYNCQ):  return 'asynccnt'     # MUST be tested before LOADQ
    if op.startswith(LOADQ):   return 'loadcnt'
    if op.startswith(DSQ):     return 'dscnt'
    if op.startswith(SMEMQ):   return 'kmcnt'
    return None


def simulate(lines, a, b, L, WCOST, ncopy=4):
    body = [l.strip() for l in lines[a:b+1] if real(l)]
    clock = 0.0
    q = {k: [] for k in ('loadcnt', 'dscnt', 'asynccnt', 'kmcnt')}
    per_copy = []
    for c in range(ncopy):
        sites, t0 = [], clock
        for i, l in enumerate(body):
            op = l.split()[0]
            clock += WCOST if op.startswith('v_wmma') else 1
            k = qof(op)
            if k:
                q[k].append((clock + L[k], op, i, c))
            m = re.match(r's_wait_(loadcnt|dscnt|asynccnt|kmcnt)\s+(0x[0-9a-f]+|\d+)', l)
            if m:
                kind, n = m.group(1), int(m.group(2), 0)
                Q, last = q[kind], None
                while len(Q) > n:
                    last = Q.pop(0)
                jump = 0.0
                if last is not None and last[0] > clock:
                    jump = last[0] - clock
                    clock = last[0]
                if last is not None:
                    sites.append((i, kind, n, last[1], last[2], jump, c - last[3]))
        per_copy.append((len(body), clock - t0, sites))
    return per_copy


L_ASYNC_BAND = (400, 800, 1600)

for p in sys.argv[1:]:
    kname = p.split('/')[-2]
    lines, bs = bodies(p)
    for t, a, b in bs:
        body = [l.strip() for l in lines[a:b + 1] if real(l)]
        if len(body) < 200:
            continue
        nasync = sum(1 for l in body if l.split()[0].startswith(ASYNCQ))
        for WCOST in (8,):
            for LA in (L_ASYNC_BAND if nasync else (800,)):
                L = {'loadcnt': 800, 'dscnt': 100, 'asynccnt': LA, 'kmcnt': 300}
                pc = simulate(lines, a, b, L, WCOST)
                n, c, sites = pc[-1]
                by = collections.Counter()
                byq = collections.Counter()
                for (_i, kk, _n, opn, _s, j, _age) in sites:
                    by[opn] += j; byq[kk] += j
                tot = sum(by.values())
                tag = f' L_ASYNC={LA}' if nasync else ''
                print(f'\n### {kname} {t} slots={n} async_ops={nasync} '
                      f'WMMA_cost={WCOST}{tag}  STEADY cycles={c:.0f} '
                      f'stall={tot:.0f} ({100*tot/c:.1f}%)')
                for kk, j in byq.most_common():
                    if j > 0.5:
                        print(f'      queue {kk:9s} {j:8.0f} cyc')
                for opn, j in by.most_common():
                    if j > 0.5:
                        print(f'      stall {j:8.0f} cyc ({100*j/max(tot,1):5.1f}%) on {opn}')
                for (i, k, nn, opn, s, j, age) in sites:
                    if j > 0.5:
                        print(f'        site@{i:4d} {k:9s} N={nn:<4d} popped {opn:32s}'
                              f' issued@{s:4d} {age} iter(s) earlier JUMP={j:7.0f}')
