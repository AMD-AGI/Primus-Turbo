"""Static WAR screen: at every s_barrier_signal, how many DS ops were issued (textual order)
since the last `s_wait_dscnt 0x0`, and the dscnt wait preceding it. Also counts wait/sched stats."""
import re, sys, collections
for path in sys.argv[1:]:
    lines = open(path).read().splitlines()
    since0, last_wait, rep = 0, None, []
    cnt = collections.Counter()
    for i, l in enumerate(lines):
        s = l.strip()
        op = s.split()[0] if s and not s.startswith(('.', ';', '/')) else ''
        if op.startswith('ds_'): since0 += 1; cnt['ds_ops'] += 1
        if op == 's_wait_dscnt':
            cnt['s_wait_dscnt'] += 1; last_wait = s.split()[1]
            if s.split()[1] == '0x0': since0 = 0; cnt['dscnt0'] += 1
        if op == 's_wait_tensorcnt': cnt['s_wait_tensorcnt'] += 1
        if op.startswith('v_wmma'): cnt['wmma'] += 1
        if op == 's_barrier_signal':
            cnt['barriers'] += 1
            rep.append(f"L{i+1}:ds_since_dscnt0={since0},last_dscnt={last_wait}")
    tot = sum(1 for l in lines if l.startswith('\t') and not l.strip().startswith(('.', ';')))
    print('/'.join(path.split('/')[:2]), dict(cnt), 'lines', tot)
    print('   ', ' '.join(rep))
