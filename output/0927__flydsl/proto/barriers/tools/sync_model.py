"""CPU proof of the L8 proto's LDS ring + barrier schedule (mirrors the kernel control flow).

For every (G, NG, split, warp order, n_iter) it replays one wave's program (all 8 waves run
the same wave-uniform control flow) and checks:
  B  barrier protocol: strict signal/wait alternation starting with a signal, #signal == #wait
     (an imbalance hangs the WG -> card wedge);
  RAW every K/V read of tile j sees its TDM write retired (tensor_wait) before a signal k
     with k <= #waits before the read (so all 8 waves' row-slices landed);
  WAR the TDM write of tile j+S (reuses tile j's slot) is issued after wait k with
     #signals before the last read of tile j < k;
  O  the epilogue O ring (group slot) is not read/written by any tile after the last wait,
     has no TDM write in flight, and its previous occupant's reads precede a signal <= #waits;
  Q  the Q staging group is not TDM-written before every wave's Q reads passed a barrier.
tensorcnt: s_wait_tensorcnt N retires all but the N newest ops (in-order counter).
"""
import itertools

OPS = 2  # TDM ops per tile per wave (K, V) at d=128


def replay(G, NG, split, lo, n):
    S, D = G * NG, G * (NG - 1)
    ev = []  # (kind, arg)
    q = []  # outstanding TDM tile ids (per op)
    retired_at = {}  # tile -> event index when retired

    def issue(j):
        q.extend([j] * OPS); ev.append(("tdm", j))

    def twait(N):
        while len(q) > N:
            j = q.pop(0)
            if j not in q:
                retired_at[j] = len(ev)
        ev.append(("tw", N))

    def cond_wait(last_needed):
        deep = OPS * G * (NG - 2)
        twait(deep if (deep and last_needed < n) else 0)

    # prologue: Q read, then K/V of tiles 0..D-1, drain, sync
    ev.append(("qread", None))
    for j in range(D):
        if j < n:
            issue(j)
    twait(0)
    if split:
        ev.append(("sig", None))
    else:
        ev.append(("sig", None)); ev.append(("wait", None))
    for i in range(n):
        top = i % G == 0

        def sync():
            if split:
                ev.append(("wait", None))
            else:
                cond_wait(i + D - 1)
                ev.append(("sig", None)); ev.append(("wait", None))

        def pf():
            for j in range(G):
                if i + D + j < n:
                    issue(i + D + j)

        if lo:
            if top: sync()
            ev.append(("read", i))  # K
            if top: pf()
        else:
            if top: sync(); pf()
            ev.append(("read", i))
        ev.append(("read", i))  # V (tr16), drained by the PV dscnt(0)
        if split and i % G == G - 1 and i + 1 < n:
            cond_wait(i + D)
            ev.append(("sig", None))
    ev.append(("oring", None))
    return ev, retired_at, S, D


def check(G, NG, split, lo, n):
    ev, ret, S, D = replay(G, NG, split, lo, n)
    sigs = waits = 0
    before = []  # (#sig, #wait) before each event
    state = "need_sig"
    for k, (kind, _) in enumerate(ev):
        before.append((sigs, waits))
        if kind == "sig":
            assert state == "need_sig", ("double signal", k); sigs += 1; state = "need_wait"
        elif kind == "wait":
            assert state == "need_wait", ("wait w/o signal", k); waits += 1; state = "need_sig"
    assert sigs == waits, ("imbalance", sigs, waits)
    # tile j occupies slot j % S; write of tile j is its tdm event
    wr = {a: k for k, (kind, a) in enumerate(ev) if kind == "tdm"}
    reads = {}
    for k, (kind, a) in enumerate(ev):
        if kind == "read":
            reads.setdefault(a, []).append(k)
    for j in range(n):
        assert j in wr or j < D or True
        # RAW: retired before some signal visible to the read
        r0 = min(reads[j])
        assert j in ret and ret[j] < r0, ("RAW-unretired", j)
        rt = ret[j]
        # first signal after retirement index = before[rt].sigs + 1 ; need it <= waits before read
        assert before[rt][0] + 1 <= before[r0][1], ("RAW", j)
        # WAR: next occupant j+S
        if j + S < n:
            w = wr[j + S]
            assert before[max(reads[j])][0] < before[w][1], ("WAR", j)
    # Q: first TDM write into group NG-1 slots must follow a wait after the Q read signal
    qk = [k for k, (kind, _) in enumerate(ev) if kind == "qread"][0]
    for j, w in wr.items():
        if (j % S) // G == NG - 1:
            assert before[qk][0] < before[w][1], ("Q-WAR", j)
    # O ring: group free = (gl + NG - 1) % NG
    ok = len(ev) - 1
    gl = (n - 1) // G
    free = (gl + NG - 1) % NG
    for j in range(n):
        if (j % S) // G == free:
            assert before[max(reads[j])][0] < before[ok][1], ("O-WAR", j)
            assert j in ret and ret[j] < ok, ("O-inflight", j)
    for j in wr:
        assert j < n
    return sigs


bad = 0
for G, NG, split, lo in itertools.product((1, 2, 4), (2, 3, 4), (0, 1), (0, 1)):
    for n in range(1, 70):
        try:
            check(G, NG, split, lo, n)
        except AssertionError as e:
            bad += 1
            if bad < 10:
                print("FAIL", G, NG, split, lo, n, e)
print("configs checked", 3 * 3 * 2 * 2 * 69, "failures", bad)
for G, NG, split in ((1, 2, 0), (2, 2, 1), (2, 3, 1)):
    print(f"G={G} NG={NG} split={split}: barriers(n=128 tiles) =", check(G, NG, split, 1, 128))
