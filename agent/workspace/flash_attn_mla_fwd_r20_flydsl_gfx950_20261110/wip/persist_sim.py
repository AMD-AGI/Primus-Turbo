"""Makespan model of the work-item schedule: hardware dispatch vs static persistent assignments.

Item cost = FIX + live KV tiles (64 wide); the item order is the kernel's XCD-grouped LPT order.
python wip/persist_sim.py [FIX_tiles] [PERSIST_FIX_tiles]
"""

import heapq
import sys

NUM_XCD, CUS = 8, 256
XCD_Q_GROUPS, XCD_TAIL_Q = (32, 16, 8), 8
SHAPES = [
    ("dsv3", 2, 4096, 128),
    ("dsv2lite", 8, 4096, 16),
    ("kimik2", 4, 4096, 64),
    ("kimik3", 2, 7168, 96),
    ("b1_s4k", 1, 4096, 128),
    ("b1_s8k", 1, 8192, 128),
    ("b1_s16k", 1, 16384, 128),
    ("b1_s32k", 1, 32768, 128),
    ("b1_s64k", 1, 65536, 128),
    ("b2_s4k", 2, 4096, 128),
    ("b2_s16k", 2, 16384, 128),
]


def item_of(lin, nh, nq):
    """(head, q_block) of in-batch linear id `lin`, as DualwaveKernelContext._setup maps it."""
    hpx = nh // NUM_XCD
    slot = lin // NUM_XCD
    nq_head = nq - XCD_TAIL_Q if nq > XCD_TAIL_Q else 0
    lo = 0
    h_local = q_rank = None
    for i, g in enumerate(XCD_Q_GROUPS):
        u = slot - lo * hpx
        grp = u // (g * hpx)
        r = u - grp * (g * hpx)
        h_sec, q_sec = r // g, lo + grp * g + r % g
        if i == 0 or slot >= lo * hpx:
            h_local, q_rank = h_sec, q_sec
        lo = lo + (nq_head - lo) // g * g
    if slot >= lo * hpx:
        t = slot - lo * hpx
        h_local, q_rank = t % hpx, lo + t // hpx
    return h_local * NUM_XCD + lin % NUM_XCD, nq - 1 - q_rank


def tiles(qb):
    n = (qb * 256 + 256 + 63) // 64
    return max(4, (n + 1) // 2 * 2)


def items(b, s, h):
    nq = (s + 255) // 256
    out = []
    for z in range(b):
        for lin in range(h * nq):
            _, qb = item_of(lin, h, nq)
            out.append(tiles(qb))
    return out


def dynamic(costs):
    """Hardware: in-order dispatch, each XCD to its first free CU (round-robin over XCDs)."""
    per_xcd = [[] for _ in range(NUM_XCD)]
    for i, c in enumerate(costs):
        per_xcd[i % NUM_XCD].append(c)
    span = 0
    for seq in per_xcd:
        heap = [0.0] * (CUS // NUM_XCD)
        for c in seq:
            t = heapq.heappop(heap)
            heapq.heappush(heap, t + c)
        span = max(span, max(heap))
    return span


def static(costs, snake):
    g = min(CUS, len(costs))
    u_n = g // NUM_XCD
    load = [0.0] * g
    for w in range(g):
        x, u = w % NUM_XCD, w // NUM_XCD
        k = 0
        while True:
            s = k * u_n + ((u_n - 1 - u) if (snake and k % 2) else u)
            lin = s * NUM_XCD + x
            if lin >= len(costs):
                break
            load[w] += costs[lin]
            k += 1
    return max(load)


def chunked(costs, m, fix, pfix):
    """Each CTA runs m consecutive slots of its XCD's sequence; CTAs dispatched as hardware does."""
    per_xcd = [[] for _ in range(NUM_XCD)]
    for i, c in enumerate(costs):
        per_xcd[i % NUM_XCD].append(c)
    span = 0
    for seq in per_xcd:
        heap = [0.0] * (CUS // NUM_XCD)
        for i in range(0, len(seq), m):
            chunk = seq[i : i + m]
            c = fix + sum(chunk) + pfix * (len(chunk) - 1)
            t = heapq.heappop(heap)
            heapq.heappush(heap, t + c)
        span = max(span, max(heap))
    return span


def atomic(costs, fix, pfix):
    """Persistent CTAs (one per CU) pull the next slot of their XCD's sequence from a counter."""
    per_xcd = [[] for _ in range(NUM_XCD)]
    for i, c in enumerate(costs):
        per_xcd[i % NUM_XCD].append(c)
    span = 0
    for seq in per_xcd:
        heap = [(0.0, True)] * (CUS // NUM_XCD)
        for c in seq:
            t, first = heapq.heappop(heap)
            heapq.heappush(heap, (t + (fix if first else pfix) + c, False))
        span = max(span, max(t for t, _ in heap))
    return span


fix = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0
pfix = float(sys.argv[2]) if len(sys.argv) > 2 else fix
print(f"item cost = {fix} (dynamic) / {pfix} (persistent) + tiles; makespan vs current dispatch")
for name, b, s, h in SHAPES:
    t = items(b, s, h)
    dyn = dynamic([fix + x for x in t])
    sn = static([pfix + x for x in t], True)
    row = f"{name:9s} items/CU {len(t) / CUS:5.1f} snake {dyn / sn - 1:+6.2%}"
    for m in (2, 4, 8):
        row += f" chunk{m} {dyn / chunked(t, m, fix, pfix) - 1:+6.2%}"
    row += f" atomic {dyn / atomic(t, fix, pfix) - 1:+6.2%}"
    print(row)
