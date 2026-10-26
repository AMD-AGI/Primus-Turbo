"""probe_hwid.py GRID SPIN_US WARPS: per-workgroup raw HW_ID1 / HW_ID2 + realtime start, to map blockIdx -> hardware placement on gfx1250."""
import sys, torch, triton, triton.language as tl

GRID, SPIN_US, WARPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


@triton.jit
def probe(out, spin):
    pid = tl.program_id(0)
    hw1 = tl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_HW_ID1)", "=s,s", [pid], dtype=tl.int32, is_pure=False, pack=1)
    hw2 = tl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_HW_ID2)", "=s,s", [pid], dtype=tl.int32, is_pure=False, pack=1)
    t0 = tl.inline_asm_elementwise("s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0", "=s,s", [pid],
                                   dtype=tl.int64, is_pure=False, pack=1)
    t = t0
    while t - t0 < spin:
        t = tl.inline_asm_elementwise("s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0", "=s,s", [pid],
                                      dtype=tl.int64, is_pure=False, pack=1)
    tl.store(out + pid * 4 + 0, hw1.to(tl.int64))
    tl.store(out + pid * 4 + 1, hw2.to(tl.int64))
    tl.store(out + pid * 4 + 2, t0)
    tl.store(out + pid * 4 + 3, t)


out = torch.zeros(GRID * 4, dtype=torch.int64, device="cuda")
for _ in range(3):
    probe[(GRID,)](out, SPIN_US * 100, num_warps=WARPS)
torch.cuda.synchronize()
r = out.view(GRID, 4).cpu()
hw1, hw2 = r[:, 0] & 0xFFFFFFFF, r[:, 1] & 0xFFFFFFFF
t0 = (r[:, 2] - r[:, 2].min()).double() / 100
varying1 = int(torch.bitwise_or.reduce(hw1 ^ hw1[0]) if hasattr(torch.bitwise_or, "reduce") else 0)
x1 = 0
x2 = 0
for v in (hw1 ^ hw1[0]).tolist():
    x1 |= v
for v in (hw2 ^ hw2[0]).tolist():
    x2 |= v
print(f"GRID {GRID} spin {SPIN_US}us warps {WARPS}: HW_ID1 varying bits {x1:#010x}, HW_ID2 varying bits {x2:#010x}")
torch.save(dict(hw1=hw1, hw2=hw2, t0=t0), f"/home/xiaompen/mega_scratch/ep4c/hwid_{GRID}_{WARPS}.pt")
for b in list(range(0, 40)) + list(range(GRID - 8, GRID)):
    print(f"  bid {b:5d} hw1 {int(hw1[b]):#010x} hw2 {int(hw2[b]):#010x} start {float(t0[b]):8.1f} us")
