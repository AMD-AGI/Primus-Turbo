import sys, time, torch
n = int(sys.argv[1])
t = torch.empty(n, dtype=torch.bfloat16, device="cuda")
for _ in range(5):
    t.zero_()
torch.cuda.synchronize()
b = 1e9
for _ in range(40):
    s = time.perf_counter(); t.zero_(); torch.cuda.synchronize()
    b = min(b, (time.perf_counter() - s) * 1e3)
print("  zero_ %d elems: %.4f ms = %.2f TB/s" % (n, b, n * 2 / b * 1e-9))
