import torch, time
torch.cuda.init(); d='cuda'
N = 1<<30  # 1Gi elements bf16 = 2 GiB
x = torch.empty(N, dtype=torch.bfloat16, device=d).normal_()
y = torch.empty_like(x)
def timeit(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    ts=[]
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]
t = timeit(lambda: y.copy_(x))
bytes_moved = 2*x.numel()*2  # read+write
print("copy  %.3f ms  %.1f GB/s (r+w)" % (t, bytes_moved/t*1e-6))
t2 = timeit(lambda: torch.sum(x, dtype=torch.float32))
print("read  %.3f ms  %.1f GB/s (r)" % (t2, x.numel()*2/t2*1e-6))
