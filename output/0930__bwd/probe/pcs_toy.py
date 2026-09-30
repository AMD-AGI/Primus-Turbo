import torch
x = torch.ones(1 << 24, device="cuda")
for _ in range(20):
    x = x * 1.0001 + 0.5
torch.cuda.synchronize()
print("PCS_TOY_DONE", float(x[0]))
