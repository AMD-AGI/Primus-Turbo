import torch
a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
for _ in range(3): b=a@a
torch.cuda.synchronize()
print("ok")
