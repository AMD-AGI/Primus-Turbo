import torch
a = torch.randn(1024, 1024, device="cuda"); b = (a * 2).sum(); torch.cuda.synchronize(); print("TINY", b.item())
