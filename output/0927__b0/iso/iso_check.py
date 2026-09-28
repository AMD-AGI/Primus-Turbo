import os, torch, time
n = torch.cuda.device_count()
p = torch.cuda.get_device_properties(0)
print(f"phys={os.environ['OE_PHYS_GPU']} count={n} pci_domain={getattr(p,'pci_domain_id','?')} bus={getattr(p,'pci_bus_id','?')} name={p.gcnArchName}")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = time.time()
while time.time() - t < 6:
    a @ a
torch.cuda.synchronize()
