import torch

# 检查 MPS 是否可用
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# 如果可用，创建一个在 MPS 设备上的张量
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    print(f"Tensor on MPS: {x}")
else:
    print("MPS not available, using CPU")