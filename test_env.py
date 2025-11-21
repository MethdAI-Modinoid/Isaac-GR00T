# test_env.py
import os, sys
import torch

print("="*40)
print("Python:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(3, 3, device=device)
    print("Tensor on GPU:", x)
    print("Sum:", x.sum().item())

print("="*40)
print("✅ Environment check completed")
