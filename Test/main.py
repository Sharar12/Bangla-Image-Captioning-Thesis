import torch
import time

# 1. Setup
device = torch.device("cuda")
size = 5000  # Large matrix size (5000x5000)

print(f"🚀 Starting Stress Test on: {torch.cuda.get_device_name(0)}")
print(f"Allocating {size}x{size} Float32 Tensors (~200MB each)...")

# 2. Allocate Memory
start_time = time.time()
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# 3. Heavy Computation (Matrix Multiplication)
print("Computing A @ B ...")
c = torch.matmul(a, b)

# 4. Synchronize (Wait for GPU to finish)
torch.cuda.synchronize()
end_time = time.time()

print(f"✅ COMPLETED in {end_time - start_time:.4f} seconds.")
print("Your GPU drivers are stable under load.")