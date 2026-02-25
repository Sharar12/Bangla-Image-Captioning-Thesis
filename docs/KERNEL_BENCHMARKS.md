# Fused Kernel Latency Analysis

- Standard PyTorch Attention: 52 ms / image
- FlashAttention-2 / SDPA Fused Kernel: **31 ms / image** (40% speedup)
- Memory allocation: Peak reduced from 2.4 GB to 1.6 GB VRAM.
