import time
import torch

def benchmark_latency(encoder, input_tensor, runs=50):
    encoder.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5): _ = encoder(input_tensor)
        start = time.perf_counter()
        for _ in range(runs):
            _ = encoder(input_tensor)
        elapsed = (time.perf_counter() - start) / runs
    return elapsed * 1000 # ms
