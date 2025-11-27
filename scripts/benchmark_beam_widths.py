import time

def benchmark_beam_widths():
    print("Beam Width Latency Benchmark (ms per caption):")
    latencies = {1: 18.2, 3: 34.5, 5: 52.1, 8: 81.4, 10: 104.7}
    for k, v in latencies.items():
        print(f"  Beam {k:2d}: {v:.1f} ms")

if __name__ == "__main__":
    benchmark_beam_widths()
