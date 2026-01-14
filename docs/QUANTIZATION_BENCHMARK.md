# Quantization Performance Benchmark

| Model Variant | Precision | Model Size | Latency (CPU) | BLEU-4 |
| :--- | :---: | :---: | :---: | :---: |
| Full Precision | FP32 | 480 MB | 142 ms | 31.6 |
| Half Precision | FP16 | 240 MB | 86 ms | 31.5 |
| Dynamic Quantized | **INT8** | **125 MB** | **48 ms** | **31.2** |

INT8 quantization yields a 74% storage reduction with negligible degradation (-0.4 BLEU-4).
