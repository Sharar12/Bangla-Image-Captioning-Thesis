import torch

def print_gpu_memory_stats():
    if not torch.cuda.is_available():
        print("[Device] Running on CPU")
        return
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"[GPU] {props.name} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
