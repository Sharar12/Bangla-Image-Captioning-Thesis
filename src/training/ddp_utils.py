import os
import torch
import torch.distributed as dist

def init_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
