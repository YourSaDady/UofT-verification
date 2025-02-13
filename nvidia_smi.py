import os
import torch

# Get GPU IDs allocated to the job
gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')

# Get memory info for each GPU
for gpu_id in gpu_ids:
    device = torch.device(f'cuda:{gpu_id}')
    mem_allocated = torch.cuda.memory_allocated(device)
    mem_reserved = torch.cuda.memory_reserved(device)
    
    print(f"GPU {gpu_id}: Allocated Memory = {mem_allocated}, Reserved Memory = {mem_reserved}")