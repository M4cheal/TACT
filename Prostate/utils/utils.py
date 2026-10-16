import os
from os.path import join, relpath
import shutil
import torch


def save_code(work_root, save_path):
    os.makedirs(join(save_path), exist_ok=True)
    
    for root, dirs, files in os.walk(work_root):
        for file in files:
            if file.endswith('.py') or file.endswith('.sh'):
                file_path = join(root, file)
                relative_path = relpath(root, work_root)
                target_dir = join(save_path, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                target_file_path = join(target_dir, file)
                shutil.copy(file_path, target_file_path)



@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x * torch.log(x + 1e-30))
    # return -(x * torch.log(x + 1e-30) + (1 - x) * torch.log(1 - x + 1e-30))