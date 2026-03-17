import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_topk(outputs, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k * 100.0 / targets.size(0)).item())
        return results

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)