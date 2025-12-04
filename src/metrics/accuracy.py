import numpy as np
import torch
from src.metrics.mapping import get_hungarian_mapping


def calculate_mof(preds, targets):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    if len(preds) == 0:
        return 0.0, preds

    mapping = get_hungarian_mapping(preds, targets)
    
    mapped_preds = np.copy(preds)
    for p_id, t_id in mapping.items():
        mapped_preds[preds == p_id] = t_id
        
    correct = np.sum(mapped_preds == targets)
    total = len(targets)
    
    mof = correct / total
    return mof, mapped_preds