import numpy as np
import torch
from src.metrics.mapping import get_hungarian_mapping
from sklearn.metrics import f1_score


def calculate_mof(preds, targets):
    
    mapped_pred = np.sum(preds == targets)

    total = len(targets)
    
    mof = mapped_pred / total
    return mof


def calculate_hungerian_mapping(preds,targets):
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    if len(preds) == 0:
        return preds

    mapping = get_hungarian_mapping(preds, targets)
    
    mapped_preds = np.copy(preds)

    for p_id, t_id in mapping.items():
        mapped_preds[preds == p_id] = t_id
        
    return mapped_preds
    


def calculate_f1(preds, targets):
    
    return f1_score(targets, preds, average="weighted")


    
