import numpy as np
import torch
from src.metrics.mapping import get_hungarian_mapping
from sklearn.metrics import f1_score


def calculate_mof(preds, targets):

    
    if len(preds) == 0:
        return 0, preds

    mapping = get_hungarian_mapping(preds, targets)

    mapped_preds = np.copy(preds)
    for p_id, t_id in mapping.items():
        mapped_preds[preds == p_id] = t_id

    mof = np.sum(mapped_preds == targets) / len(targets)
    return mof, mapped_preds



class GlobalMoF:

    def __init__(self):
        self.total_correct = 0
        self.total_frames  = 0

    def update(self, mapped_preds, targets):
        
        mapped_preds = mapped_preds.cpu().numpy()
        targets = targets.cpu().numpy()

        self.total_correct += int(np.sum(mapped_preds == targets))
        self.total_frames  += len(targets)

    def compute(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_correct / self.total_frames

    def reset(self):
        self.total_correct = 0
        self.total_frames  = 0


def calculate_f1(preds, targets):
    
    return f1_score(targets, preds, average="weighted")


    
