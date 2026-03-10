import numpy as np
from scipy.optimize import linear_sum_assignment

def get_hungarian_mapping(preds, targets):

    unique_preds = np.unique(preds)
    unique_targets = np.unique(targets)
    
    pred_to_idx   = {p: i for i, p in enumerate(unique_preds)}
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}

    C_p = len(unique_preds)
    C_t = len(unique_targets)
    
    overlap_matrix = np.zeros((C_p, C_t), dtype=int)
    for p, t in zip(preds, targets):
        overlap_matrix[pred_to_idx[p], target_to_idx[t]] += 1

    row_ind, col_ind = linear_sum_assignment(overlap_matrix.max() - overlap_matrix)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        orig_pred   = unique_preds[r]
        orig_target = unique_targets[c]
        mapping[orig_pred] = orig_target

    return mapping