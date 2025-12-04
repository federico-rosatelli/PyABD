import numpy as np
from scipy.optimize import linear_sum_assignment

def get_hungarian_mapping(preds, targets):

    unique_preds = np.unique(preds)
    unique_targets = np.unique(targets)
    
    max_p = unique_preds.max() + 1 if len(unique_preds) > 0 else 0
    max_t = unique_targets.max() + 1 if len(unique_targets) > 0 else 0
    
    overlap_matrix = np.zeros((max_p, max_t), dtype=int)
    
    for p, t in zip(preds, targets):
        overlap_matrix[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(overlap_matrix.max() - overlap_matrix)
    
    mapping = {}
    for p_idx, t_idx in zip(row_ind, col_ind):
        mapping[p_idx] = t_idx
        
    return mapping