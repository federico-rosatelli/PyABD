"""
Action Boundary Detection Module
"""
import torch
import torch.nn.functional as F

def smooth_features(features:torch.Tensor, kernel_size:int=15):

    features_t = features.t().unsqueeze(0)
    
    pad_size = kernel_size // 2
    features_smoothed = F.avg_pool1d(
        features_t, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=pad_size, 
        count_include_pad=False
    )
    
    return features_smoothed.squeeze(0).t()

def calculate_similarity(features:torch.Tensor):
    features_norm = F.normalize(features, p=2, dim=1)
    
    similarity = torch.sum(features_norm[:-1] * features_norm[1:], dim=1)
    
    return similarity

def detect_boundaries(video_feature:torch.Tensor, kernel_size:int=15, window_size:int=15):

    
    features = smooth_features(video_feature, kernel_size)
    similarity = calculate_similarity(features)

    T = similarity.shape[0]
    boundaries = []
    
    for t in range(T):
        start = max(0, t - window_size // 2)
        end = min(T, t + window_size // 2 + 1)
        
        local_window = similarity[start:end]
        min_val = torch.min(local_window)
        
        if similarity[t] == min_val:
            boundaries.append(t)
            
    return torch.tensor(boundaries), similarity