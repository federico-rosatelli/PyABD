"""
Action Boundary Detection Module
"""
import torch
import torch.nn.functional as F
import numpy as np

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
    """
    Detect action boundaries in a video feature sequence for offline mode.
    
    Args:
        video_feature (torch.Tensor): The input video features of shape (T, D).
        kernel_size (int): The size of the smoothing kernel.
        window_size (int): The size of the local window for boundary detection.
    """

    
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


class OnlineABDProcessor:
    """
    Online Action Boundary Detection Processor
    """
    def __init__(self, kernel_size: int = 15, window_size: int = 15):
        self.kernel_size = kernel_size  # k + 1
        self.window_size = window_size  # L
        self.feature_buffer = []
        self.smoothed_features = []
        
        self.similarities = []
        self.thresholds = []
        self.reject_reasons = [] 
        self.boundaries = []
        
        self.current_t = 0
        
    def step(self, x_t: torch.Tensor):
        self.feature_buffer.append(x_t)
        if len(self.feature_buffer) > self.kernel_size:
            self.feature_buffer.pop(0)
            
        g_t = torch.stack(self.feature_buffer).mean(dim=0)
        self.smoothed_features.append(g_t)
        
        if len(self.smoothed_features) > 2:
            self.smoothed_features.pop(0)
            
        boundary_detected_at = None
        
        if len(self.smoothed_features) == 2:
            sim = F.cosine_similarity(
                self.smoothed_features[0].unsqueeze(0),
                self.smoothed_features[1].unsqueeze(0)
            ).item()
            self.similarities.append(sim)
            
            if len(self.similarities) > 1:
                thr = float(np.percentile(self.similarities[:-1], 25))
            else:
                thr = None
            self.thresholds.append(thr)
            self.reject_reasons.append(None)
            
            L = self.window_size
            if len(self.similarities) >= L:
                window = self.similarities[-L:]
                center_idx = L // 2
                center_val = window[center_idx]
                
                if center_val == min(window):
                    candidate_pos = len(self.similarities) - 1 - (L - 1 - center_idx)
                    cand_thr = self.thresholds[candidate_pos]
                    
                    if cand_thr is None:
                        self.reject_reasons[candidate_pos] = "warmup"
                    elif center_val <= cand_thr:
                        self.boundaries.append(candidate_pos)
                        boundary_detected_at = candidate_pos
                    else:
                        self.reject_reasons[candidate_pos] = "threshold"
                        
        self.current_t += 1
        return boundary_detected_at