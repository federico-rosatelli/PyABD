import torch
import torch.nn.functional as F
import numpy as np

def extract_segment_features(features:torch.Tensor, boundaries:torch.Tensor):

    N = features.shape[0]
    
    if isinstance(boundaries, torch.Tensor):
        b_list = boundaries.tolist()
    else:
        b_list = list(boundaries)
        
    b_list = sorted(list(set(b_list)))
    
    if len(b_list) == 0 or b_list[-1] != N - 1:
        b_list.append(N - 1)
        
    seg_feats = []
    seg_sizes = []
    start_idx = 0
    
    for end_idx in b_list:
        segment = features[start_idx : end_idx + 1]
        
        if segment.shape[0] > 0:
            
            mean_feat = torch.mean(segment, dim=0)
            seg_feats.append(mean_feat)
            seg_sizes.append(segment.shape[0])
        else:
            seg_feats.append(torch.zeros(features.shape[1]))
            seg_sizes.append(0)

        start_idx = end_idx + 1
        
    if len(seg_feats) > 0:
        segment_features = torch.stack(seg_feats)
    else:
        segment_features = torch.zeros((0, features.shape[1]))

    segment_indices = [[i] for i in range(len(seg_feats))]
    
    return segment_features, segment_indices, seg_sizes

def compute_similarity_matrix(features:torch.Tensor):
    features_norm = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(features_norm, features_norm.t())
    return similarity_matrix

def refine_segments(features:torch.Tensor, boundaries:torch.Tensor, K:int):
    current_features, cluster_map, current_sizes = extract_segment_features(features, boundaries)
    
    num_current_clusters = current_features.shape[0]
    if num_current_clusters <= K:
        final_labels_per_segment = torch.arange(num_current_clusters)
        return expand_labels_to_frames(features.shape[0], boundaries, cluster_map, final_labels_per_segment)

    while num_current_clusters > K:
        sim_matrix = compute_similarity_matrix(current_features)
        sim_matrix.fill_diagonal_(-float('inf'))
        
        max_val = torch.max(sim_matrix)
        max_idx = torch.argmax(sim_matrix)
        
        row = (max_idx // num_current_clusters).item()
        col = (max_idx % num_current_clusters).item()
        
        idx1, idx2 = sorted((row, col))
        
        #new_feat = (current_features[idx1] + current_features[idx2]) / 2.0

        size1 = current_sizes[idx1]
        size2 = current_sizes[idx2]
        total_size = size1 + size2
        
        feat1 = current_features[idx1]
        feat2 = current_features[idx2]
        
        if total_size > 0:
            new_feat = (feat1 * size1 + feat2 * size2) / total_size
        else:
            new_feat = (feat1 + feat2) / 2.0

        new_feat = new_feat.unsqueeze(0)
        
        keep_indices = [i for i in range(num_current_clusters) if i != idx1 and i != idx2]
        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
        
        if len(keep_indices) > 0:
            remaining_features = current_features[keep_indices_tensor]
            current_features = torch.cat([remaining_features, new_feat], dim=0)
        else:
            current_features = new_feat

        merged_indices = cluster_map[idx1] + cluster_map[idx2]
        
        del cluster_map[idx2]
        del cluster_map[idx1]
        
        cluster_map.append(merged_indices)

        del current_sizes[idx2]
        del current_sizes[idx1]
        
        current_sizes.append(total_size)
        
        num_current_clusters -= 1
        
    final_labels = torch.arange(K)
    
    return expand_labels_to_frames(features.shape[0], boundaries, cluster_map, final_labels)

def expand_labels_to_frames(N_frames, boundaries:torch.Tensor, cluster_map:list[list[int]], final_labels:torch.Tensor):
    
    b_list = boundaries.tolist()
    b_list = sorted(list(set(b_list)))
    if len(b_list) == 0 or b_list[-1] != N_frames - 1:
        b_list.append(N_frames - 1)
        
    frame_predictions = torch.zeros(N_frames, dtype=torch.long)
    
    segment_ranges = []
    start = 0
    for end in b_list:
        segment_ranges.append((start, end))
        start = end + 1
        
    for cluster_id, segment_indices_in_cluster in enumerate(cluster_map):
        label = final_labels[cluster_id]
        
        for seg_idx in segment_indices_in_cluster:
            s_start, s_end = segment_ranges[seg_idx]
            frame_predictions[s_start : s_end + 1] = label
            
    return frame_predictions