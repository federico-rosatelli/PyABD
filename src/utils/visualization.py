import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

def labels_to_segments(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    segments = []
    if len(labels) == 0:
        return segments

    current_label = labels[0]
    start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((current_label, start, i - start))
            current_label = labels[i]
            start = i
            
    segments.append((current_label, start, len(labels) - start))
    return segments

def plot_abd_results(similarity:torch.Tensor, boundaries:torch.Tensor, pred_labels_mapped=None, gt_labels=None, video_name="Video Analysis"):

    similarity = similarity.cpu().numpy()
    boundaries = boundaries.cpu().numpy()
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    plt.suptitle(f"ABD Analysis: {video_name}", fontsize=16)
    
    cmap = plt.get_cmap('tab20')
    time_steps = np.arange(len(similarity))

    segments_to_plot = []
    segments_to_plot = labels_to_segments(pred_labels_mapped)

    for label, start, duration in segments_to_plot:
        end = start + duration
        
        idx_start = start
        idx_end = min(len(time_steps), end + 1) 
        
        ax1.plot(time_steps[idx_start:idx_end], 
                 similarity[idx_start:idx_end], 
                 color=cmap(label % 20), 
                 linewidth=2)
    
    tot_len = len(similarity)
    pred_len = len(pred_labels_mapped)
    
    if  pred_len < tot_len:
        start_tail = max(0, pred_len - 1)
        
        ax1.plot(time_steps[start_tail:], 
                 similarity[start_tail:], 
                 color='black', 
                 linestyle='--', 
                 linewidth=2,
                 alpha=0.7,
                 label='Outside Preds')

    if len(boundaries) > 0:
        valid_boundaries = [b for b in boundaries if b < len(similarity)]
        ax1.scatter(valid_boundaries, similarity[valid_boundaries], 
                    color='red', edgecolor='black', zorder=5, s=10, label='Change Points')

    ax1.set_ylabel("Similarity")
    ax1.set_title("Frame-wise Similarity (Colored by Segment)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    bar_height = 10
    y_positions = []
    y_labels = []

    if gt_labels is not None and len(gt_labels) > 0:
        gt_segments = labels_to_segments(gt_labels)
        xranges = [(s[1], s[2]) for s in gt_segments]
        colors = [cmap(s[0] % 20) for s in gt_segments]
        
        ax2.broken_barh(xranges, (15, bar_height), facecolors=colors, edgecolors='white')
        y_positions.append(20)
        y_labels.append("Ground Truth")

    if len(segments_to_plot) > 0:
        xranges = [(s[1], s[2]) for s in segments_to_plot]
        colors = [cmap(s[0] % 20) for s in segments_to_plot]
        
        ax2.broken_barh(xranges, (0, bar_height), facecolors=colors, edgecolors='white')
        y_positions.append(5)
        label_text = "Prediction (Refined)" if pred_labels_mapped is not None else "Raw Segments"
        y_labels.append(label_text)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels)
    ax2.set_xlabel("Time (Frames)")
    ax2.set_title("Segmentation Timeline")

    plt.tight_layout()
    # plt.show()
    plt.savefig("public/img.png")