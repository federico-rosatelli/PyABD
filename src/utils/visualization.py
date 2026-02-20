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

def plot_abd_results(similarity:torch.Tensor, boundaries:torch.Tensor, pred_labels_mapped=None, gt_labels=None, video_name="Video Analysis", save_path="public/img.png"):

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




def plot_online_abd_results(similarities,
    thresholds,
    boundaries,
    segment_ids,
    pred_labels_mapped=None,
    gt_labels =None,
    reject_reasons = None,
    video_name = "Online ABD",
    strategy_name = "lower_quartile",
    save_path = "public/online_img.png",
):
    """
    Four-panel plot for online ABD results.

    Panel 1 — Similarity + adaptive threshold curve
    Panel 2 — Boundary filter: NMS candidates coloured by reject reason
    Panel 3 — Segmentation timeline (GT vs prediction)
    Panel 4 — Threshold convergence
    """
    similarities = np.asarray(similarities, dtype=float)
    thresholds   = np.asarray(
        [t if t is not None else np.nan for t in thresholds], dtype=float
    )
    boundaries   = np.asarray(boundaries, dtype=int)
    segment_ids  = np.asarray(segment_ids, dtype=int)

    T_sim  = len(similarities)
    T_full = T_sim + 1

    sim_x = np.arange(T_sim)

    cmap         = plt.get_cmap("tab20")
    c_thresh     = "#e67e22"   # orange  — adaptive threshold
    c_sim        = "#2980b9"   # blue    — similarity
    c_accepted   = "#27ae60"   # green   — accepted boundary
    c_reject_thr = "#bdc3c7"   # grey    — rejected by threshold
    c_reject_dz  = "#9b59b6"   # purple  — rejected by dead zone
    c_reject_wu  = "#f39c12"   # yellow  — rejected by warm-up
    c_reject_sg  = "#e74c3c"   # red     — rejected by semantic gate

    REASON_COLORS = {
        "threshold":     c_reject_thr,
        "dead_zone":     c_reject_dz,
        "warmup":        c_reject_wu,
        "semantic_gate": c_reject_sg,
    }

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 14), sharex=False,
        gridspec_kw={"height_ratios": [3, 1.2, 1, 1.5]},
    )
    fig.suptitle(f"Online ABD — {video_name}", fontsize=15, fontweight="bold")

    ax1 = axes[0]

    if pred_labels_mapped is not None:
        labels_for_colour = np.asarray(pred_labels_mapped)
        for t in range(T_sim):
            lbl = int(labels_for_colour[min(t, len(labels_for_colour) - 1)])
            ax1.plot(
                [sim_x[t], sim_x[t] + 1],
                [similarities[t], similarities[min(t + 1, T_sim - 1)]],
                color=cmap(lbl % 20), linewidth=1.5, solid_capstyle="round",
            )
    else:
        ax1.plot(sim_x, similarities, color=c_sim, linewidth=1.5, label="Similarity")

    valid_mask = ~np.isnan(thresholds)
    if valid_mask.any():
        ax1.plot(
            sim_x[valid_mask], thresholds[valid_mask],
            color=c_thresh, linewidth=2, linestyle="--",
            label=f"Adaptive threshold ({strategy_name.replace('_', ' ')})", zorder=4,
        )
        ax1.fill_between(
            sim_x[valid_mask], similarities[valid_mask], thresholds[valid_mask],
            where=similarities[valid_mask] < thresholds[valid_mask],
            alpha=0.12, color=c_thresh, label="Below threshold (accepted zone)",
        )

    for b in boundaries:
        sim_idx = min(b, T_sim - 1)
        ax1.axvline(b, color=c_accepted, linewidth=1.0, alpha=0.5, zorder=3)
        ax1.scatter(sim_idx, similarities[sim_idx],
                    color=c_accepted, edgecolors="white", s=50, zorder=5)

    ax1.set_ylabel("Cosine similarity", fontsize=10)
    ax1.set_title("Frame-wise similarity with adaptive threshold", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, T_sim)

    ax2 = axes[1]

    # Re-detect all NMS minima from the similarity array
    L = 15
    nms_candidates: list[int] = []
    for t in range(L, T_sim):
        window     = similarities[t - L: t]
        centre_val = similarities[t - L // 2]
        if (centre_val - float(min(window))) <= 1e-6:
            nms_candidates.append(t - L // 2)

    accepted_set = set(boundaries.tolist())

    # Build a map: candidate_pos -> reject_reason (from processor output)
    pos_to_reason: dict[int, str] = {}
    if reject_reasons is not None:
        # reject_reasons is indexed like sim_history (one per sim step after t=0)
        # Each entry is the reason for the NMS candidate whose centre is at
        # that sim index. We scan NMS candidates and match by position.
        for c in nms_candidates:
            if c not in accepted_set:
                # find the closest reason entry
                sim_idx = c + L // 2   # approximate sim_history index
                if sim_idx < len(reject_reasons) and reject_reasons[sim_idx] is not None:
                    pos_to_reason[c] = reject_reasons[sim_idx]
                else:
                    pos_to_reason[c] = "threshold"  # default

    by_reason: dict[str, list[int]] = {}
    for c in nms_candidates:
        if c in accepted_set:
            continue
        reason = pos_to_reason.get(c, "threshold")
        by_reason.setdefault(reason, []).append(c)

    reason_labels = {
        "threshold":     "Rejected: threshold",
        "warmup":        "Rejected: warm-up period",
        "dead_zone":     "Rejected: dead zone (min segment length)",
        "semantic_gate": "Rejected: semantic gate (segments too similar)",
    }

    y_jitter = {"threshold": 0.0, "warmup": 0.15, "dead_zone": -0.15, "semantic_gate": 0.0}

    for reason, positions in by_reason.items():
        if not positions:
            continue
        ax2.scatter(
            positions,
            [y_jitter.get(reason, 0.0)] * len(positions),
            marker="|", s=250,
            color=REASON_COLORS.get(reason, "#95a5a6"),
            linewidths=2.0,
            label=reason_labels.get(reason, reason),
            zorder=3,
        )

    if len(boundaries) > 0:
        ax2.scatter(
            boundaries, np.zeros(len(boundaries)),
            marker="|", s=300,
            color=c_accepted, linewidths=2.5,
            label="Accepted boundary", zorder=5,
        )

    ax2.set_yticks([])
    ax2.set_title("Boundary candidates coloured by reject reason", fontsize=11)
    legend2 = ax2.legend(fontsize=8, loc="upper right", ncol=2)
    ax2.set_xlim(0, T_sim)
    ax2.grid(True, axis="x", alpha=0.25)

    ax3 = axes[2]
    bar_h = 8

    if gt_labels is not None and len(gt_labels) > 0:
        gt_segs = labels_to_segments(gt_labels)
        ax3.broken_barh(
            [(s[1], s[2]) for s in gt_segs], (bar_h + 2, bar_h),
            facecolors=[cmap(int(s[0]) % 20) for s in gt_segs],
            edgecolors="white", linewidth=0.5,
        )

    if pred_labels_mapped is not None:
        pred_segs = labels_to_segments(pred_labels_mapped)
        ax3.broken_barh(
            [(s[1], s[2]) for s in pred_segs], (0, bar_h),
            facecolors=[cmap(int(s[0]) % 20) for s in pred_segs],
            edgecolors="white", linewidth=0.5,
        )

    ax3.set_yticks([bar_h // 2, bar_h + 2 + bar_h // 2])
    ax3.set_yticklabels(["Prediction", "Ground Truth"], fontsize=9)
    ax3.set_title("Segmentation timeline", fontsize=11)
    ax3.set_xlim(0, T_full)
    ax3.grid(True, axis="x", alpha=0.25)

    ax4 = axes[3]

    if valid_mask.any():
        ax4.plot(
            sim_x[valid_mask], thresholds[valid_mask],
            color=c_thresh, linewidth=2, label="Adaptive threshold",
        )
        running_min = np.minimum.accumulate(similarities)
        running_max = np.maximum.accumulate(similarities)
        ax4.fill_between(
            sim_x, running_min, running_max,
            alpha=0.08, color=c_sim, label="Similarity range [running min, max]",
        )

    ax4.set_xlabel("Frame", fontsize=10)
    ax4.set_ylabel("Threshold value", fontsize=10)
    ax4.set_title("Adaptive threshold convergence", fontsize=11)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(0, T_sim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


