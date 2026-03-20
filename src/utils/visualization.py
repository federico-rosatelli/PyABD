import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


BG        = "#0f1117"
PANEL     = "#161b27"
GRID      = "#1e2535"
FG        = "#e2e8f0"
FG_DIM    = "#64748b"

C_SIM     = "#38bdf8"
C_THR     = "#f97316"
C_ACCEPT  = "#4ade80"

REASON_PALETTE = {
    "threshold":     ("#6366f1", "▲", "threshold"),
    "dead_zone":     ("#f43f5e", "■", "dead zone"),
    "semantic_gate": ("#a78bfa", "◆", "semantic gate")
}

SEG_CMAP = plt.get_cmap("tab20")



def labels_to_segments(labels):
    if hasattr(labels, "cpu"):
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


def style_ax(ax, title="", xlabel="", ylabel="", pad=5):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, color=FG, fontsize=8, fontweight="bold", pad=pad)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG_DIM, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG_DIM, fontsize=7)


def plot_online_abd_results(
    similarities,
    thresholds,
    boundaries,
    segment_ids,
    pred_labels_mapped=None,
    gt_labels=None,
    reject_reasons=None,
    video_name="Online ABD",
    strategy_name="lower_quartile",
    save_path="public/tests/online/"
):
    similarities = np.asarray(similarities, dtype=float)
    thresholds   = np.asarray(
        [t if t is not None else np.nan for t in thresholds], dtype=float
    )
    boundaries   = np.asarray(boundaries, dtype=int)
    T_sim        = len(similarities)
    T_full       = T_sim + 1
    sim_x        = np.arange(T_sim)

    fig = plt.figure(figsize=(11, 6), facecolor=BG)
    fig.suptitle(
        f"Online ABD  ·  {video_name}",
        color=FG, fontsize=10, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[3, 2, 1],
        hspace=0.45,
        left=0.07, right=0.97,
        top=0.91, bottom=0.09
    )
    ax_sim    = fig.add_subplot(gs[0])
    ax_reject = fig.add_subplot(gs[1], sharex=ax_sim)   # reject reasons
    ax_seg    = fig.add_subplot(gs[2], sharex=ax_sim)

    style_ax(ax_sim, title="Frame-wise similarity", ylabel="cosine sim")

    if pred_labels_mapped is not None:
        mapped = np.asarray(pred_labels_mapped)
        for t in range(T_sim - 1):
            lbl = int(mapped[min(t, len(mapped) - 1)])
            ax_sim.plot(
                [t, t + 1],
                [similarities[t], similarities[t + 1]],
                color=SEG_CMAP(lbl % 20), linewidth=1.2,
                solid_capstyle="round", alpha=0.9
            )
    else:
        ax_sim.plot(sim_x, similarities, color=C_SIM, linewidth=1.2)

    valid = ~np.isnan(thresholds)
    if valid.any():
        ax_sim.plot(
            sim_x[valid], thresholds[valid],
            color=C_THR, linewidth=1.4, linestyle="--",
            label=f"lower quartile", zorder=4
        )
        ax_sim.fill_between(
            sim_x[valid], similarities[valid], thresholds[valid],
            where=similarities[valid] < thresholds[valid],
            color=C_THR, alpha=0.08
        )

    for b in boundaries:
        si = min(b, T_sim - 1)
        ax_sim.axvline(b, color=C_ACCEPT, linewidth=0.8, alpha=0.4, zorder=3)
        ax_sim.scatter(si, similarities[si],
                       color=C_ACCEPT, edgecolors=BG, s=30, zorder=6, linewidths=0.8)

    ax_sim.set_xlim(0, T_sim)
    ax_sim.legend(fontsize=7, framealpha=0, labelcolor=FG_DIM, loc="lower right")

    style_ax(ax_reject, title="Boundary filter — rejected candidates", pad=15)
    ax_reject.set_facecolor(BG)

    REASONS = list(REASON_PALETTE.keys())
    lane_y  = {r: i for i, r in enumerate(REASONS)}   # 0..3, bottom to top
    y_ticks = []
    y_labels = []

    for r, y in lane_y.items():
        color, marker, label = REASON_PALETTE[r]
        ax_reject.axhspan(y - 0.4, y + 0.4, color=color, alpha=0.04, zorder=0)
        ax_reject.axhline(y, color=color, linewidth=0.4, alpha=0.18, zorder=1)
        y_ticks.append(y)
        y_labels.append(label)

    L = max(len(similarities) // 60, 5)
    accepted_set = set(boundaries.tolist())

    pos_to_reason = {}
    if reject_reasons is not None:
        for pos, reason in enumerate(reject_reasons):
            if reason is not None and pos not in accepted_set:
                pos_to_reason[pos] = reason

    reason_counts = {r: 0 for r in REASONS}
    for pos, reason in pos_to_reason.items():
        if reason not in REASON_PALETTE:
            continue
        color, marker, _ = REASON_PALETTE[reason]
        y = lane_y[reason]
        reason_counts[reason] += 1
        ax_reject.vlines(pos, y - 0.38, y + 0.38,
                         color=color, linewidth=3, alpha=0.18, zorder=2)
        ax_reject.vlines(pos, y - 0.32, y + 0.32,
                         color=color, linewidth=1.0, alpha=0.85, zorder=3)

    if len(boundaries) > 0:
        accept_y = len(REASONS)
        ax_reject.axhspan(accept_y - 0.4, accept_y + 0.4,
                          color=C_ACCEPT, alpha=0.04, zorder=0)
        ax_reject.axhline(accept_y, color=C_ACCEPT, linewidth=0.4, alpha=0.18)
        for b in boundaries:
            ax_reject.vlines(b, accept_y - 0.38, accept_y + 0.38,
                             color=C_ACCEPT, linewidth=3, alpha=0.22, zorder=2)
            ax_reject.vlines(b, accept_y - 0.32, accept_y + 0.32,
                             color=C_ACCEPT, linewidth=1.1, alpha=0.9, zorder=3)
        y_ticks.append(accept_y)
        y_labels.append(f"accepted")

    ax_reject.set_yticks(y_ticks)
    ax_reject.set_yticklabels(y_labels, fontsize=7)
    for tick, label in zip(ax_reject.get_yticklabels(), y_labels):
        key = next((k for k, v in REASON_PALETTE.items() if v[2] == label), None)
        color = REASON_PALETTE[key][0] if key else C_ACCEPT
        tick.set_color(color)

    ax_reject.set_ylim(-0.7, len(REASONS) + 0.7)
    ax_reject.tick_params(axis="x", colors=FG_DIM, labelsize=7)

    legend_parts = []
    for r in REASONS:
        color, _, label = REASON_PALETTE[r]
        n = reason_counts[r]
        patch = mpatches.Patch(color=color, alpha=0.7,
                               label=f"{label}  {n}")
        legend_parts.append(patch)
    accept_patch = mpatches.Patch(color=C_ACCEPT, alpha=0.7,
                                  label=f"accepted  {len(boundaries)}")
    legend_parts.append(accept_patch)
    ax_reject.legend(
        handles=legend_parts, fontsize=6.5, framealpha=0,
        labelcolor=FG_DIM, loc="upper right",
        handlelength=1, handletextpad=0.5, ncol=len(legend_parts), bbox_to_anchor=(0.43, 1.15)
    )

    style_ax(ax_seg, title="Segmentation", xlabel="frame")
    bar_h = 1.0
    gap   = 0.15

    if gt_labels is not None and len(gt_labels) > 0:
        for lbl, start, dur in labels_to_segments(gt_labels):
            ax_seg.barh(
                1 + gap / 2, dur, left=start, height=bar_h,
                color=SEG_CMAP(int(lbl) % 20), edgecolor=BG, linewidth=0.3,
            )

    if pred_labels_mapped is not None:
        for lbl, start, dur in labels_to_segments(pred_labels_mapped):
            ax_seg.barh(
                0 - gap / 2, dur, left=start, height=bar_h,
                color=SEG_CMAP(int(lbl) % 20), edgecolor=BG, linewidth=0.3,
            )

    ax_seg.set_yticks([0, 1])
    ax_seg.set_yticklabels(["pred", "GT"], fontsize=7, color=FG_DIM)
    ax_seg.set_ylim(-0.7, 1.7)
    ax_seg.set_xlim(0, T_full)
    ax_seg.grid(False)
    
    save_path = os.path.join(save_path, f"{video_name}.png")

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_abd_results(
    similarity, boundaries, pred_labels_mapped=None,
    gt_labels=None, video_name="Video Analysis", save_path="public/tests/offline/"):
    
    if hasattr(similarity, "cpu"):
        similarity = similarity.cpu().numpy()
    else:
        similarity = np.asarray(similarity)
    if hasattr(boundaries, "cpu"):
        boundaries = boundaries.cpu().numpy()
    else:
        boundaries = np.asarray(boundaries, dtype=int)
 
    T        = len(similarity)
    sim_x    = np.arange(T)
 
    fig = plt.figure(figsize=(11, 5), facecolor=BG)
    fig.suptitle(
        f"Offline ABD  ·  {video_name}",
        color=FG, fontsize=10, fontweight="bold", y=0.98,
    )
 
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 1],
        hspace=0.55,
        left=0.07, right=0.97,
        top=0.91, bottom=0.09
    )
    ax_sim = fig.add_subplot(gs[0])
    ax_seg = fig.add_subplot(gs[1], sharex=ax_sim)
 
    style_ax(ax_sim, title="Frame-wise similarity", ylabel="cosine sim")
 
    segments_plot = labels_to_segments(pred_labels_mapped) if pred_labels_mapped is not None else []
 
    for label, start, duration in segments_plot:
        end = start + duration
        s = start
        e = min(T, end + 1)
        ax_sim.plot(sim_x[s:e], similarity[s:e],
                    color=SEG_CMAP(label % 20), linewidth=1.2,
                    solid_capstyle="round", alpha=0.9)
 
    if pred_labels_mapped is not None and len(pred_labels_mapped) < T:
        tail = max(0, len(pred_labels_mapped) - 1)
        ax_sim.plot(sim_x[tail:], similarity[tail:],
                    color=FG_DIM, linestyle="--", linewidth=1.0, alpha=0.5)
 
    valid_b = [int(b) for b in boundaries if b < T]
    for b in valid_b:
        ax_sim.axvline(b, color=C_ACCEPT, linewidth=0.8, alpha=0.4, zorder=3)
        ax_sim.scatter(b, similarity[b],
                       color=C_ACCEPT, edgecolors=BG, s=30, zorder=6, linewidths=0.8)
 
    ax_sim.set_xlim(0, T)
 
    style_ax(ax_seg, title="Segmentation", xlabel="frame")
    bar_h = 1.0
    gap   = 0.15
 
    if gt_labels is not None and len(gt_labels) > 0:
        for lbl, start, dur in labels_to_segments(gt_labels):
            ax_seg.barh(
                1 + gap / 2, dur, left=start, height=bar_h,
                color=SEG_CMAP(int(lbl) % 20), edgecolor=BG, linewidth=0.3,
            )
 
    for lbl, start, dur in segments_plot:
        ax_seg.barh(
            0 - gap / 2, dur, left=start, height=bar_h,
            color=SEG_CMAP(int(lbl) % 20), edgecolor=BG, linewidth=0.3,
        )
 
    ax_seg.set_yticks([0, 1])
    ax_seg.set_yticklabels(["pred", "GT"], fontsize=7, color=FG_DIM)
    ax_seg.set_ylim(-0.7, 1.7)
    ax_seg.set_xlim(0, T)
    ax_seg.grid(False)

    save_path = os.path.join(save_path, f"{video_name}.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
 