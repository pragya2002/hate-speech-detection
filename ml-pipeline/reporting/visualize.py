"""Matplotlib chart rendering for the moderation report.

Pure-Python module: takes already-aggregated data (lists of dicts / dicts)
and writes PNG files. No Spark dependency. Imported by generate_report.py
inside a try/except so the pipeline degrades gracefully when matplotlib
is not installed on the cluster's driver Python.

Outputs:
  category_bar.png  -- flagged comment counts per toxicity category
  distribution.png  -- toxic vs non-toxic donut chart
  user_heatmap.png  -- top high-risk users x labels heatmap
  volume_trend.png  -- flagged volume across dataset id buckets
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

LABEL_COLORS = {
    "toxic":         "#d9534f",
    "severe_toxic":  "#a02622",
    "obscene":       "#f0ad4e",
    "threat":        "#5bc0de",
    "insult":        "#9966cc",
    "identity_hate": "#5cb85c",
}


def chart_category_bar(categories, out_path):
    labels = [c["label"] for c in categories]
    counts = [int(c["count"]) for c in categories]
    colors = [LABEL_COLORS.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Flagged comments")
    ax.set_title("Flagged comments by toxicity category")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def chart_distribution_pie(summary, out_path):
    total = int(summary["total_scanned"])
    flagged = int(summary["total_flagged"])
    clean = max(0, total - flagged)

    fig, ax = plt.subplots(figsize=(6, 6))
    if total == 0:
        ax.text(0.5, 0.5, "No data scanned", ha="center", va="center")
        ax.axis("off")
    else:
        ax.pie(
            [clean, flagged],
            labels=[f"Clean\n{clean:,}", f"Flagged\n{flagged:,}"],
            colors=["#5cb85c", "#d9534f"],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            wedgeprops={"width": 0.4, "edgecolor": "white"},
            textprops={"fontsize": 11},
        )
        ax.set_title(f"Toxic vs. clean (n = {total:,})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def chart_user_heatmap(high_risk_rows, out_path, top_n=20):
    rows = (high_risk_rows or [])[:top_n]
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5, 0.5,
            "No users crossed the high-risk threshold",
            ha="center", va="center", fontsize=12, color="#666",
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    user_ids = [str(r["user_id"]) for r in rows]
    matrix = np.array([
        [int(r.get(f"{l}_count", 0) or 0) for l in LABELS]
        for r in rows
    ])
    vmax = max(matrix.max(), 1)

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * len(rows) + 1.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(user_ids)))
    ax.set_yticklabels([f"user_{u}" for u in user_ids])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v == 0:
                continue
            ax.text(
                j, i, str(int(v)),
                ha="center", va="center",
                color="white" if v > vmax / 2 else "#333",
                fontsize=8,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Flagged count")
    ax.set_title(f"Top {len(rows)} high-risk users by category")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def chart_volume_trend(buckets, out_path):
    sorted_b = sorted(buckets or [], key=lambda b: int(b["bucket_index"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not sorted_b:
        ax.text(0.5, 0.5, "No bucketed data", ha="center", va="center")
        ax.axis("off")
    else:
        x = [int(b["bucket_index"]) for b in sorted_b]
        flagged = [int(b["flagged_count"]) for b in sorted_b]
        total = [int(b["comment_count"]) for b in sorted_b]
        ax.plot(x, total, label="Total scanned", color="#888", linestyle="--", linewidth=1.2)
        ax.plot(x, flagged, label="Flagged", color="#d9534f", linewidth=2)
        ax.fill_between(x, flagged, alpha=0.25, color="#d9534f")
        ax.set_xlabel("Comment id bucket (proxy time index)")
        ax.set_ylabel("Comment count")
        ax.set_title("Volume trend across dataset (id-bucket proxy; no real timestamps)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_all(summary, categories, high_risk_rows, buckets_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return {
        "category_bar": chart_category_bar(categories, os.path.join(out_dir, "category_bar.png")),
        "distribution": chart_distribution_pie(summary, os.path.join(out_dir, "distribution.png")),
        "user_heatmap": chart_user_heatmap(high_risk_rows, os.path.join(out_dir, "user_heatmap.png")),
        "volume_trend": chart_volume_trend(buckets_rows, os.path.join(out_dir, "volume_trend.png")),
    }
