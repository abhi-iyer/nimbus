"""radar.pdf: per-task radar of Nimbus vs the |D| baseline ensemble (ViT-B/32).

a 5 rows x 3 cols grid.
  rows: accuracy, top-3, top-5, NLL, ECE
  cols: logit average, softmax average, param average
each cell overlays Nimbus against the |D|-member baseline ensemble of the
matching ensembling kind, over the N task spokes at clean input. reads
results/summary_table.txt.

usage:
    python -m experiments.vit.figures.radar_plots [--table <summary_table.txt>]
"""
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.vit import config as cfg
from experiments.vit import tables as RT
from ._common import task_display

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica", "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# two series.
COLOR_SIMPLEX = "#B39DDB"   # pastel purple, our method
COLOR_BASELINE   = "#F4A582"   # pastel coral, |D| baseline ensemble
LABEL_SIMPLEX = "Nimbus (ours)"
LABEL_BASELINE   = r"Baseline ensemble ($|\mathcal{D}|$)"
MARKER_SIMPLEX, MARKER_BASELINE = "o", "s"

M_SIMPLEX, M_BASELINE = "nimbus_face", "baseline_ens"

# (row title, metric key, rticks, (rmin, rmax), decimals).
ROWS = [
    ("Accuracy", "acc",  [0.6, 0.7, 0.8, 0.9, 1.0],          (0.6, 1.0),  1),
    ("Top-3",    "top3", [0.7, 0.8, 0.9, 1.0],               (0.7, 1.0),  1),
    ("Top-5",    "top5", [0.8, 0.9, 1.0],                    (0.8, 1.0),  1),
    ("NLL",      "nll",  [0.0, 0.5, 1.0, 1.5, 2.0],          (0.0, 2.0),  1),
    ("ECE",      "ece",  [0.0, 0.05, 0.10, 0.15, 0.20, 0.25], (0.0, 0.25), 2),
]

# (column title, ensemble mode key)
COLUMNS = [
    ("Logit average",   "logit_avg"),
    ("Softmax average", "softmax_avg"),
    ("Param average",   "weight_avg"),
]


def task_series(summary, tasks, method, mode, metric):
    row = summary[RT.summary_rowname(method, mode)]
    mi = RT.METRIC_IDX[metric]
    return np.array([row[t][mi] for t in tasks])


def fmt_tick(v, decimals):
    if np.isclose(v, 0.0):
        return "0.0"
    return f"{v:.{decimals}f}"


def style_radar(ax, labels, rmin, rmax, rticks, decimals, show_rlabels):
    N = len(labels)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(rmin, rmax)

    ax.yaxis.grid(True, color="0.80", linewidth=0.5)
    ax.xaxis.grid(True, color="0.80", linewidth=0.5)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_edgecolor("black")
    ax.spines["polar"].set_linewidth(0.9)

    ax.set_rticks(rticks)
    if show_rlabels:
        labels_r = ["" if np.isclose(r, rmin) else fmt_tick(r, decimals)
                    for r in rticks]
        ax.set_yticklabels(labels_r)
        ax.set_rlabel_position(180.0 / N)
        ax.tick_params(axis="y", length=0, pad=1, labelsize=9, colors="black")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, labelsize=0)
    ax.tick_params(axis="x", which="both", pad=16, length=0, labelsize=10)


def plot_series(ax, series):
    """series: [(values, color, marker)], later entries drawn on top."""
    N = len(series[0][0])
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta_c = np.concatenate([theta, theta[:1]])
    for i, (values, color, marker) in enumerate(series):
        v = np.concatenate([values, values[:1]])
        ax.fill(theta_c, v, color=color, alpha=0.20, zorder=2 + 2 * i, linewidth=0)
        ax.plot(theta_c, v, color=color, linewidth=1.8, marker=marker,
                markersize=5, zorder=3 + 2 * i)


def make_figure(tasks, summary, out_path):
    labels = task_display(tasks)
    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig = plt.figure(figsize=(15, 24))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.75, wspace=0.55,
                          left=0.13, right=0.82, top=0.925, bottom=0.045)
    axes_grid = [[None] * n_cols for _ in range(n_rows)]

    for r, (row_title, metric, rticks, (rmin, rmax), dec) in enumerate(ROWS):
        for c, (col_title, mode) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r, c], projection="polar")
            axes_grid[r][c] = ax
            # baseline first, Nimbus last so Nimbus sits on top.
            plot_series(ax, [
                (task_series(summary, tasks, M_BASELINE, mode, metric), COLOR_BASELINE, MARKER_BASELINE),
                (task_series(summary, tasks, M_SIMPLEX, mode, metric), COLOR_SIMPLEX, MARKER_SIMPLEX),
            ])
            style_radar(ax, labels, rmin, rmax, rticks, dec, show_rlabels=(c == 0))
            if r == 0:
                ax.text(0.5, 1.20, col_title, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=14, fontweight="bold")
        axes_grid[r][0].text(-0.42, 0.5, row_title, transform=axes_grid[r][0].transAxes,
                             ha="right", va="center", fontsize=14, fontweight="bold")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COLOR_SIMPLEX, lw=1.8, marker=MARKER_SIMPLEX,
               markersize=6, label=LABEL_SIMPLEX),
        Line2D([], [], color=COLOR_BASELINE, lw=1.8, marker=MARKER_BASELINE,
               markersize=6, label=LABEL_BASELINE),
    ]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=(0.85, 0.5), frameon=False, fontsize=13)
    fig.suptitle(
        rf"ViT-B/32 merge -- per-task radar (clean input;  N={cfg.N}, "
        rf"$\mathcal{{D}}$={tuple(cfg.DEGREES)}, r={cfg.RANK})",
        fontsize=14, fontweight="bold", y=0.985)

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None,
                    help="summary_table.txt path (default: results/summary_table.txt)")
    args = ap.parse_args()
    summary = RT.parse_summary(args.table or (cfg.RESULTS_DIR / "summary_table.txt"))
    tasks = [s for s in next(iter(summary.values())) if s != "all"]
    make_figure(tasks, summary, cfg.FIGURES_DIR / "radar.pdf")


if __name__ == "__main__":
    main()
