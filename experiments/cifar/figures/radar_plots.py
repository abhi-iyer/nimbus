"""
radar plots of per-task metrics for each trained Nimbus model (N=3,5,7,10).

layout: 5 rows x 3 cols per N.
  rows: accuracy, top-3, top-5, nll, ece
  cols: logit average, softmax average, param average
each cell overlays Nimbus against the |D|-member baseline ensemble of the
matching kind; the param-average column shows Nimbus only.

values are parsed from ../results/N={N}/summary_table.txt.

usage:
    python radar_plots.py --N 3
    python radar_plots.py --all
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._common import setup_mpl
except ImportError:
    from _common import setup_mpl
setup_mpl()

FIG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FIG_DIR.parent / 'results'

# two series, consistent across every row/column.
COLOR_SIMPLEX = '#B39DDB'  # pastel purple -- nimbus
COLOR_BASELINE   = '#F4A582'  # pastel coral -- |D|-member baseline ensemble

LABEL_SIMPLEX = 'Nimbus (ours)'
LABEL_BASELINE   = r'Baseline ensemble ($|\mathcal{D}|$)'

MARKER_SIMPLEX = 'o'   # circle
MARKER_BASELINE   = 's'   # square

NAME_SIMPLEX_LOGIT    = 'Simplex logit ens uniform degree'
NAME_SIMPLEX_SOFTMAX  = 'Simplex softmax ens uniform degree'
NAME_SIMPLEX_PARAM    = 'Simplex param avg'
NAME_BASELINE_LOGIT      = 'Baseline logit ens'
NAME_BASELINE_SOFTMAX    = 'Baseline softmax ens'

# (row title, metric index [0=acc, 1=top3, 2=top5, 3=nll, 4=ece], rticks, (rmin, rmax), decimals)
ROWS = [
    ('Accuracy', 0, [0.0, 0.25, 0.5, 0.75, 1.0],         (0.0, 1.0),  2),
    ('Top-3',    1, [0.0, 0.25, 0.5, 0.75, 1.0],         (0.0, 1.0),  2),
    ('Top-5',    2, [0.0, 0.25, 0.5, 0.75, 1.0],         (0.0, 1.0),  2),
    ('NLL',      3, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],      (0.0, 2.5),  1),
    ('ECE',      4, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25], (0.0, 0.25), 2),
]

# (column title, simplex model in summary table, baseline model or None)
COLUMNS = [
    ('Logit average',   NAME_SIMPLEX_LOGIT,   NAME_BASELINE_LOGIT),
    ('Softmax average', NAME_SIMPLEX_SOFTMAX, NAME_BASELINE_SOFTMAX),
    ('Param average',   NAME_SIMPLEX_PARAM,   None),
]


def parse_summary(path: Path) -> dict:
    """parse the summary table into {model_name: {scope: [acc, top3, top5, nll, ece]}}."""
    out: dict = {}
    current = None
    for line in path.read_text().splitlines():
        if not line.startswith('│'):
            continue
        cells = [c.strip() for c in line.strip('│').split('│')]
        if len(cells) < 7:
            continue
        model, scope = cells[0], cells[1]
        try:
            vals = [float(c) for c in cells[2:7]]
        except ValueError:
            continue
        if model:
            current = model
        if current is None:
            continue
        out.setdefault(current, {})[scope] = vals
    return out


def task_series(parsed: dict, model: str, metric_idx: int, N: int) -> np.ndarray:
    """per-task values of one metric for one model."""
    d = parsed[model]
    return np.array([d[f'task_{i}'][metric_idx] for i in range(N)])


def fmt_tick(v: float, decimals: int) -> str:
    if np.isclose(v, 0.0):
        return '0.0'
    return f'{v:.{decimals}f}'


def style_radar(ax, N: int, rmin: float, rmax: float,
                rticks, decimals: int, show_rlabels: bool):
    """style one radar subplot: spokes, radial mesh, and tick labels."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ax.set_xticks(theta)
    ax.set_xticklabels([f'Task {i}' for i in range(N)])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(rmin, rmax)

    # outer circle, angular spokes, and radial mesh.
    ax.yaxis.grid(True, color='0.80', linewidth=0.5)
    ax.xaxis.grid(True, color='0.80', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_edgecolor('black')
    ax.spines['polar'].set_linewidth(0.9)

    # rticks set on every subplot; only the leftmost column shows radial labels.
    ax.set_rticks(rticks)
    if show_rlabels:
        # label the mesh circles (skip 0.0), placed in the gap between spokes.
        labels = ['' if np.isclose(r, 0.0) else fmt_tick(r, decimals)
                  for r in rticks]
        ax.set_yticklabels(labels)
        ax.set_rlabel_position(180.0 / N)
        ax.tick_params(axis='y', length=0, pad=1, labelsize=9,
                       colors='black')
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0, labelsize=0)

    # angular (task) tick labels, pushed out from the vertices.
    ax.tick_params(axis='x', which='both', pad=18, length=0, labelsize=12)


def plot_series(ax, N: int, series):
    """
    plot each series (label, values, color, marker) in order, later on top.
    returns the line handles.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta_c = np.concatenate([theta, theta[:1]])
    handles = []
    for i, (label, values, color, marker) in enumerate(series):
        v = np.concatenate([values, values[:1]])
        z_fill = 2 + 2 * i
        z_line = 3 + 2 * i
        ax.fill(theta_c, v, color=color, alpha=0.20, zorder=z_fill,
                linewidth=0)
        line, = ax.plot(theta_c, v, color=color, linewidth=1.8, label=label,
                        marker=marker, markersize=5, zorder=z_line)
        handles.append(line)
    return handles


def make_figure(N: int, out_path: Path):
    """build and save the radar figure for one N."""
    summary_path = RESULTS_DIR / f'N={N}' / 'summary_table.txt'
    parsed = parse_summary(summary_path)

    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig = plt.figure(figsize=(15, 24))
    gs = fig.add_gridspec(n_rows, n_cols,
                          hspace=0.75, wspace=0.60,
                          left=0.13, right=0.88, top=0.955, bottom=0.045)

    axes_grid = [[None] * n_cols for _ in range(n_rows)]

    for r_idx, (row_title, metric_idx, rticks, (rmin, rmax), decimals) in enumerate(ROWS):
        for c_idx, (col_title, simplex_name, baseline_name) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r_idx, c_idx], projection='polar')
            axes_grid[r_idx][c_idx] = ax

            # draw baseline first, Simplex last, so Simplex is on top.
            series = []
            if baseline_name is not None:
                series.append((LABEL_BASELINE,
                               task_series(parsed, baseline_name, metric_idx, N),
                               COLOR_BASELINE, MARKER_BASELINE))
            series.append((LABEL_SIMPLEX,
                           task_series(parsed, simplex_name, metric_idx, N),
                           COLOR_SIMPLEX, MARKER_SIMPLEX))

            plot_series(ax, N, series)
            style_radar(ax, N, rmin, rmax, rticks, decimals,
                        show_rlabels=(c_idx == 0))

            if r_idx == 0:
                ax.text(0.5, 1.22, col_title, transform=ax.transAxes,
                        ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

        # row label left of the leftmost subplot.
        ax_left = axes_grid[r_idx][0]
        ax_left.text(-0.42, 0.5, row_title, transform=ax_left.transAxes,
                     ha='right', va='center',
                     fontsize=14, fontweight='bold')

    # figure legend on the right, outside the axes.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COLOR_SIMPLEX, linewidth=1.8, marker=MARKER_SIMPLEX,
               markersize=6, label=LABEL_SIMPLEX),
        Line2D([], [], color=COLOR_BASELINE,   linewidth=1.8, marker=MARKER_BASELINE,
               markersize=6, label=LABEL_BASELINE),
    ]
    fig.legend(handles=handles, loc='center left',
               bbox_to_anchor=(0.895, 0.5), frameon=False, fontsize=13)

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=None,
                    help='draw only this N; default draws all of {3, 5, 7, 10}.')
    args = ap.parse_args()

    Ns = [args.N] if args.N is not None else [3, 5, 7, 10]
    for N in Ns:
        make_figure(N, FIG_DIR / f'radar_N{N}.pdf')


if __name__ == '__main__':
    main()
