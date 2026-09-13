"""
radar plots of per-task metrics for the LLM leg (N=7).

layout: 5 rows x 3 cols.
  rows: accuracy, top-3, top-5, NLL, ECE
  cols: logit average, softmax average, param average
each cell overlays Nimbus against the |D|-member baseline ensemble of the
matching ensembling kind. the param-avg column shows Nimbus only.

values are parsed from ../results/summary_table.txt.

usage:
    python radar_plots.py
"""
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

N = 7
TASK_LABELS = [
    'Early Modern English',
    'WikiText',
    'TinyStories',
    'Code',
    'arXiv',
    'Legal',
    'French/German',
]

# two series, consistent across every row/column.
COLOR_SIMPLEX = '#B39DDB'  # pastel purple, our method
COLOR_BASELINE   = '#F4A582'  # pastel coral, |D|-member baseline ensemble

LABEL_SIMPLEX = 'Nimbus (ours)'
LABEL_BASELINE   = r'Baseline ensemble ($|\mathcal{D}|$)'

MARKER_SIMPLEX = 'o'   # circle
MARKER_BASELINE   = 's'   # square

# row names in summary_table.txt (post-calibration Nimbus).
NAME_SIMPLEX_LOGIT    = 'Simplex logit ens uniform degree'
NAME_SIMPLEX_SOFTMAX  = 'Simplex softmax ens uniform degree'
NAME_SIMPLEX_PARAM    = 'Simplex param avg'
# the |D|-member compute+param-matched baseline ensemble.
NAME_BASELINE_LOGIT      = 'Baseline logit ens'
NAME_BASELINE_SOFTMAX    = 'Baseline softmax ens'

# (row title, metric index [0=acc, 1=top3, 2=top5, 3=nll, 4=ece], rticks, (rmin, rmax), decimals)
# LLM-tuned radial ranges.
ROWS = [
    ('Accuracy', 0, [0.0, 0.2, 0.4, 0.6, 0.8],         (0.0, 0.8),  1),
    ('Top-3',    1, [0.0, 0.2, 0.4, 0.6, 0.8],         (0.0, 0.8),  1),
    ('Top-5',    2, [0.0, 0.2, 0.4, 0.6, 0.8],         (0.0, 0.8),  1),
    ('NLL',      3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],    (0.0, 5.0),  0),
    ('ECE',      4, [0.0, 0.02, 0.04, 0.06, 0.08],     (0.0, 0.08), 2),
]

# (column title, simplex model in summary table, baseline model or None)
COLUMNS = [
    ('Logit average',   NAME_SIMPLEX_LOGIT,   NAME_BASELINE_LOGIT),
    ('Softmax average', NAME_SIMPLEX_SOFTMAX, NAME_BASELINE_SOFTMAX),
    ('Param average',   NAME_SIMPLEX_PARAM,   None),
]


def parse_summary(path: Path) -> dict:
    """return {model_name: {scope: [acc, top3, top5, nll, ece]}}."""
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


def task_series(parsed: dict, model: str, metric_idx: int) -> np.ndarray:
    d = parsed[model]
    return np.array([d[f'task_{i}'][metric_idx] for i in range(N)])


def fmt_tick(v: float, decimals: int) -> str:
    if np.isclose(v, 0.0):
        return '0.0'
    return f'{v:.{decimals}f}'


def style_radar(ax, rmin: float, rmax: float,
                rticks, decimals: int, show_rlabels: bool):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ax.set_xticks(theta)
    ax.set_xticklabels(TASK_LABELS)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(rmin, rmax)

    # keep outer circle, angular spokes, and the radial mesh.
    ax.yaxis.grid(True, color='0.80', linewidth=0.5)
    ax.xaxis.grid(True, color='0.80', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_edgecolor('black')
    ax.spines['polar'].set_linewidth(0.9)

    ax.set_rticks(rticks)
    if show_rlabels:
        # label the mesh circles directly (skip 0.0); place labels in the
        # angular gap between two adjacent task spokes.
        labels = ['' if np.isclose(r, 0.0) else fmt_tick(r, decimals)
                  for r in rticks]
        ax.set_yticklabels(labels)
        ax.set_rlabel_position(180.0 / N)
        ax.tick_params(axis='y', length=0, pad=1, labelsize=9,
                       colors='black')
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0, labelsize=0)

    # angular (task) tick labels, pushed out from the outer circle.
    ax.tick_params(axis='x', which='both', pad=22, length=0, labelsize=11)


def plot_series(ax, series):
    """draw each series (label, values, color, marker) in order; later on top."""
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


def make_figure(out_path: Path):
    summary_path = RESULTS_DIR / 'summary_table.txt'
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

            # baseline first, Simplex last so Simplex sits on top.
            series = []
            if baseline_name is not None:
                series.append((LABEL_BASELINE,
                               task_series(parsed, baseline_name, metric_idx),
                               COLOR_BASELINE, MARKER_BASELINE))
            series.append((LABEL_SIMPLEX,
                           task_series(parsed, simplex_name, metric_idx),
                           COLOR_SIMPLEX, MARKER_SIMPLEX))

            plot_series(ax, series)
            style_radar(ax, rmin, rmax, rticks, decimals,
                        show_rlabels=(c_idx == 0))

            if r_idx == 0:
                ax.text(0.5, 1.22, col_title, transform=ax.transAxes,
                        ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

        ax_left = axes_grid[r_idx][0]
        ax_left.text(-0.42, 0.5, row_title, transform=ax_left.transAxes,
                     ha='right', va='center',
                     fontsize=14, fontweight='bold')

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
    make_figure(FIG_DIR / 'radar.pdf')


if __name__ == '__main__':
    main()
