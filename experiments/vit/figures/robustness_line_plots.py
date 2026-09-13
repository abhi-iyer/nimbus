"""robustness_{input,weight}.pdf: all-scope metrics vs perturbation strength.

one tall PDF per noise-type with two stacked sections separated by a vertical gap:
  top section    -- main text: Nimbus vs the compute/parameter-matched baseline
                    ensemble (|D|) and the larger |D|+N ensemble.
  bottom section -- appendix: every method (adds linear interpolation and the
                    Fisher / TIES / task-arithmetic parameter merges).

both sections share a 5 rows x 3 cols layout:
  rows: accuracy, top-3, top-5, NLL, ECE
  cols: logit average, softmax average, param average
the ensembles (Nimbus, baseline, linear interpolation) are drawn with each
column's aggregation; the single-model parameter merges are identical across all
three aggregations, so they are drawn once per column from the 'single' curve.
one mean line per method (no shaded bands).

color encodes the family of method:
  purple     = Nimbus
  coral      = |D|-member baseline ensemble
  teal       = larger |D|+N ensemble
  steel blue = linear interpolation (the same manifold with the bumps gated off)
  grayscale  = parameter merges of the vertices (Fisher / TIES / task arithmetic),
               separated by linestyle

data: results/robustness_table.txt.

usage:
    python -m experiments.vit.figures.robustness_line_plots
    python -m experiments.vit.figures.robustness_line_plots --table <robustness_table.txt>
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.vit import config as cfg
from experiments.vit import tables as RT

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Helvetica', 'Nimbus Sans', 'Arial', 'Liberation Sans', 'DejaVu Sans'
]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# (mode key, column title): the figure's three columns.
MODES = [("logit_avg", "Logit average"), ("softmax_avg", "Softmax average"),
         ("weight_avg", "Param average")]

# --- colors ---
COLOR_SIMPLEX  = '#B39DDB'  # pastel purple, our method
COLOR_BASELINE = '#F4A582'  # pastel coral, |D|-member baseline ensemble
COLOR_BASELINE_DPN = '#3CAEA3'  # teal, larger |D|+N ensemble
COLOR_LINEAR   = '#4C72B0'  # steel blue, linear interpolation (bumps off)
# parameter merges of the vertices: grayscale, distinct linestyles.
COLOR_FISHER    = '#1a1a1a'  # ~10% gray
COLOR_TIES      = '#4d4d4d'  # ~30% gray
COLOR_TASKARITH = '#7a7a7a'  # ~48% gray

# --- labels ---
LABEL_SIMPLEX   = 'Nimbus (ours)'
LABEL_BASELINE  = r'Baseline ensemble ($|\mathcal{D}|$)'
LABEL_BASELINE_DPN = r'Baseline ensemble ($|\mathcal{D}|{+}N$)'
LABEL_LINEAR    = 'Linear interpolation'
LABEL_FISHER    = 'Fisher Merge'
LABEL_TIES      = 'TIES Merge'
LABEL_TASKARITH = 'Task Arithmetic'

# --- method bases as they key res["robust"] (before the "_{kind}" suffix) ---
M_SIMPLEX  = 'nimbus'
M_BASELINE = 'baseline_ens'
M_BDPN     = 'baseline_ens_dpn'
M_LINEAR   = 'linear_interp'
M_FISHER   = 'fisher'
M_TIES     = 'ties'
M_TASK     = 'task_arithmetic'

# --- line styles / weights / z-orders (background -> foreground) ---
LS_SOLID, LS_DASHED, LS_DASHDOT = '-', '--', '-.'
LW_SIMPLEX  = 2.0    # Nimbus, alone at the top
LW_ENS      = 1.6    # baseline ensemble + linear interpolation
LW_BASELINE = 1.0    # parameter merges
Z_BASELINE  = 2      # single-model parameter merges
Z_ENS       = 4      # ensembles (baseline + linear)
Z_SIMPLEX   = 5      # Nimbus on top

# main-text (top) section shows Nimbus, the |D| ensemble, and the |D|+N ensemble.
# linear interpolation and the grayscale merges go to the appendix section only.
EXCLUDE_FROM_MAIN = frozenset({M_LINEAR, M_FISHER, M_TIES, M_TASK})

# universal baselines: single models, identical across aggregations, drawn on
# every column from the 'single' curve. (method, label, color, ls, lw, zorder)
UNIVERSAL_BASELINES = [
    (M_FISHER, LABEL_FISHER,    COLOR_FISHER,    LS_SOLID,   LW_BASELINE, Z_BASELINE),
    (M_TIES,   LABEL_TIES,      COLOR_TIES,      LS_DASHED,  LW_BASELINE, Z_BASELINE),
    (M_TASK,   LABEL_TASKARITH, COLOR_TASKARITH, LS_DASHDOT, LW_BASELINE, Z_BASELINE),
]
# column-specific ensembles (drawn with the column's aggregation); Nimbus is
# drawn separately, last, so it stays on top.
COL_ENSEMBLES = [
    (M_LINEAR,   LABEL_LINEAR,   COLOR_LINEAR,   LS_SOLID, LW_ENS, Z_ENS),
    (M_BASELINE, LABEL_BASELINE, COLOR_BASELINE, LS_SOLID, LW_ENS, Z_ENS),
    (M_BDPN, LABEL_BASELINE_DPN, COLOR_BASELINE_DPN, LS_SOLID, LW_ENS, Z_ENS),
]

# (row title, metric key, yticks, (ymin, ymax), decimals). subplots are re-fit
# tight via fit_yaxis; ymax is the NLL/ECE clip ceiling, yticks/decimals are
# unused fallbacks.
ROWS = [
    ('Accuracy', 'acc',  [0.0, 0.25, 0.5, 0.75, 1.0], (0.0, 1.0), 2),
    ('Top-3',    'top3', [0.0, 0.25, 0.5, 0.75, 1.0], (0.0, 1.0), 2),
    ('Top-5',    'top5', [0.0, 0.25, 0.5, 0.75, 1.0], (0.0, 1.0), 2),
    ('NLL',      'nll',  [0.0, 1.5, 3.0, 4.5, 6.0],   (0.0, 8.0), 1),
    ('ECE',      'ece',  [0.0, 0.1, 0.2, 0.3, 0.4],   (0.0, 1.0), 1),
]

XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
XLIM   = (0.0, 1.0)


def fmt_tick(v: float, decimals: int) -> str:
    if np.isclose(v, 0.0):
        return '0.0'
    return f'{v:.{decimals}f}'


def _nice_step(raw):
    """smallest 1/2/2.5/5 x 10^k value >= raw, for evenly spaced tick steps."""
    if raw <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5):
        if m * mag >= raw:
            return float(m * mag)
    return float(10 * mag)


def _decimals_for(step):
    """fewest decimals that render step (hence every tick) exactly."""
    for d in range(5):
        if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-6:
            return d
    return 4


def fit_yaxis(ax, anchor0, cap):
    """
    tighten one subplot's y-axis to just above its top-most line. lower-is-better
    rows (anchor0=True) pin the floor at 0 and clip the top at cap; bounded
    higher-is-better rows fit both ends inside [0, 1].
    """
    ys = [np.asarray(l.get_ydata(), dtype=float) for l in ax.get_lines()]
    vals = np.concatenate([y for y in ys if y.size]) if ys else np.array([0.0, 1.0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0, 1.0])
    lo, hi = float(vals.min()), float(vals.max())
    if anchor0:
        bot = 0.0
        top = hi * 1.06 if cap is None else min(hi * 1.06, cap)
    else:  # bounded higher-is-better
        span = (hi - lo) or max(abs(hi), 1e-3)
        bot = max(0.0, lo - span * 0.06)
        top = min(1.0, hi + span * 0.06)
    if top <= bot:
        top = bot + 1e-3
    step = _nice_step((top - bot) / 4.0)
    dec = _decimals_for(step)
    ticks, t = [], np.ceil(bot / step - 1e-9) * step
    while t <= top + 1e-9:
        ticks.append(round(float(t), 6) or 0.0)  # `or 0.0` kills -0.0
        t += step
    if len(ticks) < 2:
        ticks = [round(bot, 6), round(bot + step, 6)]
    ax.set_ylim(bot, top)
    ax.set_yticks(ticks)
    # snap any tick that rounds to zero to +0.0 so it never renders as "-0.0".
    ax.set_yticklabels(
        [f'{(0.0 if abs(v) < 0.5 * 10 ** -dec else v):.{dec}f}' for v in ticks])


def style_axes(ax):
    """x-axis, ticks, and spines. the y-axis is set separately by fit_yaxis."""
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([fmt_tick(t, 1) for t in XTICKS])
    ax.tick_params(axis='both', which='both', direction='out',
                   length=4, width=0.8, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.9)


def _draw(ax, rob, method, kind, mode, metric, color, lw, z, ls='-'):
    """draw one method's mean curve; no-op if absent for this noise."""
    d = rob.get((RT.robust_cname(method, mode), kind))
    if not d:
        return
    eps = sorted(d)
    mi = RT.METRIC_IDX[metric]
    ax.plot(np.array(eps), np.array([d[e][mi] for e in eps]),
            color=color, linewidth=lw, linestyle=ls, zorder=z)


def plot_cell(ax, rob, kind, mode, metric, exclude):
    """all (non-excluded) methods, background -> foreground so Nimbus is on top."""
    # parameter merges: identical across aggregations, so read the 'single' curve.
    for name, _lbl, color, ls, lw, z in UNIVERSAL_BASELINES:
        if name in exclude:
            continue
        _draw(ax, rob, name, kind, 'single', metric, color, lw, z, ls)
    # column-specific ensembles: this column's aggregation.
    for name, _lbl, color, ls, lw, z in COL_ENSEMBLES:
        if name in exclude:
            continue
        _draw(ax, rob, name, kind, mode, metric, color, lw, z, ls)
    # foreground: Nimbus (never excluded).
    _draw(ax, rob, M_SIMPLEX, kind, mode, metric,
          COLOR_SIMPLEX, LW_SIMPLEX, Z_SIMPLEX, LS_SOLID)


# full legend definition. (label, color, lw, ls, method). an entry is kept iff
# its method is not in the exclude set.
def _legend_entries():
    return [
        (LABEL_SIMPLEX,   COLOR_SIMPLEX,   LW_SIMPLEX,  LS_SOLID,   M_SIMPLEX),
        (LABEL_BASELINE,  COLOR_BASELINE,  LW_ENS,      LS_SOLID,   M_BASELINE),
        (LABEL_BASELINE_DPN, COLOR_BASELINE_DPN, LW_ENS,   LS_SOLID,   M_BDPN),
        (LABEL_LINEAR,    COLOR_LINEAR,    LW_ENS,      LS_SOLID,   M_LINEAR),
        (LABEL_FISHER,    COLOR_FISHER,    LW_BASELINE, LS_SOLID,   M_FISHER),
        (LABEL_TIES,      COLOR_TIES,      LW_BASELINE, LS_DASHED,  M_TIES),
        (LABEL_TASKARITH, COLOR_TASKARITH, LW_BASELINE, LS_DASHDOT, M_TASK),
    ]


def render_section(fig, gs, rob, kind, exclude):
    """populate one 5x3 GridSpec block with the given exclude set."""
    n_rows = len(ROWS)
    for r_idx, (row_title, metric, yticks, ylim, decimals) in enumerate(ROWS):
        anchor0 = metric in ('nll', 'ece')
        for c_idx, (mode, col_title) in enumerate(MODES):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            plot_cell(ax, rob, kind, mode, metric, exclude)
            style_axes(ax)
            # per-subplot tight y-axis; ylim[1] is the row's clip ceiling.
            fit_yaxis(ax, anchor0, ylim[1] if anchor0 else None)
            if r_idx == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold', pad=10)
            if r_idx == n_rows - 1:
                ax.set_xlabel(r'Noise magnitude  $\varepsilon$', fontsize=12,
                              labelpad=6)
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=14, fontweight='bold',
                              labelpad=8)


def render_legend(fig, x_anchor, y_anchor, exclude):
    """add a section legend, dropping entries whose method is excluded."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=c, linewidth=lw, linestyle=ls, label=lbl)
        for (lbl, c, lw, ls, name) in _legend_entries() if name not in exclude
    ]
    fig.legend(handles=handles, loc='center left',
               bbox_to_anchor=(x_anchor, y_anchor), frameon=False, fontsize=11)


def make_figure(rob, kind, out_path):
    """one tall PDF: main-text section on top, appendix on bottom, gap between."""
    n_rows, n_cols = len(ROWS), len(MODES)
    fig = plt.figure(figsize=(15, 26.84))

    # vertical layout (figure-fraction, top -> bottom): main [0.97, 0.55], gap,
    # appendix [0.45, 0.03].
    gs_main = fig.add_gridspec(n_rows, n_cols, hspace=0.50, wspace=0.40,
                               left=0.08, right=0.74, top=0.97, bottom=0.55)
    gs_app = fig.add_gridspec(n_rows, n_cols, hspace=0.50, wspace=0.40,
                              left=0.08, right=0.74, top=0.45, bottom=0.03)

    render_section(fig, gs_main, rob, kind, exclude=EXCLUDE_FROM_MAIN)
    render_legend(fig, x_anchor=0.755, y_anchor=0.76, exclude=EXCLUDE_FROM_MAIN)

    render_section(fig, gs_app, rob, kind, exclude=frozenset())
    render_legend(fig, x_anchor=0.755, y_anchor=0.24, exclude=frozenset())

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None,
                    help="robustness_table.txt path (default: results/robustness_table.txt)")
    args = ap.parse_args()
    rob = RT.parse_robustness(args.table or (cfg.RESULTS_DIR / "robustness_table.txt"))
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("input", "weight"):
        make_figure(rob, kind, cfg.FIGURES_DIR / f"robustness_{kind}.pdf")


if __name__ == "__main__":
    main()
