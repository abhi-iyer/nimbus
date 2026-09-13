"""
robustness line plots: metrics vs noise magnitude \\varepsilon (LLM leg).

one tall PDF per noise-type (input, weight), with two stacked sections:
  top section: main-text version, Nimbus vs the |D|-member baseline ensemble.
  bottom section: appendix version, every method (adds linear interpolation and
  the parameter merges).
each subplot's y-axis is fit tight to its own top-most line; NLL/ECE keep a floor
at 0 and clip blow-ups via a high ceiling.

both sections share a 5 rows x 3 cols layout:
  rows: accuracy, top-3, top-5, NLL, ECE
  cols: logit average, softmax average, param average
each subplot plots every (non-excluded) method on the 'all' scope vs \\varepsilon.

color encodes the method family:
  purple = Nimbus
  coral = |D|-member baseline ensemble
  steel blue = linear interpolation (bumps gated off)
  grayscale = parameter merges of the vertices (Fisher / TIES)

data: ../results/robustness_table.txt

usage:
    python robustness_line_plots.py --noise input
    python robustness_line_plots.py --all
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

# --- colors (matched to experiments/cifar/figures/robustness_line_plots.py) ---
COLOR_SIMPLEX     = '#B39DDB'  # pastel purple, our method
COLOR_LINEAR      = '#4C72B0'  # steel blue, linear interpolation (bumps off)
COLOR_BASELINE    = '#F4A582'
COLOR_BASELINE_DPN = '#3CAEA3'  # teal, larger |D|+N ensemble
# parameter merges: grayscale, linestyles separate similar luminances.
COLOR_FISHER      = '#1a1a1a'  # ~10% gray
COLOR_TIES        = '#4d4d4d'  # ~30% gray

# --- labels ---
LABEL_SIMPLEX     = 'Nimbus (ours)'
LABEL_LINEAR      = 'Linear interpolation'
LABEL_BASELINE    = r'Baseline ensemble ($|\mathcal{D}|$)'
LABEL_BASELINE_DPN = r'Baseline ensemble ($|\mathcal{D}|{+}N$)'
LABEL_FISHER      = 'Fisher Merge'
LABEL_TIES        = 'TIES Merge'

# --- method names as they appear in robustness_table.txt ---
NAME_SIMPLEX_LOGIT    = 'Simplex / logit ens uniform degree'
NAME_SIMPLEX_SOFTMAX  = 'Simplex / softmax ens uniform degree'
NAME_SIMPLEX_PARAM    = 'Simplex / param avg'
NAME_LINEAR_LOGIT     = 'Linear logit ens'
NAME_LINEAR_SOFTMAX   = 'Linear softmax ens'
NAME_LINEAR_PARAM     = 'Linear param avg'
NAME_BASELINE_LOGIT   = 'Baseline logit ens'
NAME_BASELINE_SOFTMAX = 'Baseline softmax ens'
NAME_BASELINE_PARAM   = 'Baseline param avg'
NAME_BDPN_LOGIT       = 'Baseline+N logit ens'
NAME_BDPN_SOFTMAX     = 'Baseline+N softmax ens'
NAME_BDPN_PARAM       = 'Baseline+N param avg'
NAME_FISHER           = 'Fisher merge'
NAME_TIES             = 'TIES merge'

# per-column method names for the ensembles (logit / softmax / param avg).
LINEAR_BY_COL   = {'logit': NAME_LINEAR_LOGIT, 'softmax': NAME_LINEAR_SOFTMAX,
                   'param': NAME_LINEAR_PARAM}
BASELINE_BY_COL = {'logit': NAME_BASELINE_LOGIT, 'softmax': NAME_BASELINE_SOFTMAX,
                   'param': NAME_BASELINE_PARAM}
BDPN_BY_COL     = {'logit': NAME_BDPN_LOGIT, 'softmax': NAME_BDPN_SOFTMAX,
                   'param': NAME_BDPN_PARAM}

# --- line styles ---
LS_SOLID   = '-'
LS_DASHED  = '--'
LS_DASHDOT = '-.'
LS_DOTTED  = ':'

# --- line weights ---
LW_SIMPLEX  = 2.0    # Simplex, alone at the top
LW_ENS      = 1.6    # baseline ensemble + linear interpolation
LW_BASELINE = 1.0    # parameter merges

# --- z-orders (background to foreground) ---
Z_BASELINE  = 2      # single-model parameter merges
Z_ENS       = 4      # ensembles (baseline + linear)
Z_SIMPLEX   = 5      # Simplex on top

# main-text (top) section: Nimbus vs the |D| baseline ensemble and the |D|+N
# ensemble. linear interpolation and the merges go to the appendix section only.
EXCLUDE_FROM_MAIN = frozenset({
    NAME_LINEAR_LOGIT, NAME_LINEAR_SOFTMAX, NAME_LINEAR_PARAM,
    NAME_FISHER, NAME_TIES,
})


# universal baselines: single models drawn on every subplot.
# (method_name, label, color, linestyle, linewidth, zorder)
UNIVERSAL_BASELINES = [
    (NAME_FISHER,     LABEL_FISHER,     COLOR_FISHER,     LS_SOLID,   LW_BASELINE, Z_BASELINE),
    (NAME_TIES,       LABEL_TIES,       COLOR_TIES,       LS_DASHED,  LW_BASELINE, Z_BASELINE),
]


def col_specific_ensembles(col_kind: str):
    """
    (name, label, color, linestyle, linewidth, zorder) tuples for the linear and
    baseline ensembles in the given column.
    """
    return [
        (LINEAR_BY_COL[col_kind],   LABEL_LINEAR,   COLOR_LINEAR,   LS_SOLID, LW_ENS, Z_ENS),
        (BASELINE_BY_COL[col_kind], LABEL_BASELINE, COLOR_BASELINE, LS_SOLID, LW_ENS, Z_ENS),
        (BDPN_BY_COL[col_kind], LABEL_BASELINE_DPN, COLOR_BASELINE_DPN, LS_SOLID, LW_ENS, Z_ENS),
    ]


# Acc/Top-N share a single scale across both sections.
ACC_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ACC_LIM   = (0.0, 0.7)


# each subplot is re-fit tight by fit_yaxis; NLL/ECE ymax is only an outlier clip
# ceiling and the ticks/decimals are unused fallbacks.
ROWS = [
    ('Accuracy', 0, ACC_TICKS, ACC_LIM, 1, 'linear'),
    ('Top-3',    1, ACC_TICKS, ACC_LIM, 1, 'linear'),
    ('Top-5',    2, ACC_TICKS, ACC_LIM, 1, 'linear'),
    ('NLL',      3, [0.0, 5.0, 10.0, 15.0, 20.0], (0.0, 20.0), 0, 'linear'),
    ('ECE',      4, [0.0, 0.25, 0.5, 0.75, 1.0],  (0.0, 1.0),  2, 'linear'),
]

# (col title, simplex method name, col_kind)
COLUMNS = [
    ('Logit average',   NAME_SIMPLEX_LOGIT,   'logit'),
    ('Softmax average', NAME_SIMPLEX_SOFTMAX, 'softmax'),
    ('Param average',   NAME_SIMPLEX_PARAM,   'param'),
]

XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
XLIM   = (0.0, 1.0)


def parse_robustness(path: Path) -> dict:
    """parse robustness_table.txt into
    {(method, noise): {eps: {scope: [acc, top3, top5, nll, ece]}}}."""
    out: dict = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith('Method') or line.startswith('-') or line.startswith('='):
            continue
        cells = [c.strip() for c in line.split('|')]
        if len(cells) < 9:
            continue
        method, noise, eps_str, scope = cells[0], cells[1], cells[2], cells[3]
        try:
            eps = float(eps_str)
            vals = [float(c) for c in cells[4:9]]
        except ValueError:
            continue
        out.setdefault((method, noise), {}).setdefault(eps, {})[scope] = vals
    return out


def method_curve(parsed, method, noise, metric_idx, scope='all'):
    d = parsed.get((method, noise), {})
    eps_vals = sorted(d.keys())
    if not eps_vals:
        raise KeyError(f'no data for ({method!r}, {noise!r})')
    y = np.array([d[e][scope][metric_idx] for e in eps_vals])
    return np.array(eps_vals), y


def fmt_tick_linear(v: float, decimals: int) -> str:
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
    """fewest decimals that render `step` (hence every tick) exactly."""
    for d in range(5):
        if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-6:
            return d
    return 4


def fit_yaxis(ax, anchor0, cap):
    """
    tighten one subplot's y-axis just above its top-most line. lower-is-better
    rows (anchor0=True) pin the floor at 0 and clip the top at `cap`; bounded
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


def style_axes(ax, xticks, xlim, yticks, ylim, decimals_y: int, yscale: str,
               show_xticklabels: bool, show_yticklabels: bool):
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)

    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    if show_xticklabels:
        ax.set_xticklabels([fmt_tick_linear(t, 1) for t in xticks])
    else:
        ax.set_xticklabels([])
    if show_yticklabels:
        ax.set_yticklabels([fmt_tick_linear(t, decimals_y) for t in yticks])
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='both', which='both', direction='out',
                   length=4, width=0.8, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.9)


def _draw(ax, parsed, method, noise, metric_idx, color, linewidth, zorder,
          linestyle='-'):
    """draw one method's curve; silently no-op if absent for this noise type."""
    try:
        eps, y = method_curve(parsed, method, noise, metric_idx)
    except KeyError:
        return
    ax.plot(eps, y, color=color, linewidth=linewidth, linestyle=linestyle,
            zorder=zorder)


def plot_cell(ax, parsed, noise, simplex_name, col_kind, metric_idx,
              exclude=frozenset()):
    """every (non-excluded) method on the subplot, background to foreground so
    Simplex stays on top."""
    # universal single-model baselines (parameter merges), drawn on every column.
    for name, _label, color, ls, lw, z in UNIVERSAL_BASELINES:
        if name in exclude:
            continue
        _draw(ax, parsed, name, noise, metric_idx, color, lw, z, linestyle=ls)

    # column-specific ensembles (small + large).
    for name, _label, color, ls, lw, z in col_specific_ensembles(col_kind):
        if name in exclude:
            continue
        _draw(ax, parsed, name, noise, metric_idx, color, lw, z, linestyle=ls)

    # foreground: Simplex (never excluded).
    _draw(ax, parsed, simplex_name, noise, metric_idx,
          COLOR_SIMPLEX, LW_SIMPLEX, Z_SIMPLEX, linestyle=LS_SOLID)


# legend entries: (label, color, linewidth, linestyle, methods_represented).
# an entry is kept iff a represented method is not excluded.
def _legend_entries():
    return [
        (LABEL_SIMPLEX,     COLOR_SIMPLEX,    LW_SIMPLEX,  LS_SOLID,
         frozenset([NAME_SIMPLEX_LOGIT, NAME_SIMPLEX_SOFTMAX, NAME_SIMPLEX_PARAM])),
        # coral: the |D|-member compute+param-matched baseline ensemble
        (LABEL_BASELINE,    COLOR_BASELINE,   LW_ENS,      LS_SOLID,
         frozenset([NAME_BASELINE_LOGIT, NAME_BASELINE_SOFTMAX, NAME_BASELINE_PARAM])),
        # teal: the larger |D|+N ensemble
        (LABEL_BASELINE_DPN, COLOR_BASELINE_DPN, LW_ENS, LS_SOLID,
         frozenset([NAME_BDPN_LOGIT, NAME_BDPN_SOFTMAX, NAME_BDPN_PARAM])),
        # steel blue: linear interpolation (bumps gated off)
        (LABEL_LINEAR,      COLOR_LINEAR,     LW_ENS,      LS_SOLID,
         frozenset([NAME_LINEAR_LOGIT, NAME_LINEAR_SOFTMAX, NAME_LINEAR_PARAM])),
        # grayscale: parameter merges of the vertices
        (LABEL_FISHER,      COLOR_FISHER,     LW_BASELINE, LS_SOLID,
         frozenset([NAME_FISHER])),
        (LABEL_TIES,        COLOR_TIES,       LW_BASELINE, LS_DASHED,
         frozenset([NAME_TIES])),
    ]


def render_section(fig, gs, parsed, noise, exclude, rows):
    """populate one 5x3 GridSpec block with the given exclude set and row config."""
    n_rows = len(rows)
    for r_idx, (row_title, metric_idx, yticks, ylim, decimals, yscale) in enumerate(rows):
        for c_idx, (col_title, simplex_name, col_kind) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            plot_cell(ax, parsed, noise,
                      simplex_name=simplex_name,
                      col_kind=col_kind,
                      metric_idx=metric_idx,
                      exclude=exclude)
            style_axes(ax, XTICKS, XLIM, yticks, ylim, decimals, yscale,
                       show_xticklabels=True, show_yticklabels=True)
            # per-subplot tight y-axis (overrides style_axes).
            anchor0 = row_title in ('NLL', 'ECE')
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
    """add a section legend, dropping entries whose methods are all excluded."""
    from matplotlib.lines import Line2D
    entries = [e for e in _legend_entries() if not e[4].issubset(exclude)]
    handles = [
        Line2D([], [], color=c, linewidth=lw, linestyle=ls, label=lbl)
        for (lbl, c, lw, ls, _) in entries
    ]
    fig.legend(handles=handles, loc='center left',
               bbox_to_anchor=(x_anchor, y_anchor), frameon=False, fontsize=11)


def make_figure(noise: str, out_path: Path):
    """render one tall PDF: main-text section on top, appendix on bottom."""
    path = RESULTS_DIR / 'robustness_table.txt'
    parsed = parse_robustness(path)

    n_rows, n_cols = len(ROWS), len(COLUMNS)
    # figure height chosen so each subplot lands near the golden ratio.
    fig = plt.figure(figsize=(15, 26.84))

    # vertical layout (figure-fraction coords, top to bottom):
    #   [0.97, 0.55]  main-text 5x3
    #   [0.55, 0.45]  gap
    #   [0.45, 0.03]  appendix 5x3
    gs_main = fig.add_gridspec(
        n_rows, n_cols, hspace=0.50, wspace=0.40,
        left=0.08, right=0.74, top=0.97, bottom=0.55,
    )
    gs_app = fig.add_gridspec(
        n_rows, n_cols, hspace=0.50, wspace=0.40,
        left=0.08, right=0.74, top=0.45, bottom=0.03,
    )

    render_section(fig, gs_main, parsed, noise,
                   exclude=EXCLUDE_FROM_MAIN, rows=ROWS)
    render_legend(fig, x_anchor=0.755, y_anchor=0.76,
                  exclude=EXCLUDE_FROM_MAIN)

    render_section(fig, gs_app, parsed, noise,
                   exclude=frozenset(), rows=ROWS)
    render_legend(fig, x_anchor=0.755, y_anchor=0.24, exclude=frozenset())

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    # 'input' is uniform token replacement; 'weight' is relative gaussian noise
    # on the parameters.
    ap.add_argument('--noise', choices=['input', 'weight'], default=None,
                    help='noise type; default draws both.')
    args = ap.parse_args()

    noises = [args.noise] if args.noise is not None else ['input', 'weight']

    for noise in noises:
        make_figure(noise, FIG_DIR / f'robustness_{noise}.pdf')


if __name__ == '__main__':
    main()
