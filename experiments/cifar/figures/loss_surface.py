"""
loss / metric surfaces on the 2-simplex for the N=3 Simplex model.

evaluates all five metrics (acc, top3, top5, nll, ece) on a regular triangular
grid over the N=3 CIFAR-100 task subset and caches to
../results/N=3/loss_surface.pkl. re-running hits the cache and only redraws;
--refresh re-evaluates.

output: loss_surface_N3.pdf, a 6x1 strip of filled contour triangles (one per
metric) with topographic overlay lines. a sixth panel shows the nll of the
linear interpolation of the vertex params (no bump terms). both nll panels
share one color scale capped at the max linear-interp nll.

usage:
    python -m experiments.cifar.figures.loss_surface
    python -m experiments.cifar.figures.loss_surface --refresh
"""
import argparse
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call

FIG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FIG_DIR.parent / 'results'
REPO_ROOT = FIG_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nimbus.simplex import SimplexInterpolator, build_bump_params, bp_from_cpu, dg_from_cpu  # noqa: E402
from nimbus.metrics import metrics_from_logits  # noqa: E402
from nimbus.seeds import device  # noqa: E402
from experiments.cifar.config import GLOBAL_SEED  # noqa: E402
from experiments.cifar.data import CIFAR100_CL, CL_DataLoader  # noqa: E402
from experiments.cifar.model import make_convnext, load_state  # noqa: E402

try:
    from ._common import setup_mpl
except ImportError:
    from _common import setup_mpl
setup_mpl()

N = 3
DEGREES = [2, 3]

# equilateral triangle vertices in cartesian plotting coordinates.
VERTICES_2D = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2],
])

# (dx, dy, ha, va) per-vertex label offset.
LABEL_OFFSETS = [
    (-0.025, -0.025, 'right', 'top'),
    ( 0.025, -0.025, 'left',  'top'),
    ( 0.0,    0.04,  'center', 'bottom'),
]

# (title, dict key, cbar ticks, (vmin, vmax), decimals). acc/top3/top5 share a
# fixed [0, 1] range; the nll row's range is overridden in plot_all_metrics.
_ACC_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
_ACC_RANGE = (0.0, 1.0)
METRICS = [
    ('Accuracy', 'acc',  _ACC_TICKS,                 _ACC_RANGE,   2),
    ('Top-3',    'top3', _ACC_TICKS,                 _ACC_RANGE,   2),
    ('Top-5',    'top5', _ACC_TICKS,                 _ACC_RANGE,   2),
    ('NLL',      'nll',  [0.0, 0.5, 1.0, 1.5, 2.0],  (0.0, 2.0),   1),
    ('ECE',      'ece',  [0.0, 0.1, 0.2, 0.3],       (0.0, 0.3),   1),
]

# FILL_LEVELS: number of filled contour levels. SUBDIVS_PER_TICK: topographic
# line subdivisions per tick span.
FILL_LEVELS = 40
SUBDIVS_PER_TICK = 3


# sampling

def simplex_grid(n):
    """
    regular triangular grid on the 2-simplex: (M, 3) barycentric points with
    M = (n+1)(n+2)/2.
    """
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return np.array(pts, dtype=np.float32)


# evaluation

def task_split_for_N(N):
    all_tasks = torch.randperm(
        N * 10, generator=torch.Generator().manual_seed(GLOBAL_SEED)
    )
    return [all_tasks[i*10:(i+1)*10] for i in range(N)]


def load_model():
    ckpt_dir = RESULTS_DIR / f'N={N}' / 'ckpts'
    task_split = task_split_for_N(N)
    all_classes = [c.item() for t in task_split for c in t]
    num_classes = max(all_classes) + 1

    sds = torch.load(ckpt_dir / 'vertex.pt', map_location=device)
    vertex_models = []
    for vi in range(N):
        m = make_convnext(seed=vi + 1, num_classes=num_classes)
        load_state(m, sds[vi])
        vertex_models.append(m.to(device).eval())

    vertex_params = [
        {n: p.detach().clone()
         for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    for vm in vertex_models:
        for p in vm.parameters():
            p.requires_grad_(False)

    post = torch.load(ckpt_dir / 'simplex_post.pt', map_location=device)
    bump_params = bp_from_cpu(post['bp'])
    degree_gates = dg_from_cpu(post['dg'])
    for pd in bump_params.values():
        for p in pd.values():
            p.requires_grad_(False)
    for g in degree_gates.values():
        g.requires_grad_(False)

    _, alpha_to_key, alphas_by_degree = build_bump_params(vertex_models, DEGREES)

    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)

    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )
    interp.cache_bump_flat()

    return dummy, interp, all_classes


def cache_test_tensors(all_classes):
    dl = CL_DataLoader(CIFAR100_CL(split='test', seed=GLOBAL_SEED),
                       batch_size=128, shuffle=True, seed=GLOBAL_SEED)
    dl.set_epoch_data(all_classes)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def linear_params(interp, w):
    """
    params for the linear interpolation of the vertices at w: the barycentric
    blend w @ vertex_flat with no bump terms.
    """
    flat = w @ interp.vertex_flat
    param_map = {}
    offset = 0
    for name, shape, sz in zip(interp.param_names, interp.param_shapes,
                               interp.param_sizes):
        param_map[name] = flat[offset:offset + sz].view(shape)
        offset += sz
    return {**param_map, **interp.fixed_sd}


@torch.no_grad()
def _metrics_from_params(dummy, params, x_all, y_all, batch_size=512):
    all_logits = []
    for i in range(0, len(x_all), batch_size):
        logits = functional_call(dummy, params, (x_all[i:i+batch_size],))
        all_logits.append(logits)
    return metrics_from_logits(torch.cat(all_logits, dim=0), y_all)


@torch.no_grad()
def metrics_at_w(dummy, interp, w_np, x_all, y_all, batch_size=512):
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    return _metrics_from_params(dummy, interp(w), x_all, y_all, batch_size)


@torch.no_grad()
def linear_metrics_at_w(dummy, interp, w_np, x_all, y_all, batch_size=512):
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    return _metrics_from_params(dummy, linear_params(interp, w),
                                x_all, y_all, batch_size)


def run_eval(n_per_side):
    dummy, interp, all_classes = load_model()
    print(f'caching test tensors ({len(all_classes)} classes)...')
    x_all, y_all = cache_test_tensors(all_classes)
    print(f'  {len(x_all)} test images cached')

    ws = simplex_grid(n_per_side)
    M = len(ws)
    print(f'evaluating metrics at {M} regular-grid points (n={n_per_side})...')

    metric_keys = [entry[1] for entry in METRICS]
    metrics = {key: np.empty(M, dtype=np.float64) for key in metric_keys}
    # nll of the linear interpolation (no bump terms) at each grid point.
    metrics['nll_linear'] = np.empty(M, dtype=np.float64)
    log_every = max(1, M // 20)
    for i in range(M):
        m = metrics_at_w(dummy, interp, ws[i], x_all, y_all)
        for key in metric_keys:
            metrics[key][i] = m[key]
        ml = linear_metrics_at_w(dummy, interp, ws[i], x_all, y_all)
        metrics['nll_linear'][i] = ml['nll']
        if (i + 1) % log_every == 0:
            print(f'  [{i+1}/{M}] nll={m["nll"]:.3f}  '
                  f'nll_lin={ml["nll"]:.3f}  acc={m["acc"]:.3f}')
    return ws, metrics


# plotting

def barycentric_to_xy(w_arr):
    return w_arr @ VERTICES_2D


def fmt_tick(v, decimals):
    if np.isclose(v, 0.0):
        return '0.0'
    return f'{v:.{decimals}f}'


def draw_panel(ax, ws, values, title, ticks, vmin, vmax, decimals, cmap_name):
    xy = barycentric_to_xy(ws)
    x, y = xy[:, 0], xy[:, 1]

    # dense fill so the color gradient looks continuous.
    fill_levels = np.linspace(vmin, vmax, FILL_LEVELS)
    extend = 'max' if bool(np.nanmax(values) > vmax) else 'neither'
    cs = ax.tricontourf(
        x, y, values, levels=fill_levels, cmap=cmap_name,
        vmin=vmin, vmax=vmax, extend=extend, antialiased=True,
    )

    # sparse topographic overlay at SUBDIVS_PER_TICK subdivisions per tick.
    n_topo = SUBDIVS_PER_TICK * (len(ticks) - 1) + 1
    topo_levels = np.linspace(vmin, vmax, n_topo)
    ax.tricontour(
        x, y, values, levels=topo_levels,
        colors='black', linewidths=0.4, alpha=0.6,
    )

    tri_x = np.append(VERTICES_2D[:, 0], VERTICES_2D[0, 0])
    tri_y = np.append(VERTICES_2D[:, 1], VERTICES_2D[0, 1])
    ax.plot(tri_x, tri_y, color='black', linewidth=1.0,
            clip_on=False, zorder=5)

    for i, (vx, vy) in enumerate(VERTICES_2D):
        ax.plot(vx, vy, 'o', color='black', markersize=5,
                clip_on=False, zorder=6)
        dx, dy, ha, va = LABEL_OFFSETS[i]
        ax.text(vx + dx, vy + dy, f'Task {i}',
                ha=ha, va=va, fontsize=11, fontweight='bold')

    # room around the triangle for labels and title.
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.10, np.sqrt(3) / 2 + 0.10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)

    cbar = ax.figure.colorbar(cs, ax=ax, shrink=0.78, pad=0.04, fraction=0.06)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([fmt_tick(t, decimals) for t in ticks])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(direction='out', length=4, labelsize=10, colors='black')
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)


def nice_ticks(vmax, target=5):
    """
    about `target` ticks from 0 up to vmax, snapped to a 1/2/2.5/5 x 10^k step.
    """
    if vmax <= 0:
        return [0.0]
    raw = vmax / target
    mag = 10.0 ** np.floor(np.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    n = int(np.floor(vmax / step + 1e-9))
    return [round(step * i, 4) for i in range(n + 1)]


def plot_all_metrics(ws, metrics, out_path, cmap):
    # shared nll color scale: vmax is the max linear-interp nll, used by both
    # nll panels.
    nll_vmax = float(np.nanmax(metrics['nll_linear']))
    nll_ticks = nice_ticks(nll_vmax)

    # panel specs (title, values, ticks, vmin, vmax, decimals), with the linear-
    # interp nll panel inserted after the nll panel.
    panels = []
    for title, key, ticks, (vmin, vmax), decimals in METRICS:
        if key == 'nll':
            panels.append((title, metrics['nll'], nll_ticks, 0.0, nll_vmax, 1))
            panels.append(('NLL (linear interp)', metrics['nll_linear'],
                           nll_ticks, 0.0, nll_vmax, 1))
        else:
            panels.append((title, metrics[key], ticks, vmin, vmax, decimals))

    n_panels = len(panels)
    # taller figure and larger hspace so titles clear the panel above.
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(6.5, 5.5 * n_panels),
        gridspec_kw=dict(hspace=0.50),
    )

    for ax, (title, values, ticks, vmin, vmax, decimals) in zip(axes, panels):
        draw_panel(ax, ws, values, title, ticks, vmin, vmax,
                   decimals, cmap)

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


N_GRID = 40          # grid resolution per side; total pts = (n+1)(n+2)/2
CMAP = 'viridis'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='force re-evaluation, ignoring cached pkl.')
    args = ap.parse_args()

    cache_path = RESULTS_DIR / f'N={N}' / 'loss_surface.pkl'

    # cache shape: {'ws': (M, 3), 'metrics': {...nll_linear: (M,)}, 'n': n_per_side}.
    # older caches (missing nll_linear or a different grid) are rejected.
    loaded = None
    if cache_path.exists() and not args.refresh:
        with cache_path.open('rb') as f:
            loaded = pickle.load(f)
        if ('metrics' not in loaded or loaded.get('n') != N_GRID
                or 'nll_linear' not in loaded['metrics']):
            print(f'cache at {cache_path} is for a different schema/grid -- '
                  f're-evaluating')
            loaded = None

    if loaded is None:
        ws, metrics = run_eval(N_GRID)
        with cache_path.open('wb') as f:
            pickle.dump({'ws': ws, 'metrics': metrics, 'n': N_GRID}, f)
        print(f'cached to {cache_path}')
    else:
        print(f'loaded cached samples from {cache_path}')
        ws, metrics = loaded['ws'], loaded['metrics']

    out_path = FIG_DIR / 'loss_surface_N3.pdf'
    plot_all_metrics(ws, metrics, out_path, CMAP)


if __name__ == '__main__':
    main()
