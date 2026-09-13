"""
misc_figs.pdf: the manifold-plateau figure (real data), a 1x2 figure.

  (A) 3d schematic: vertex -> centre along the manifold vs the linear average.
  (B) CIFAR N=3: multitask NLL vs normalized parameter step (rays).

usage:
    python -m misc.misc_figs             # compute the ray cache if missing, then draw
    python -m misc.misc_figs --refresh   # recompute the ray cache, then draw
"""
import argparse
import pickle

import numpy as np
import torch

from misc._common import (
    OUT, PURPLE, PURPLE_D, CORAL, CORAL_D, device,
    flat_to_params, nimbus_flat, mean_pairwise_dist, setup_mpl,
    load_cifar,
)

# computation

def t_grid(n_near=12, n_far=16):
    near = np.linspace(0.0, 0.15, n_near)
    far = np.linspace(0.15, 1.0, n_far)[1:]
    return np.concatenate([near, far])


def compute_rays(bundle, t_pts):
    interp, vflat = bundle['interp'], bundle['vertex_flat']
    N = bundle['N']
    uniform = torch.full((N,), 1.0 / N, device=device)
    out = {m: {'dist': np.zeros((N, len(t_pts))), 'loss': np.zeros((N, len(t_pts)))}
           for m in ('nimbus', 'linear')}

    for i in range(N):
        e = torch.zeros(N, device=device); e[i] = 1.0
        v_i = vflat[i]
        for ti, t in enumerate(t_pts):
            w = (1 - t) * e + t * uniform
            # nimbus
            nf = nimbus_flat(interp, w)
            out['nimbus']['dist'][i, ti] = torch.norm(nf - v_i).item()
            out['nimbus']['loss'][i, ti] = bundle['loss_fn'](flat_to_params(interp, nf))
            # linear weight average along the same ray
            lf = w @ vflat
            out['linear']['dist'][i, ti] = torch.norm(lf - v_i).item()
            out['linear']['loss'][i, ti] = bundle['loss_fn'](flat_to_params(interp, lf))
        print(f"    [{bundle['kind']} N={N}] ray {i+1}/{N} done")

    dbar = mean_pairwise_dist(vflat)
    for m in out:
        out[m]['dist'] /= dbar
    out['dbar'] = dbar
    out['t'] = t_pts
    return out


CASES = {'cifar3': ('cifar', 3)}


def load_case(name):
    _kind, N = CASES[name]
    return load_cifar(N)


def do_compute(refresh=False):
    """ray traces for panel (B): the vertex->centre curves for each case."""
    tp = t_grid()
    for name in CASES:
        pk = OUT / f'{name}.pkl'
        data = pickle.load(pk.open('rb')) if pk.exists() else {}
        if 'rays' in data and not refresh:
            print(f"[compute] {name} cached -> {pk}")
            continue
        print(f"[compute] {name} ...")
        bundle = load_case(name)
        rays = compute_rays(bundle, tp)
        data.update({'rays': rays, 'kind': bundle['kind'], 'N': bundle['N']})
        with pk.open('wb') as f:
            pickle.dump(data, f)
        print(f"[compute] {name} cached -> {pk}  (dbar={rays['dbar']:.3g})")


# plotting

def _agg(rays, method):
    """mean parameter-distance and mean, min, max loss across the N rays."""
    d = rays[method]['dist']; l = rays[method]['loss']
    return d.mean(0), l.mean(0), l.min(0), l.max(0)


def draw_panel_A(ax):
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *a, **k):
            super().__init__((0, 0), (0, 0), *a, **k)
            self._xyz = (x, y, z); self._d = (dx, dy, dz)

        def do_3d_projection(self, renderer=None):
            x, y, z = self._xyz; dx, dy, dz = self._d
            xs, ys, zs = proj3d.proj_transform((x, x + dx), (y, y + dy), (z, z + dz), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return float(np.min(zs))

    V2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    AMP = 0.34
    bary = lambda a, b, c: a * V2[0] + b * V2[1] + c * V2[2]
    bump = lambda a, b, c: AMP * (1 - a * a - b * b - c * c)

    n = 90; pts, z = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a, b, c = i / n, j / n, (n - i - j) / n
            pts.append(bary(a, b, c)); z.append(bump(a, b, c))
    pts = np.array(pts); z = np.array(z)
    # pastel-pink sheet
    ax.plot_trisurf(pts[:, 0], pts[:, 1], z, color='#F9C2D6', alpha=0.45, shade=False,
                    linewidth=0, antialiased=True, edgecolor='none')

    cxy = V2.mean(0); z_man = bump(1/3, 1/3, 1/3)

    def arrow(p0, p1, col, lw, mut=13):
        d = p1 - p0
        ax.add_artist(Arrow3D(*p0, *d, mutation_scale=mut, lw=lw, arrowstyle='-|>',
                              color=col, zorder=20))

    def edge_tip(path, gap):
        """trim a path (ending at a centroid) so it terminates `gap` data-units
        before the centre, on the dot's outer outline."""
        end = path[-1]
        d = np.linalg.norm(path - end, axis=1)
        keep = path[d >= gap]
        if len(keep) < 1:
            return path[:1]
        direction = end - keep[-1]
        n = np.linalg.norm(direction)
        if n < 1e-9:
            return keep
        tip = end - gap * direction / n           # point on the dot edge
        return np.vstack([keep, tip])

    # gap ~= centroid dot radius in data units
    GAP_LIN, GAP_CUR = 0.034, 0.038
    for i in range(3):
        e = np.eye(3)[i]; ts = np.linspace(0, 1, 90)
        ws = [(1 - t) * e + t * np.full(3, 1/3) for t in ts]
        lin = np.column_stack([np.array([bary(*w) for w in ws]), np.zeros(len(ts))])
        cur = np.column_stack([np.array([bary(*w) for w in ws]), [bump(*w) for w in ws]])
        lt, ct = edge_tip(lin, GAP_LIN), edge_tip(cur, GAP_CUR)
        if i == 0:
            ax.plot(*lt[:-1].T, color=CORAL, lw=1.8, zorder=12); arrow(lt[-2], lt[-1], CORAL, 1.8, mut=11)
            ax.plot(*ct[:-1].T, color=PURPLE, lw=2.0, zorder=14); arrow(ct[-2], ct[-1], PURPLE, 2.0, mut=11)
        else:
            ax.plot(*lt[:-1].T, color=CORAL, lw=1.0, ls=(0, (4, 3)), alpha=0.8, zorder=8)
            arrow(lt[-2], lt[-1], CORAL, 1.0, mut=8)
            ax.plot(*ct[:-1].T, color=PURPLE, lw=1.0, ls=(0, (4, 3)), alpha=0.8, zorder=9)
            arrow(ct[-2], ct[-1], PURPLE, 1.0, mut=8)

    for i, v in enumerate(V2):
        ax.scatter([v[0]], [v[1]], [0], s=45, color='black', depthshade=False, zorder=18)
        ax.text(v[0], v[1], -0.02, f'$\\theta_{i}$', fontsize=12, fontweight='bold',
                ha='center', va='top')
    ax.scatter([cxy[0]], [cxy[1]], [0], s=70, color=CORAL, edgecolor=CORAL_D, lw=1.2,
               depthshade=False, alpha=1.0, zorder=30)
    ax.scatter([cxy[0]], [cxy[1]], [z_man], s=85, color=PURPLE, edgecolor=PURPLE_D, lw=1.2,
               depthshade=False, alpha=1.0, zorder=31)

    ax.legend(handles=[Line2D([], [], color=PURPLE, lw=2.8, label='Nimbus (on manifold)'),
                       Line2D([], [], color=CORAL, lw=2.6, label='Linear weight avg')],
              loc='upper left', frameon=False, fontsize=8.5, bbox_to_anchor=(-0.02, 0.99))
    ax.set_title('(A)  vertex $\\rightarrow$ centre: two paths, two centroids',
                 fontsize=11, pad=-2)
    ax.set_axis_off(); ax.view_init(elev=22, azim=-60); ax.set_box_aspect((1, 1, 0.6))


def _ray_panel(ax, name, title, legend=False):
    rays = pickle.load((OUT / f'{name}.pkl').open('rb'))['rays']
    for method, col in (('linear', CORAL), ('nimbus', PURPLE)):
        xm, ym, ylo, yhi = _agg(rays, method)
        ax.fill_between(xm, ylo, yhi, color=col, alpha=0.22, lw=0)
        ax.plot(xm, ym, color=col, lw=2.6 if method == 'nimbus' else 2.0,
                label='Nimbus' if method == 'nimbus' else 'Linear weight avg', zorder=5)
    ax.scatter([0], [rays['nimbus']['loss'].mean(0)[0]], s=36, color='black', zorder=6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'param step  $\|\Delta\theta\| / \bar{D}_{\mathrm{expert}}$', fontsize=11)
    ax.set_ylabel('multitask NLL', fontsize=11)
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='best')


def draw_main(plt):
    """draw misc_figs.pdf: (A) schematic, (B) CIFAR N=3 rays."""
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.26,
                          left=0.06, right=0.97, top=0.90, bottom=0.14)
    draw_panel_A(fig.add_subplot(gs[0, 0], projection='3d'))
    _ray_panel(fig.add_subplot(gs[0, 1]), 'cifar3', '(B)  CIFAR  N=3', legend=True)
    fig.savefig(OUT / 'misc_figs.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / 'misc_figs.pdf')


def do_plot():
    plt = setup_mpl()
    draw_main(plt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='recompute the ray cache even if present')
    args = ap.parse_args()

    do_compute(refresh=args.refresh)
    do_plot()


if __name__ == '__main__':
    main()
