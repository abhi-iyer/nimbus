"""
nimbus degree-set sweeps, one figure per experiment (sweep_<exp>.pdf).

for each experiment, retrain the bezier bumps for a family of candidate degree
sets D. panels, per experiment:

  (A) NLL from uniform sampling over Nimbus, by degree set D  (boxplot)
  (B) ensemble NLL by degree set D                            (logit / softmax)
  (C) logit-ens   robustness to input  noise, by degree set D
  (D) logit-ens   robustness to weight noise, by degree set D
  (E) softmax-ens robustness to input  noise, by degree set D
  (F) softmax-ens robustness to weight noise, by degree set D

per-set bumps are cached to degsweep_N{N}_D{tag}.pt.

usage:
    python -m misc.sweep --exp cifar7     # compute (if needed) + tables + sweep_cifar7.pdf
    python -m misc.sweep --all            # every experiment
    python -m misc.sweep --exp cifar7 --refresh   # recompute everything
"""
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from misc._common import (
    OUT, PURPLE, PURPLE_D, device,
    flat_to_params, deg_tag, deg_label, setup_mpl,
    load_cifar, _trainable_vp,
)
from nimbus.simplex import (
    SimplexInterpolator, build_bump_params, bp_from_cpu, dg_from_cpu,
    bp_to_cpu, dg_to_cpu, clone_bp, clone_dg,
)
from nimbus.metrics import metrics_from_logits, metrics_from_probs

EXPS = ('cifar3', 'cifar5', 'cifar7', 'cifar10')
EXP_N = {'cifar3': 3, 'cifar5': 5, 'cifar7': 7, 'cifar10': 10}

# candidate degree sets per experiment (display order: with-top family sparse->
# full, then no-top controls, then linear)
CONFIGS_BY_EXP = {
    'cifar3':  [(2,), (3,), (2, 3), ()],
    'cifar5':  [(2, 5), (2, 4, 5), (2, 3, 4, 5), (2,), (5,), (2, 3), ()],
    'cifar7':  [(2, 7), (2, 6, 7), (2, 5, 6, 7), (2, 3, 4, 5, 6, 7),
                (2,), (7,), (2, 3, 4), ()],
    'cifar10': [(2, 10), (2, 9, 10), (2, 8, 9, 10), (2, 7, 8, 9, 10), tuple(range(2, 11)),
                (2,), (10,), (2, 3, 4, 5), ()],
}

# optional per-(exp, noise, mode) y-axis cap for the robustness panels; off-scale
# lines keep an end tag annotated with their true value
ROB_YMAX = {
    ('cifar7', 'weight', 'logit'): 15,
}

EPSILONS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0])


# compute: CIFAR (per-model NLL distribution + logit/softmax ensemble)

def _cifar_train_ctx(N):
    """fresh (unfrozen) vertex models and loaders for retraining bumps."""
    from experiments.cifar.config import GLOBAL_SEED, BATCH_SIZE, RESULTS_DIR
    from experiments.cifar.data import CIFAR100_CL, CL_DataLoader
    from experiments.cifar.model import make_convnext, load_state

    ckpt_dir = RESULTS_DIR / f'N={N}' / 'ckpts'
    all_tasks = torch.randperm(N * 10, generator=torch.Generator().manual_seed(GLOBAL_SEED))
    task_split = [all_tasks[i * 10:(i + 1) * 10] for i in range(N)]
    num_classes = max(c.item() for t in task_split for c in t) + 1
    sds = torch.load(ckpt_dir / 'vertex.pt', map_location=device)
    vms = []
    for vi in range(N):
        m = make_convnext(seed=vi + 1, num_classes=num_classes)
        load_state(m, sds[vi])
        vms.append(m.to(device).eval())
    train_loader = CL_DataLoader(CIFAR100_CL(split='train', seed=GLOBAL_SEED),
                                 batch_size=BATCH_SIZE, shuffle=True, seed=GLOBAL_SEED)
    test_loader = CL_DataLoader(CIFAR100_CL(split='test', seed=GLOBAL_SEED),
                                batch_size=128, shuffle=True, seed=GLOBAL_SEED)
    return vms, task_split, train_loader, test_loader, ckpt_dir


def _cifar_bumps(D, N, vms, train_loader, test_loader, task_split, refresh=False):
    """load (or train and cache) the bezier bumps for degree set D."""
    from experiments.cifar.config import (
        EPOCHS, CAL_EPOCHS, LR, CAL_LR, N_CAL_SAMPLES, CAL_CHUNKS, stage_seed,
    )
    from experiments.cifar.train import train_simplex, calibrate_simplex

    tag = deg_tag(D)
    _, a2k, abd = build_bump_params(vms, D)
    cache = OUT / f'degsweep_N{N}_D{tag}.pt'
    if cache.exists() and not refresh:
        d = torch.load(cache, map_location=device)
        print(f"[degsweep] {deg_label(D):16s}  loaded cached bumps")
        return bp_from_cpu(d['bp']), dg_from_cpu(d['dg']), a2k, abd
    print(f"\n[degsweep] training D={D} ...")
    bp1, a2k, abd, dg1 = train_simplex(
        vertex_models=vms, train_loader=train_loader, test_loader=test_loader,
        task_split=task_split, epochs=EPOCHS, lr=LR,
        seed=stage_seed(N, 'simplex_pre'), degrees=list(D))
    bp, dg = clone_bp(bp1), clone_dg(dg1)
    calibrate_simplex(
        vms, bp, a2k, abd, dg, train_loader, test_loader, task_split,
        CAL_EPOCHS, CAL_LR, stage_seed(N, 'simplex_post'), list(D),
        n_samples=N_CAL_SAMPLES, chunks=CAL_CHUNKS)
    torch.save({'bp': bp_to_cpu(bp), 'dg': dg_to_cpu(dg)}, cache)
    print(f"[degsweep] saved bumps -> {cache}")
    return bp, dg, a2k, abd


def recompute_degsweep(exp, n=256, seed=1, refresh=False):
    """panels A/B: per-degree-set single-model NLL distribution and logit/softmax
    ensemble NLL over the same uniform-dirichlet manifold pool."""
    N = EXP_N[exp]
    configs = CONFIGS_BY_EXP[exp]
    bundle = load_cifar(N)
    vflat, dummy = bundle['vertex_flat'], bundle['dummy']
    logits_fn, y_all = bundle['logits_fn'], bundle['y_all']
    vms, task_split, train_loader, test_loader, _ = _cifar_train_ctx(N)
    vp = _trainable_vp(vms)

    rng = np.random.default_rng(seed)
    ws = [torch.from_numpy(rng.dirichlet(np.ones(N))).float().to(device) for _ in range(n)]
    nll, ens = {}, {}

    def _measure(params_of, D):
        tag = deg_tag(D)
        single, sum_logits, sum_probs = [], None, None
        for w in ws:
            lg = logits_fn(params_of(w))
            single.append(metrics_from_logits(lg, y_all)['nll'])
            p = F.softmax(lg, dim=-1)
            sum_logits = lg if sum_logits is None else sum_logits + lg
            sum_probs = p if sum_probs is None else sum_probs + p
        nll[tag] = np.array(single)
        ens[tag] = {'logit': metrics_from_logits(sum_logits / len(ws), y_all)['nll'],
                    'softmax': metrics_from_probs(sum_probs / len(ws), y_all)['nll']}
        print(f"[degsweep] {deg_label(D):16s}  single={nll[tag].mean():.4f}  "
              f"logit-ens={ens[tag]['logit']:.4f}  softmax-ens={ens[tag]['softmax']:.4f}")

    for D in configs:
        if not D:   # linear: weight-averaged vertices, no bumps
            _measure(lambda w: flat_to_params(bundle['interp'], w @ vflat), D)
            continue
        bp, dg, a2k, abd = _cifar_bumps(D, N, vms, train_loader, test_loader,
                                        task_split, refresh=refresh)
        for pd in bp.values():
            for p in pd.values():
                p.requires_grad_(False)
        for g in dg.values():
            g.requires_grad_(False)
        interp = SimplexInterpolator(vp, bp, a2k, abd, dummy, degree_gates=dg)
        interp.cache_bump_flat()
        _measure(lambda w, _i=interp: _i(w), D)

    _store(exp, {'configs': list(configs), 'nll': nll, 'ens': ens, 'n': n, 'N': N})
    print(f"[degsweep] cached -> {OUT / f'{exp}.pkl'}")


def recompute_degsweep_rob(exp, n_pool=32, n_trials=2, seed=3, refresh=False):
    """panels C-F: robustness of the logit/softmax ensemble to input- and
    weight-space noise, per degree set. inference-only, reuses the cached bumps."""
    from experiments.cifar.baselines import noisy_input_sweep, noisy_weight_sweep
    from nimbus.ensemble import build_simplex_pool

    N = EXP_N[exp]
    configs = CONFIGS_BY_EXP[exp]
    bundle = load_cifar(N)
    vflat, dummy = bundle['vertex_flat'], bundle['dummy']
    vms, task_split, train_loader, test_loader, _ = _cifar_train_ctx(N)
    all_classes = [c.item() for t in task_split for c in t]

    rng = np.random.default_rng(seed)
    ws = [torch.from_numpy(rng.dirichlet(np.ones(N))).float().to(device) for _ in range(n_pool)]
    rob = {}

    for D in configs:
        tag = deg_tag(D)
        if not D:
            pool = [{k: v.detach().clone()
                     for k, v in flat_to_params(bundle['interp'], w @ vflat).items()}
                    for w in ws]
        else:
            bp, dg, a2k, abd = _cifar_bumps(D, N, vms, train_loader, test_loader,
                                            task_split, refresh=refresh)
            pool = build_simplex_pool(vms, bp, a2k, abd, dg, ws, dummy)
        inp, _ = noisy_input_sweep(pool, dummy, test_loader, all_classes, EPSILONS,
                                   n_trials=n_trials, seed=seed, task_split=task_split)
        wgt, _ = noisy_weight_sweep(pool, dummy, test_loader, all_classes, EPSILONS,
                                    n_trials=n_trials, seed=seed, task_split=task_split)
        rob[tag] = {
            'input':  {'logit': list(inp['logit_avg']['all']['nll']),
                       'softmax': list(inp['softmax_avg']['all']['nll'])},
            'weight': {'logit': list(wgt['logit_avg']['all']['nll']),
                       'softmax': list(wgt['softmax_avg']['all']['nll'])},
        }
        print(f"[rob] {deg_label(D):16s}  input/softmax "
              f"{rob[tag]['input']['softmax'][0]:.3f}->{rob[tag]['input']['softmax'][-1]:.3f}  "
              f"weight/softmax "
              f"{rob[tag]['weight']['softmax'][0]:.3f}->{rob[tag]['weight']['softmax'][-1]:.3f}")
        del pool

    pk = OUT / f'{exp}.pkl'
    data = pickle.load(pk.open('rb'))
    if 'degsweep' not in data:
        raise RuntimeError(f'{exp}: panels A/B must be computed before C-F')
    data['degsweep']['rob'] = rob
    data['degsweep']['rob_eps'] = EPSILONS.tolist()
    data['degsweep']['rob_npool'] = n_pool
    with pk.open('wb') as f:
        pickle.dump(data, f)
    print(f"[rob] cached -> {pk}")


def _store(exp, degsweep):
    # overwrite the whole degsweep block
    pk = OUT / f'{exp}.pkl'
    data = pickle.load(pk.open('rb')) if pk.exists() else {}
    data['degsweep'] = degsweep
    with pk.open('wb') as f:
        pickle.dump(data, f)


# plotting

def _palette(configs):
    """monochrome purple ramp, one shade per degree set: darker = larger sum(D)."""
    import matplotlib.pyplot as plt
    sums = [sum(D) for D in configs]
    smax = max(sums) or 1
    return {deg_tag(D): plt.cm.Purples(0.45 + 0.5 * (s / smax))
            for D, s in zip(configs, sums)}


def _end_labels(ax, ends, ymax=None):
    """tag each line at its right end, dodging vertically. ends is a list of
    (y_end, text, color). if ymax is given, cap the axis and annotate off-scale
    lines with their true endpoint."""
    y0, y1 = ax.get_ylim()
    if ymax is not None:
        y1 = ymax
        ax.set_ylim(y0, ymax)
    gap = (y1 - y0) * 0.058
    items = [(min(yv, y1), txt + (f'  $\\uparrow${yv:.0f}' if yv > y1 + 1e-6 else ''), col)
             for yv, txt, col in ends]
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    dod = [items[i][0] for i in order]
    for k in range(1, len(dod)):
        if dod[k] - dod[k - 1] < gap:
            dod[k] = dod[k - 1] + gap
    if ymax is None:
        ax.set_ylim(min(y0, dod[0] - 0.4 * gap), max(y1, dod[-1] + 0.4 * gap))
    trans = ax.get_yaxis_transform()
    for i, yy in zip(order, dod):
        yv, txt, col = items[i]
        ax.plot([1.0, 1.03], [yv, yy], color=col, lw=0.7, transform=trans,
                clip_on=False, zorder=2)
        ax.text(1.045, yy, txt, transform=trans, va='center', ha='left',
                fontsize=7.3, color=col, clip_on=False)


def _panel_nll(ax, d):
    """(A) per-degree-set distribution of per-model multitask NLL over uniform-
    dirichlet manifold samples (boxplot)."""
    configs = [tuple(D) for D in d['configs']]
    ypos = np.arange(len(configs))[::-1]
    data = [np.asarray(d['nll'][deg_tag(D)]) for D in configs]
    ax.boxplot(data, positions=ypos, vert=False, widths=0.6, patch_artist=True,
               whis=(0, 100), showfliers=False,
               medianprops=dict(color=PURPLE_D, lw=1.5),
               whiskerprops=dict(color=PURPLE_D, lw=1.0),
               capprops=dict(color=PURPLE_D, lw=1.0),
               boxprops=dict(facecolor=PURPLE, edgecolor=PURPLE_D, alpha=0.55, lw=1.0))
    ax.set_yticks(ypos); ax.set_yticklabels([deg_label(D) for D in configs], fontsize=10)
    ax.set_ylim(-0.6, len(data) - 0.4)
    allv = np.concatenate(data)
    ax.set_xlim(allv.min() * 0.92, np.percentile(allv, 97))   # clip catastrophic tails
    ax.set_xlabel('multitask NLL  (models sampled uniformly over the Nimbus manifold)',
                  fontsize=10.5)
    ax.set_title(r'(A)  NLL from uniform sampling over Nimbus, by degree set $\mathcal{D}$',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def _panel_ens(ax, d):
    """(B) logit- and softmax-ensemble NLL per degree set (dumbbell)."""
    from matplotlib.lines import Line2D
    if 'ens' not in d:
        ax.axis('off'); return
    configs = [tuple(D) for D in d['configs']]
    ypos = np.arange(len(configs))[::-1]
    for D, y in zip(configs, ypos):
        e = d['ens'][deg_tag(D)]
        ax.plot([e['logit'], e['softmax']], [y, y], color='0.78', lw=1.2, zorder=2)
        ax.scatter([e['logit']], [y], s=85, color=PURPLE, marker='o',
                   edgecolor=PURPLE_D, lw=1.1, zorder=4)
        ax.scatter([e['softmax']], [y], s=150, color=PURPLE, marker='*',
                   edgecolor=PURPLE_D, lw=1.1, zorder=4)
    ax.set_yticks(ypos); ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(configs) - 0.4)
    ax.set_xlabel('ensemble multitask NLL  (over the same manifold pool)', fontsize=10.5)
    ax.set_title(r'(B)  ensemble NLL by degree set $\mathcal{D}$',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=PURPLE, marker='o', lw=0, markersize=8,
               markeredgecolor=PURPLE_D, label='logit ensemble'),
        Line2D([], [], color=PURPLE, marker='*', lw=0, markersize=13,
               markeredgecolor=PURPLE_D, label='softmax ensemble'),
    ], frameon=False, fontsize=8.5, loc='lower right')


def _panel_rob(ax, d, exp, noise, mode, letter, cols):
    """(C-F) NLL of the `mode` ensemble vs `noise`-noise eps, one line per
    degree set, each tagged with its D at the right end."""
    if 'rob' not in d:
        ax.axis('off')
        ax.text(0.5, 0.5, f'robustness not computed yet\nrun: python -m misc.sweep --exp {exp}',
                ha='center', va='center', fontsize=10, color='0.5',
                style='italic', transform=ax.transAxes)
        return
    eps = np.array(d['rob_eps'])
    configs = [tuple(D) for D in d['configs']]
    ends = []
    for D in configs:
        y = np.array(d['rob'][deg_tag(D)][noise][mode])
        c = cols[deg_tag(D)]
        ax.plot(eps, y, color=c, lw=1.6, zorder=3)
        ends.append((float(y[-1]), deg_label(D), c))
    titles = {'logit': 'logit-ens', 'softmax': 'softmax-ens'}
    ax.set_xlabel(rf'{noise}-noise  $\varepsilon$', fontsize=10.5)
    ax.set_ylabel('ensemble multitask NLL', fontsize=10.5)
    ax.set_title(f'({letter})  {titles[mode]} robustness to {noise} noise',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _end_labels(ax, ends, ymax=ROB_YMAX.get((exp, noise, mode)))


def draw_sweep(exp, plt):
    title = exp.replace('cifar', 'CIFAR N=')
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 2, wspace=0.34, hspace=0.46,
                          left=0.10, right=0.95, top=0.95, bottom=0.06,
                          height_ratios=[1.0, 0.95, 0.95])
    d = pickle.load((OUT / f'{exp}.pkl').open('rb')).get('degsweep') \
        if (OUT / f'{exp}.pkl').exists() else None
    if d is None:
        ax = fig.add_subplot(gs[:, :]); ax.axis('off')
        ax.text(0.5, 0.5, f'{exp}: degree-set sweep not computed yet\n'
                f'run:  python -m misc.sweep --exp {exp}',
                ha='center', va='center', fontsize=12, color='0.5', style='italic')
    else:
        cols = _palette([tuple(D) for D in d['configs']])
        _panel_nll(fig.add_subplot(gs[0, 0]), d)
        _panel_ens(fig.add_subplot(gs[0, 1]), d)
        _panel_rob(fig.add_subplot(gs[1, 0]), d, exp, 'input', 'logit', 'C', cols)
        _panel_rob(fig.add_subplot(gs[1, 1]), d, exp, 'weight', 'logit', 'D', cols)
        _panel_rob(fig.add_subplot(gs[2, 0]), d, exp, 'input', 'softmax', 'E', cols)
        _panel_rob(fig.add_subplot(gs[2, 1]), d, exp, 'weight', 'softmax', 'F', cols)
    fig.suptitle(f'Degree-set sweep -- {title}', fontsize=13, fontweight='bold')
    fig.savefig(OUT / f'sweep_{exp}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / f'sweep_{exp}.pdf')


# markdown table dump (text mirror of the sweep_<exp>.pdf panels A-F)
KEEP_EPS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def _fmt(x):
    """round to <=2 decimals, dropping trailing zeros but keeping one (e.g. 1.0, 0.92)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _row_label(D):
    """degree-set row label; the linear control reads 'linear interpolation'."""
    return "linear interpolation" if not D else deg_label(D)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def dump_tables(exp):
    """write sweep_<exp>.txt: markdown mirror of the sweep figure panels (A-F)."""
    pkl = OUT / f'{exp}.pkl'
    d = pickle.load(pkl.open('rb')).get('degsweep') if pkl.exists() else None
    if d is None:
        print(f'skip {exp}: no degsweep in {pkl}')
        return
    configs = [tuple(D) for D in d['configs']]
    eps = np.asarray(d['rob_eps'])
    keep = [i for i, e in enumerate(eps) if any(abs(e - k) < 1e-6 for k in KEEP_EPS)]

    # (A) per-model NLL distribution over the manifold pool: Q1 / median / Q3 / mean.
    rows_a = []
    for D in configs:
        v = np.asarray(d['nll'][deg_tag(D)])
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows_a.append([_row_label(D), _fmt(q1), _fmt(med), _fmt(q3), _fmt(v.mean())])
    tbl_a = _md_table(["degree set $\\mathcal{D}$", "Q1", "median", "Q3", "mean"], rows_a)

    # (B) ensemble NLL (logit / softmax) over the same pool.
    rows_b = []
    for D in configs:
        e = d['ens'][deg_tag(D)]
        rows_b.append([_row_label(D), _fmt(e['logit']), _fmt(e['softmax'])])
    tbl_b = _md_table(["degree set $\\mathcal{D}$", "logit ensemble", "softmax ensemble"], rows_b)

    # robustness sweeps: logit/softmax folded into one table per noise, rows =
    # eps x mode (eps blank on the softmax row), columns = degree set.
    rob_modes = [("logit", "Logit"), ("softmax", "Softmax")]

    def rob_table(noise):
        header = ["$\\varepsilon$", "mode"] + [_row_label(D) for D in configs]
        rows = []
        for i in keep:
            for j, (mode_key, mode_disp) in enumerate(rob_modes):
                ecell = _fmt(eps[i]) if j == 0 else ""
                rows.append([ecell, mode_disp] +
                            [_fmt(d['rob'][deg_tag(D)][noise][mode_key][i]) for D in configs])
        return _md_table(header, rows)

    S = [
        "**NLL from uniform sampling over Nimbus, by degree set**:\n\n" + tbl_a,
        "**Ensemble NLL by degree set**:\n\n" + tbl_b,
        "**Robustness to input noise**:\n\n" + rob_table("input"),
        "**Robustness to weight noise**:\n\n" + rob_table("weight"),
    ]
    out = OUT / f'sweep_{exp}.txt'
    out.write_text("\n\n".join(S) + "\n")
    print('wrote', out)


def _has(exp, key):
    """true if the {exp}.pkl degsweep block already holds `key` (e.g. 'nll', 'rob')."""
    pk = OUT / f'{exp}.pkl'
    if not pk.exists():
        return False
    d = pickle.load(pk.open('rb')).get('degsweep')
    return d is not None and key in d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=EXPS, default=None)
    ap.add_argument('--all', action='store_true', help='apply to every experiment')
    ap.add_argument('--n-deg', type=int, default=256)
    ap.add_argument('--n-rob-pool', type=int, default=32)
    ap.add_argument('--refresh', action='store_true',
                    help='ignore cached results and bumps, recompute everything')
    args = ap.parse_args()

    exps = list(EXPS) if args.all else ([args.exp] if args.exp else [])
    if not exps:
        ap.error('pass --exp <name> or --all')

    plt = setup_mpl()
    for exp in exps:
        if args.refresh or not _has(exp, 'nll'):
            recompute_degsweep(exp, n=args.n_deg, refresh=args.refresh)
        if args.refresh or not _has(exp, 'rob'):
            recompute_degsweep_rob(exp, n_pool=args.n_rob_pool, refresh=args.refresh)
        dump_tables(exp)
        draw_sweep(exp, plt)


if __name__ == '__main__':
    main()
