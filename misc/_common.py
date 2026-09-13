"""shared loaders, colors, and helpers for the misc CIFAR figures."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from nimbus.simplex import (
    SimplexInterpolator, build_bump_params, bp_from_cpu, dg_from_cpu,
)
from nimbus.metrics import metrics_from_logits
from nimbus.seeds import device

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# Nimbus purple, linear coral, each with a darker shade for edges and accents
PURPLE, PURPLE_D = '#B39DDB', '#6A1B9A'
CORAL, CORAL_D = '#F4A582', '#C0623A'


def deg_tag(D):
    """filename/key-safe tag for a degree set, e.g. (2,6,7) -> '2-6-7'."""
    if D:
        return '-'.join(map(str, D))
    return 'linear'


def deg_label(D):
    """human label, e.g. (2,6,7) -> '{2,6,7}', () -> 'linear (none)'."""
    if D:
        return '{' + ','.join(map(str, D)) + '}'
    return 'linear (none)'


def _trainable_vp(models):
    return [{n: p.detach().clone() for n, p in m.named_parameters()
             if p.requires_grad}
            for m in models]


def _freeze(models):
    for m in models:
        for p in m.parameters():
            p.requires_grad_(False)


def flat_to_params(interp, flat):
    """unflatten a trainable-param vector into a functional_call dict of
    trainable params plus the interpolator's fixed buffers."""
    pm, off = {}, 0
    for name, shape, sz in zip(interp.param_names, interp.param_shapes,
                               interp.param_sizes):
        pm[name] = flat[off:off + sz].view(shape)
        off += sz
    return {**pm, **interp.fixed_sd}


def nimbus_flat(interp, w):
    pm = interp(w)
    return torch.cat([pm[n].reshape(-1) for n in interp.param_names])


def mean_pairwise_dist(vertex_flat):
    N = vertex_flat.shape[0]
    ds = [torch.norm(vertex_flat[i] - vertex_flat[j]).item()
          for i, j in combinations(range(N), 2)]
    return float(np.mean(ds))


def setup_mpl():
    from matplotlib import pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return plt


def load_cifar(N):
    from experiments.cifar.config import (
        EXPERIMENT_CONFIGS, GLOBAL_SEED, RESULTS_DIR)
    from experiments.cifar.data import CIFAR100_CL, CL_DataLoader
    from experiments.cifar.model import make_convnext, load_state

    degrees = dict((k, d) for k, d, _ in EXPERIMENT_CONFIGS)[N]
    ckpt_dir = RESULTS_DIR / f'N={N}' / 'ckpts'

    all_tasks = torch.randperm(
        N * 10, generator=torch.Generator().manual_seed(GLOBAL_SEED))
    task_split = [all_tasks[i * 10:(i + 1) * 10] for i in range(N)]
    all_classes = [c.item() for t in task_split for c in t]
    num_classes = max(all_classes) + 1

    sds = torch.load(ckpt_dir / 'vertex.pt', map_location=device)
    vertex_models = []
    for vi in range(N):
        m = make_convnext(seed=vi + 1, num_classes=num_classes)
        load_state(m, sds[vi])
        vertex_models.append(m.to(device).eval())
    vp = _trainable_vp(vertex_models)
    _freeze(vertex_models)

    post = torch.load(ckpt_dir / 'simplex_post.pt', map_location=device)
    bp, dg = bp_from_cpu(post['bp']), dg_from_cpu(post['dg'])
    for pd in bp.values():
        for p in pd.values():
            p.requires_grad_(False)
    for g in dg.values():
        g.requires_grad_(False)

    _, a2k, abd = build_bump_params(vertex_models, degrees)
    dummy = deepcopy(vertex_models[0]).to(device).eval()
    _freeze([dummy])
    interp = SimplexInterpolator(vp, bp, a2k, abd, dummy, degree_gates=dg)
    interp.cache_bump_flat()

    dl = CL_DataLoader(CIFAR100_CL(split='test', seed=GLOBAL_SEED),
                       batch_size=128, shuffle=True, seed=GLOBAL_SEED)
    dl.set_epoch_data(all_classes)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    x_all, y_all = torch.cat(xs).to(device), torch.cat(ys).to(device)

    @torch.no_grad()
    def logits_fn(params, bs=512):
        return torch.cat([functional_call(dummy, params, (x_all[i:i + bs],))
                          for i in range(0, len(x_all), bs)], dim=0)

    @torch.no_grad()
    def loss_fn(params):
        return metrics_from_logits(logits_fn(params), y_all)['nll']

    return dict(kind='cifar', N=N, dummy=dummy, interp=interp,
                vertex_flat=interp.vertex_flat, loss_fn=loss_fn,
                logits_fn=logits_fn, y_all=y_all)
