"""sampling and pooling of models drawn from the simplex manifold."""

from itertools import combinations

import numpy as np
import torch

from .simplex import SimplexInterpolator, sample_w_for_degree


def sample_face_ws(N, degrees, n_total, seed=42, device=None):
    """
    sample n_total simplex weights spread across the faces of the given degrees.

    allocates slots per face so each degree carries equal total weight, then
    draws one weight per slot via sample_w_for_degree.
    """
    rng = np.random.default_rng(seed)
    all_faces = [(r, face) for r in sorted(degrees) for face in combinations(range(N), r)]
    n_faces = len(all_faces)
    cnt = {r: sum(1 for rr, _ in all_faces if rr == r) for r in degrees}
    raw = np.array([1.0 / cnt[r] for r, _ in all_faces])
    raw = raw / raw.sum()
    alloc = np.ones(n_faces, dtype=int)
    remaining = n_total - n_faces
    if remaining > 0:
        extra = (raw * remaining).astype(int)
        leftover = remaining - extra.sum()
        for i in np.argsort(-raw)[:leftover]:
            extra[i] += 1
        alloc += extra
    ws = []
    for idx, (r, face) in enumerate(all_faces):
        for _ in range(alloc[idx]):
            w, _ = sample_w_for_degree(N, r, rng=rng, device=device, face=face)
            ws.append(w)
    return ws


def weight_average_params(param_dicts):
    """average a list of param dicts (float tensors averaged, others cloned)."""
    avg = {}
    N = len(param_dicts)
    for k in param_dicts[0]:
        v = param_dicts[0][k]
        if v.is_floating_point():
            avg[k] = sum(pd[k] for pd in param_dicts) / N
        else:
            avg[k] = v.clone()
    return avg


def flatten_trainable(sd, trainable_names):
    """concatenate the named params of sd into one flat vector."""
    return torch.cat([sd[n].reshape(-1) for n in trainable_names])


def unflatten_trainable(flat, sd, trainable_names, shapes, sizes):
    """write a flat vector back into a copy of sd along trainable_names."""
    new_sd = dict(sd)
    offset = 0
    for n, shape, sz in zip(trainable_names, shapes, sizes):
        new_sd[n] = flat[offset:offset + sz].view(shape)
        offset += sz
    return new_sd


def build_simplex_pool(vertex_models, bp, a2k, abd, dg, ws, dummy):
    """materialize a pool of param dicts by evaluating the interpolator at each w."""
    vp = [{n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
          for vm in vertex_models]
    interp = SimplexInterpolator(vp, bp, a2k, abd, dummy, degree_gates=dg)
    interp.cache_bump_flat()
    return [{k: v.detach().clone() for k, v in interp(w).items()} for w in ws]


def canonical_pd(sd, trainable_names):
    """
    drop non-trainable keys tied (shared storage) to a trainable param, matching
    the interpolator's fixed-state handling.
    """
    trainable_set = set(trainable_names)
    ptr_to_keys = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.numel() > 0:
            ptr_to_keys.setdefault(v.data_ptr(), []).append(k)
    drop = set()
    for keys in ptr_to_keys.values():
        if len(keys) > 1 and any(k in trainable_set for k in keys):
            for k in keys:
                if k not in trainable_set:
                    drop.add(k)
    return {k: v for k, v in sd.items() if k not in drop}
