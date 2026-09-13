"""ensemble robustness sweeps over a pool of interpolated models."""

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import (
    METRICS, new_scoped_states, accumulate_scoped, finalize_scoped,
)
from .seeds import device


ENS_MODES = ("logit_avg", "softmax_avg", "weight_avg", "single")


@torch.no_grad()
def perturb_weight_vector(flat, epsilon, generator=None):
    """add gaussian noise to a flat vector, scaled to ||noise|| = epsilon * ||flat||."""
    if epsilon <= 0.0:
        return flat.clone()
    w_norm = flat.norm().clamp(min=1e-12)
    noise = torch.randn(flat.shape, device=flat.device, dtype=flat.dtype,
                        generator=generator)
    noise = noise * (epsilon * w_norm / noise.norm().clamp(min=1e-12))
    return flat + noise


def empty_sweep_results(scopes):
    """nested result dict: mode -> scope -> metric -> list."""
    return {m: {s: {met: [] for met in METRICS} for s in scopes} for m in ENS_MODES}


def finalize_sweep(results):
    """convert the result lists into numpy arrays."""
    return {m: {s: {met: np.array(v) for met, v in d.items()}
                for s, d in by_s.items()}
            for m, by_s in results.items()}


def _nanmean(vals):
    """mean over finite entries, nan if none are finite."""
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _nanstd(vals):
    """std over finite entries, nan if none are finite."""
    a = np.asarray(vals, dtype=float)
    return float(np.nanstd(a)) if np.isfinite(a).any() else float("nan")


def merge_trial(results, trial, scopes):
    """append each scope/metric's trial mean into results."""
    for m in ENS_MODES:
        for s in scopes:
            for met in METRICS:
                results[m][s][met].append(_nanmean(trial[m][s][met]))


def merge_trial_stats(mean_results, std_results, trial, scopes):
    """append each scope/metric's trial mean and std into the two result dicts."""
    for m in ENS_MODES:
        for s in scopes:
            for met in METRICS:
                vals = trial[m][s][met]
                mean_results[m][s][met].append(_nanmean(vals))
                std_results[m][s][met].append(_nanstd(vals))


@torch.no_grad()
def sweep_one_pass(pool_pds, weight_avg_pd, dummy, batches, forward_fn, scopes,
                   batch_transform=None,
                   forward_kwargs_fn=None,
                   per_member_kwargs_fn=None):
    """
    evaluate a pool of param dicts over `batches` under every ensemble mode.

    each pool member is run through forward_fn and accumulated into logit-average,
    softmax-average, and single-member metrics; weight_avg_pd gives the
    weight-average mode. metrics are collected per scope.

    forward_kwargs_fn and per_member_kwargs_fn supply extra forward kwargs
    (shared across members, or per member); pass at most one.
    """
    if forward_kwargs_fn is not None and per_member_kwargs_fn is not None:
        raise ValueError("pass forward_kwargs_fn or per_member_kwargs_fn, not both")
    st = {m: new_scoped_states(scopes, device) for m in ENS_MODES}
    any_b = False
    N = len(pool_pds)
    for x, y, t in batches:
        if batch_transform is not None:
            x, y = batch_transform(x, y)
        shared = forward_kwargs_fn(x, y) if forward_kwargs_fn is not None else {}
        logit_sum = None
        prob_sum  = None
        for i, pd in enumerate(pool_pds):
            kw = (per_member_kwargs_fn(x, y, i)
                  if per_member_kwargs_fn is not None else shared)
            logits = forward_fn(dummy, pd, x, **kw)
            # member 0 stands in for the single-model baseline
            if i == 0:
                accumulate_scoped(logits, y, t, st["single"])
            if logit_sum is None:
                logit_sum = logits.clone()
            else:
                logit_sum.add_(logits)
            probs = F.log_softmax(logits, dim=-1).exp()
            del logits
            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum.add_(probs)
                del probs
        accumulate_scoped(logit_sum / N, y, t, st["logit_avg"])
        accumulate_scoped(prob_sum / N, y, t, st["softmax_avg"], from_probs=True)
        del logit_sum, prob_sum
        wa_kw = (per_member_kwargs_fn(x, y, None)
                 if per_member_kwargs_fn is not None else shared)
        wa_logits = forward_fn(dummy, weight_avg_pd, x, **wa_kw)
        accumulate_scoped(wa_logits, y, t, st["weight_avg"])
        any_b = True
    return {m: finalize_scoped(st[m], any_b) for m in ENS_MODES}
