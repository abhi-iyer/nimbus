"""per-task + all-scope evaluation for the ViT leg.

streams over the pool (members materialized one at a time), decodes each task's
eval batches once (prepare_eval_batches) and reuses them across every member /
eps / trial, and reports the full nimbus.metrics set (acc, top3, top5, nll, ece)
via its streaming accumulators.

ensemble modes mirror nimbus.robustness.ENS_MODES; tasks have disjoint label
spaces so everything routes per task with that task's head.
"""
import numpy as np
import torch
import torch.nn.functional as F

from nimbus.seeds import device
from nimbus.metrics import (
    METRICS, new_metric_state, accumulate_into, accumulate_from_probs_into,
    finalize_metrics,
)
from nimbus.robustness import perturb_weight_vector

from .model import encode, head_logits, base_projection_scale


ENS_MODES = ("logit_avg", "softmax_avg", "weight_avg", "single")


@torch.no_grad()
def prepare_eval_batches(task_loaders, task_names, max_batches):
    """
    decode each task's first max_batches test batches once (kept on cpu, moved
    to gpu per use).

    returns {ti: [(x_cpu, y_cpu), ...]}.
    """
    out = {}
    for ti, _ in enumerate(task_names):
        batches = []
        for bi, (x, y) in enumerate(task_loaders[ti]["test"]):
            if bi >= max_batches:
                break
            batches.append((x.clone(), y.clone()))
        out[ti] = batches
    return out


@torch.no_grad()
def _perturb_input(x, eps, gen):
    if eps <= 0:
        return x
    noise = torch.randn(x.shape, device=x.device, generator=gen)
    fx, fn = x.reshape(x.size(0), -1), noise.reshape(x.size(0), -1)
    scale = eps * fx.norm(dim=1, keepdim=True) / fn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return x + (fn * scale).reshape_as(x)


@torch.no_grad()
def _perturb_params(pd, eps, gen):
    """
    weight noise via the shared nimbus.robustness: flat relative-L2 over the
    concatenated float weights, noise rescaled to eps * ||w|| across the whole
    vector.
    """
    if eps <= 0:
        return pd
    names = [k for k, v in pd.items() if v.is_floating_point()]
    flat = torch.cat([pd[k].reshape(-1) for k in names])
    flat = perturb_weight_vector(flat, eps, generator=gen)
    out = dict(pd)
    off = 0
    for k in names:
        n = pd[k].numel()
        out[k] = flat[off:off + n].view_as(pd[k])
        off += n
    return out


def single_member(param_dict):
    """accessor for a one-member pool (baselines / merged models)."""
    return (lambda i: param_dict), 1


@torch.no_grad()
def eval_pool(get_member, n_members, dummy, eval_batches, heads, task_names,
              input_eps=0.0, weight_eps=0.0, max_batches=8, seed=0, log=None):
    """
    stream members over pre-decoded batches.

    returns {mode: {scope: {acc, top3, top5, nll, ece}}} for scope in
    task_names + ["all"].
    """
    proj, scale = base_projection_scale()
    acc = {}                       # (ti,bi) -> dict(logit_sum, soft_sum, y, single)
    wsum = None                    # streamed parameter sum (for weight_avg)

    for mi in range(n_members):
        cp = get_member(mi)
        # weight_avg is the clean parameter average, perturbed once below; each
        # member is perturbed independently.
        wsum = ({k: v.clone() for k, v in cp.items()} if wsum is None
                else {k: wsum[k] + cp[k] for k in wsum})
        if weight_eps > 0:
            g = torch.Generator(device=device); g.manual_seed(seed + 7919 * mi)
            cp = _perturb_params(cp, weight_eps, g)
        for ti, _ in enumerate(task_names):
            for bi, (xc, yc) in enumerate(eval_batches[ti]):
                if bi >= max_batches:
                    break
                x, y = xc.to(device), yc.to(device)
                if input_eps > 0:
                    g = torch.Generator(device=device); g.manual_seed(seed + ti * 131 + bi)
                    x = _perturb_input(x, input_eps, g)
                lg = head_logits(encode(dummy, cp, x), heads[ti], proj, scale)
                sm = F.softmax(lg, dim=-1)
                key = (ti, bi)
                if key not in acc:
                    acc[key] = {"logit_sum": lg.clone(), "soft_sum": sm,
                                "y": y, "single": lg.clone() if mi == 0 else None}
                else:
                    acc[key]["logit_sum"] += lg
                    acc[key]["soft_sum"] += sm
        del cp
        if log and n_members > 1 and (mi + 1) % max(1, n_members // 4) == 0:
            log(f"    eval member {mi+1}/{n_members}")

    wavg = {k: v / n_members for k, v in wsum.items()}
    if weight_eps > 0:                 # one noise vector on the clean average,
        g = torch.Generator(device=device); g.manual_seed(seed + 104729)
        wavg = _perturb_params(wavg, weight_eps, g)

    scopes = list(task_names) + ["all"]
    states = {m: {s: new_metric_state(device) for s in scopes} for m in ENS_MODES}

    def feed_logits(mode, tname, logits, y):
        accumulate_into(logits, y, states[mode][tname])
        accumulate_into(logits, y, states[mode]["all"])

    def feed_probs(mode, tname, probs, y):
        accumulate_from_probs_into(probs, y, states[mode][tname])
        accumulate_from_probs_into(probs, y, states[mode]["all"])

    for (ti, bi), a in acc.items():
        tname = task_names[ti]; y = a["y"]
        feed_logits("logit_avg", tname, a["logit_sum"] / n_members, y)
        feed_probs("softmax_avg", tname, (a["soft_sum"] / n_members).clamp(min=1e-12), y)
        feed_logits("single", tname, a["single"], y)

    # weight_avg: one forward through the parameter-average
    for ti, _ in enumerate(task_names):
        tname = task_names[ti]
        for bi, (xc, yc) in enumerate(eval_batches[ti]):
            if bi >= max_batches:
                break
            x, y = xc.to(device), yc.to(device)
            if input_eps > 0:
                g = torch.Generator(device=device); g.manual_seed(seed + ti * 131 + bi)
                x = _perturb_input(x, input_eps, g)
            feed_logits("weight_avg", tname, head_logits(encode(dummy, wavg, x), heads[ti], proj, scale), y)

    return {m: {s: finalize_metrics(states[m][s]) for s in scopes} for m in ENS_MODES}


@torch.no_grad()
def noise_sweep(get_member, n_members, dummy, eval_batches, heads, task_names,
                epsilons, kind, n_trials=3, seed=42, max_batches=6, log=None):
    """
    all-scope metrics (mean/std over trials) across epsilons for input/weight
    noise.

    returns {mode: {"eps": [...], "<metric>_mean": [...], "<metric>_std": [...]}}.
    """
    assert kind in ("input", "weight")
    res = {m: {"eps": list(map(float, epsilons)),
               **{f"{met}_mean": [] for met in METRICS},
               **{f"{met}_std": [] for met in METRICS}} for m in ENS_MODES}
    for ei, eps in enumerate(epsilons):
        trials = {m: {met: [] for met in METRICS} for m in ENS_MODES}
        for tr in range(n_trials):
            kw = {"input_eps": float(eps)} if kind == "input" else {"weight_eps": float(eps)}
            out = eval_pool(get_member, n_members, dummy, eval_batches, heads, task_names,
                            max_batches=max_batches, seed=seed + tr * 9973 + int(eps * 1e6), **kw)
            for m in ENS_MODES:
                for met in METRICS:
                    trials[m][met].append(out[m]["all"][met])
        for m in ENS_MODES:
            for met in METRICS:
                a = np.array(trials[m][met])
                res[m][f"{met}_mean"].append(float(a.mean())); res[m][f"{met}_std"].append(float(a.std()))
        if log:
            log(f"    {kind}-noise eps {ei+1}/{len(epsilons)} (={float(eps):.2f}) "
                f"logit nll={res['logit_avg']['nll_mean'][-1]:.3f} acc={res['logit_avg']['acc_mean'][-1]:.3f}")
    return res
