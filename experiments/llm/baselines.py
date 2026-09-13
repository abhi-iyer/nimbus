"""
baselines for the LLM leg (same roster as experiments/vit).

  * baseline ensemble: M = |D| independently trained NanoGPTs, each at Nimbus
    phase-1 sample-passes / |D| (config.baseline_ens_steps).
  * |D|+N ensemble: the |D| members plus N more at the same per-member budget,
    reported as an upper reference.
  * TIES / Fisher: parameter merges of the vertices.

Nimbus and the linear-interpolation baseline live in run.py. ensemble members
train at a constant lr == SIMPLEX_LR with AdamW(0.9, 0.95). the task-vector
merges use the mean of the vertices as base; Task Arithmetic is omitted.
"""
from copy import deepcopy

import torch
from torch.func import functional_call
from tqdm import tqdm

from nimbus.ensemble import (
    weight_average_params, flatten_trainable, unflatten_trainable,
)
from nimbus.metrics import METRICS, scope_keys
from nimbus.robustness import (
    ENS_MODES, perturb_weight_vector, sweep_one_pass,
    empty_sweep_results, finalize_sweep, merge_trial_stats,
)
from nimbus.seeds import device, set_seed

from .config import (
    N, TASK_NAMES, CKPT_DIR, BATCH_SIZE,
    N_TRIALS, MAX_BATCHES, GLOBAL_SEED, SIMPLEX_LR,
    FISHER_BATCHES, TIES_KEEP_FRACTION,
)
from .data import (
    CyclingLoader, make_loader, mixed_val_with_tasks, val_batches,
)
from .model import NanoGPT
from .perturbations import corrupt_tokens
from .train import eval_metrics, stage_seed


DEVICE = device


def llm_forward(dummy, pd, x, **kwargs):
    return functional_call(dummy, pd, (x, None), kwargs=kwargs)[0]


def _interleaved_train_loader(train_arrays, block_size, batch_size, seed):
    loaders = [
        CyclingLoader(make_loader(train_arrays[n], block_size, batch_size,
                                  shuffle=True, seed=seed + i + 7000))
        for i, n in enumerate(TASK_NAMES)
    ]
    i = 0
    while True:
        x, y = next(loaders[i % len(loaders)])
        i += 1
        yield x, y


# --- baseline ensemble: |D| param- and compute-matched members ---------------

def train_baseline_ensemble(train_arrays, val_arrays, mcfg, n_models, n_steps,
                            stage_name='baseline', ckpt_prefix="baseline",
                            refresh=False):
    """
    `n_models` independent multitask NanoGPTs, `n_steps` each at constant LR.

    members differ only in seed (init + data order). cached per member as
    {ckpt_prefix}_{i}.pt.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    tokens, task_ids = mixed_val_with_tasks(val_arrays, mcfg.block_size)
    for ai in range(n_models):
        ckpt = CKPT_DIR / f"{ckpt_prefix}_{ai}.pt"
        s = stage_seed(stage_name, ai)
        # seed before constructing the model so each member's init is distinct.
        set_seed(s)
        m = NanoGPT(mcfg).to(DEVICE)
        if ckpt.exists() and not refresh:
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m.eval()
            print(f"[{ckpt_prefix} {ai}] loaded {ckpt}")
            models.append(m)
            continue
        loader = _interleaved_train_loader(train_arrays, mcfg.block_size, BATCH_SIZE, seed=s)
        # constant SIMPLEX_LR, no warmup/decay.
        opt = torch.optim.AdamW(m.parameters(), lr=SIMPLEX_LR,
                                betas=(0.9, 0.95), weight_decay=0.0)
        m.train()
        pbar = tqdm(range(n_steps), desc=f"{ckpt_prefix} {ai}")
        loss_ema = None
        for it in pbar:
            x, y = next(loader)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = m(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            l = loss.item()
            loss_ema = l if loss_ema is None else 0.95 * loss_ema + 0.05 * l
            if it % 100 == 0:
                pbar.set_postfix(loss=f"{loss_ema:.3f}")
        torch.save(m.state_dict(), ckpt)
        vm = eval_metrics(m, tokens, mcfg.block_size, BATCH_SIZE, task_ids=task_ids)
        print(f"[{ckpt_prefix} {ai}] mixed val all-scope nll={vm['all']['nll']:.3f} "
              f"acc={vm['all']['acc']:.3f}")
        m.eval()
        models.append(m)
    return models


# --- parameter merges of the vertices ---------------------------------------

def _float_sds(vertex_models):
    return [{n: p.detach().float() for n, p in vm.state_dict().items()
             if p.is_floating_point()} for vm in vertex_models]


def _mean_base(sds):
    """the vertex mean, used as the merge base."""
    return {n: sum(sd[n] for sd in sds) / len(sds) for n in sds[0]}


def _load_merged(template, merged):
    """load a merged float state dict into a copy of `template`, preserving dtypes."""
    m = deepcopy(template).to(DEVICE).eval()
    full = m.state_dict()
    for n in full:
        if n in merged:
            full[n] = merged[n].to(full[n].dtype)
    m.load_state_dict(full, strict=True)
    return m


def ties_merge(vertex_models, keep_fraction=TIES_KEEP_FRACTION):
    """TIES (trim -> elect sign -> disjoint mean) against the vertex mean."""
    sds = _float_sds(vertex_models)
    base = _mean_base(sds)
    N_ = len(vertex_models)
    merged = {}
    for n, b in base.items():
        tvecs = torch.stack([sd[n] - b for sd in sds], dim=0)          # [N, *shape]
        flat = tvecs.reshape(N_, -1)
        thresh = torch.quantile(flat.abs(), 1.0 - keep_fraction, dim=1)
        for _ in range(tvecs.dim() - 1):
            thresh = thresh.unsqueeze(-1)
        trimmed = tvecs * (tvecs.abs() >= thresh).float()
        elected = trimmed.sign().sum(dim=0).sign()
        elected[elected == 0] = 1.0
        agree = (trimmed.sign() == elected.unsqueeze(0)) | (trimmed == 0)
        disjoint = trimmed * agree.float()
        counts = (disjoint != 0).float().sum(dim=0).clamp(min=1)
        merged[n] = b + disjoint.sum(dim=0) / counts
    return _load_merged(vertex_models[0], merged)


def compute_diagonal_fisher(vertex, train_array, block_size, n_batches=FISHER_BATCHES,
                            seed=GLOBAL_SEED):
    """diagonal Fisher of a vertex on its own task, from per-token LM CE grads."""
    m = deepcopy(vertex).to(DEVICE).train()
    fisher = {n: torch.zeros_like(p) for n, p in m.named_parameters() if p.requires_grad}
    loader = CyclingLoader(make_loader(train_array, block_size, BATCH_SIZE,
                                       shuffle=True, seed=seed))
    for _ in range(n_batches):
        x, y = next(loader)
        m.zero_grad(set_to_none=True)
        _, loss = m(x.to(DEVICE), y.to(DEVICE))
        loss.backward()
        for name, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach().pow(2)
    for name in fisher:
        fisher[name] /= max(n_batches, 1)
    del m
    return fisher


def fisher_merge(vertex_models, train_arrays, block_size, n_batches=FISHER_BATCHES):
    """diagonal-Fisher-weighted average of the vertices (no base needed)."""
    print("[fisher] computing diagonal Fishers...")
    fishers = [compute_diagonal_fisher(vertex_models[i], train_arrays[TASK_NAMES[i]],
                                       block_size, n_batches)
               for i in range(len(vertex_models))]
    sds = [vm.state_dict() for vm in vertex_models]
    merged = {}
    for name, val in sds[0].items():
        if name in fishers[0] and val.is_floating_point():
            num = sum(fishers[i][name] * sds[i][name].detach()
                      for i in range(len(sds)))
            den = sum(fishers[i][name] for i in range(len(sds))).clamp(min=1e-8)
            merged[name] = num / den
        else:
            merged[name] = val.detach().clone()
    del fishers
    return _load_merged(vertex_models[0], merged)


# --- robustness sweeps ------------------------------------------------------

@torch.no_grad()
def noisy_input_sweep(param_dicts, dummy, mixed_val, task_ids, block_size, batch_size,
                      rates, vocab_size, n_trials=N_TRIALS, seed=GLOBAL_SEED,
                      max_batches=MAX_BATCHES):
    scopes = scope_keys(N)
    avg_params = weight_average_params(param_dicts)
    results_mean = empty_sweep_results(scopes)
    results_std = empty_sweep_results(scopes)
    for p in tqdm(rates, desc="token corruption"):
        trial = empty_sweep_results(scopes)
        for tr in range(n_trials):
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(seed + (hash((float(p), tr)) & 0xFFFFFFFF))
            def transform(x, y, _p=float(p), _gen=gen):
                return corrupt_tokens(x, _p, vocab_size, generator=_gen), y
            batches = val_batches(mixed_val, block_size, batch_size, max_batches,
                                   task_ids=task_ids)
            out = sweep_one_pass(
                param_dicts, avg_params, dummy, batches, llm_forward, scopes,
                batch_transform=transform,
            )
            for m in ENS_MODES:
                for s in scopes:
                    for met in METRICS:
                        trial[m][s][met].append(out[m][s][met])
        merge_trial_stats(results_mean, results_std, trial, scopes)
    return finalize_sweep(results_mean), finalize_sweep(results_std)


@torch.no_grad()
def noisy_weight_sweep(param_dicts, dummy, mixed_val, task_ids, block_size, batch_size,
                       epsilons, n_trials=N_TRIALS, seed=GLOBAL_SEED,
                       max_batches=MAX_BATCHES):
    scopes = scope_keys(N)
    trainable_names = [n for n, _ in dummy.named_parameters()]
    sd = dummy.state_dict()
    shapes = [sd[n].shape for n in trainable_names]
    sizes = [sd[n].numel() for n in trainable_names]
    avg_params = weight_average_params(param_dicts)
    results_mean = empty_sweep_results(scopes)
    results_std = empty_sweep_results(scopes)
    for eps in tqdm(epsilons, desc="weight noise"):
        trial = empty_sweep_results(scopes)
        for tr in range(n_trials):
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(seed + (hash((float(eps), tr)) & 0xFFFFFFFF))
            perturbed_pds = []
            for pd in param_dicts:
                flat = flatten_trainable(pd, trainable_names)
                pf = perturb_weight_vector(flat, float(eps), generator=gen)
                perturbed_pds.append(
                    unflatten_trainable(pf, pd, trainable_names, shapes, sizes)
                )
            avg_flat = flatten_trainable(avg_params, trainable_names)
            avg_pf = perturb_weight_vector(avg_flat, float(eps), generator=gen)
            perturbed_avg = unflatten_trainable(
                avg_pf, avg_params, trainable_names, shapes, sizes
            )
            batches = val_batches(mixed_val, block_size, batch_size, max_batches,
                                   task_ids=task_ids)
            out = sweep_one_pass(
                perturbed_pds, perturbed_avg, dummy, batches, llm_forward, scopes,
            )
            for m in ENS_MODES:
                for s in scopes:
                    for met in METRICS:
                        trial[m][s][met].append(out[m][s][met])
        merge_trial_stats(results_mean, results_std, trial, scopes)
    return finalize_sweep(results_mean), finalize_sweep(results_std)
