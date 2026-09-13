"""
orchestrate the LLM (NanoGPT) leg: train the N=7 vertices, fit Nimbus
(phase 1 bumps -> phase 2 gates), run the baselines, and evaluate clean and
under input/weight noise. writes results/*.txt for experiments/llm/figures.

only the post-calibration Nimbus pool is reported.

    python -m experiments.llm.run                # full run, reusing any cache
    python -m experiments.llm.run --data-only    # fetch + tokenize corpora, then stop
    python -m experiments.llm.run --refresh      # ignore caches and recompute everything
"""
import argparse
from copy import deepcopy

import numpy as np
import pickle

import torch

from nimbus.ensemble import (
    sample_face_ws, build_simplex_pool, canonical_pd,
)
from nimbus.metrics import METRICS, scope_keys
from nimbus.seeds import device, set_seed
from nimbus.simplex import (
    build_bump_params, clone_bp, clone_dg,
    pack_simplex_state, unpack_simplex_state, load_simplex_if_match,
)
from nimbus.tables import (
    write_summary_table, write_robustness_table, write_robustness_stats_table,
    with_mean, print_wide_table,
)

from .baselines import (
    train_baseline_ensemble, ties_merge, fisher_merge,
    noisy_input_sweep, noisy_weight_sweep,
)
from .config import (
    N, TASK_NAMES, TASK_DISPLAY, GLOBAL_SEED,
    BATCH_SIZE, MAX_BATCHES, N_BASELINE, N_BASELINE_LARGE, baseline_ens_steps,
    NIMBUS_PHASE1_PASSES, N_ENS, N_FACE_ENS, DEGREES,
    INPUT_RATES, WEIGHT_EPSILONS, DATA_DIR, CKPT_DIR, RESULTS_DIR,
)
from .data import prepare_data, split_train_val, mixed_val_with_tasks
from .model import NanoGPT, ModelConfig
from .train import (
    train_vertex_models, train_simplex, calibrate_simplex,
    eval_metrics, stage_seed,
)


DEVICE = device

# Nimbus rows: face-stratified pool for the logit/softmax ensembles, uniform
# (interior) pool for the parameter average.
METHOD_MAP = {
    "logit ens uniform degree":   ("face",     "logit_avg"),
    "softmax ens uniform degree": ("face",     "softmax_avg"),
    "param avg":                  ("interior", "weight_avg"),
}

BASELINE_LINES = [
    ("Linear logit ens",     "lin_interior",    "logit_avg"),
    ("Linear softmax ens",   "lin_interior",    "softmax_avg"),
    ("Linear param avg",     "lin_interior",    "weight_avg"),
    ("Baseline logit ens",   "baseline",        "logit_avg"),
    ("Baseline softmax ens", "baseline",        "softmax_avg"),
    ("Baseline param avg",   "baseline",        "weight_avg"),
    ("Baseline+N logit ens",   "baseline_dpn",  "logit_avg"),
    ("Baseline+N softmax ens", "baseline_dpn",  "softmax_avg"),
    ("Baseline+N param avg",   "baseline_dpn",  "weight_avg"),
    ("TIES merge",           "ties",   "single"),
    ("Fisher merge",         "fisher", "single"),
]


def run_main(data_only=False, refresh=False):
    set_seed(GLOBAL_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(f"N={N} tasks: {TASK_NAMES}")
    print(f"degrees={DEGREES}, phase-1 budget={NIMBUS_PHASE1_PASSES:,} sample-passes")

    tokenizer, token_arrays = prepare_data()
    if data_only:
        print("data prep only: done.")
        return

    train_arrays, val_arrays = {}, {}
    for n, arr in token_arrays.items():
        tr, va = split_train_val(arr)
        train_arrays[n] = tr
        val_arrays[n] = va
        print(f"  {n}: train={len(tr):,} tok, val={len(va):,} tok")

    mcfg = ModelConfig(vocab_size=len(tokenizer.vocab))
    n_params = sum(p.numel() for p in NanoGPT(mcfg).parameters())
    print(f"model: vocab={mcfg.vocab_size}, embd={mcfg.n_embd}, "
          f"layer={mcfg.n_layer}, head={mcfg.n_head}, block={mcfg.block_size}")
    print(f"per-vertex params: {n_params:,}")

    vertex_models = train_vertex_models(train_arrays, val_arrays, mcfg, refresh=refresh)

    BLOCK = mcfg.block_size
    BS = BATCH_SIZE
    MIXED_VAL, MIXED_TASKS = mixed_val_with_tasks(val_arrays, BLOCK)

    print("\n[eval] vertex specialists per task (clean)")
    vertex_rows = []
    for vi, vm in enumerate(vertex_models):
        by_scope = {}
        for ti, name in enumerate(TASK_NAMES):
            by_scope[f'task_{ti}'] = eval_metrics(
                vm, val_arrays[name], BLOCK, BS, max_batches=MAX_BATCHES
            )['all']
        vertex_rows.append((f"vertex_{vi} ({TASK_NAMES[vi]})", with_mean(by_scope, N)))

    # --- Nimbus: phase 1 (bumps) then phase 2 (gates), both cached ----------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pre_ckpt = CKPT_DIR / "simplex_pre.pt"
    post_ckpt = CKPT_DIR / "simplex_post.pt"
    expected_cfg = {"degrees": tuple(sorted(DEGREES))}
    _, a2k, abd = build_bump_params(vertex_models, DEGREES)

    pre_state = None if refresh else load_simplex_if_match(pre_ckpt, expected_cfg)
    if pre_state is not None:
        bp, dg = unpack_simplex_state(pre_state)
        print(f"[simplex] loaded pre from {pre_ckpt}")
    else:
        print("\n" + "=" * 60 + "\n[phase 1: matched KL]\n" + "=" * 60)
        bp, _a2k, _abd, dg = train_simplex(
            vertex_models, train_arrays, mcfg,
            degrees=DEGREES, seed=stage_seed('simplex_pre'),
        )
        torch.save(pack_simplex_state(bp, dg, DEGREES), pre_ckpt)
        print(f"[simplex] cached pre to {pre_ckpt}")

    post_state = None if refresh else load_simplex_if_match(post_ckpt, expected_cfg)
    if post_state is not None:
        cal_bp, cal_dg = unpack_simplex_state(post_state)
        print(f"[simplex] loaded post from {post_ckpt}")
    else:
        print("\n" + "=" * 60 + "\n[phase 2: nll+both + matched KL]\n" + "=" * 60)
        cal_bp = clone_bp(bp)
        cal_dg = clone_dg(dg)
        calibrate_simplex(
            vertex_models, cal_bp, a2k, abd, cal_dg, train_arrays, mcfg,
            degrees=DEGREES, seed=stage_seed('simplex_post'),
        )
        torch.save(pack_simplex_state(cal_bp, cal_dg, DEGREES), post_ckpt)
        print(f"[simplex] cached post to {post_ckpt}")

    # --- baseline ensemble: M = |D| members, parameter- and compute-matched --
    n_steps = baseline_ens_steps()
    print(f"\n[baseline-ens] M={N_BASELINE} members x {n_steps:,} steps "
          f"(= {NIMBUS_PHASE1_PASSES:,} phase-1 sample-passes / {N_BASELINE})")
    baseline_models = train_baseline_ensemble(
        train_arrays, val_arrays, mcfg, n_models=N_BASELINE, n_steps=n_steps,
        stage_name='baseline', ckpt_prefix="baseline", refresh=refresh,
    )

    # --- |D|+N ensemble: N extra members at the same per-member budget, under a
    # separate seed namespace + checkpoint prefix so the |D| members are reused.
    n_extra = N_BASELINE_LARGE - N_BASELINE
    print(f"\n[baseline-ens +N] {n_extra} extra members x {n_steps:,} steps "
          f"(total {N_BASELINE_LARGE} = |D|+N)")
    baseline_extra_models = train_baseline_ensemble(
        train_arrays, val_arrays, mcfg, n_models=n_extra, n_steps=n_steps,
        stage_name='baseline_extra', ckpt_prefix="baseline_extra", refresh=refresh,
    )
    baseline_dpn_models = baseline_models + baseline_extra_models

    # --- parameter merges of the vertices -----------------------------------
    merge_models = {
        "ties":   ties_merge(vertex_models),
        "fisher": fisher_merge(vertex_models, train_arrays, BLOCK),
    }

    dummy = deepcopy(vertex_models[0]).to(DEVICE).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)

    # --- pools --------------------------------------------------------------
    _rng_int = np.random.default_rng(GLOBAL_SEED)
    interior_ws = [torch.from_numpy(_rng_int.dirichlet(np.ones(N))).float().to(DEVICE)
                   for _ in range(N_ENS)]
    face_ws = sample_face_ws(N, DEGREES, N_FACE_ENS, seed=GLOBAL_SEED, device=DEVICE)
    print(f"\n[pools] interior N={len(interior_ws)}, face_ud N={len(face_ws)}")

    # linear interpolation = the interpolator with bumps gated off (log-gate -inf,
    # so bump strength exp(gate) = 0), sampled with uniform Dirichlet ws.
    dg_off = {k: torch.full_like(v, float("-inf")) for k, v in cal_dg.items()}

    pools = {
        "interior":     build_simplex_pool(vertex_models, cal_bp, a2k, abd, cal_dg,
                                           interior_ws, dummy),
        "face":         build_simplex_pool(vertex_models, cal_bp, a2k, abd, cal_dg,
                                           face_ws, dummy),
        "lin_interior": build_simplex_pool(vertex_models, cal_bp, a2k, abd, dg_off,
                                           interior_ws, dummy),
    }

    trainable_names = [n for n, _ in dummy.named_parameters()]

    def _pd(m):
        return {k: v.detach().clone().to(DEVICE)
                for k, v in canonical_pd(m.state_dict(), trainable_names).items()}

    all_pools = {
        **pools,
        "baseline": [_pd(m) for m in baseline_models],
        "baseline_dpn": [_pd(m) for m in baseline_dpn_models],
        **{name: [_pd(m)] for name, m in merge_models.items()},
    }

    del merge_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- sweeps -------------------------------------------------------------
    input_sweeps, weight_sweeps = {}, {}
    input_sweeps_std, weight_sweeps_std = {}, {}
    # sweeps cached per (pool, axis); a cache is reused if it covers the current
    # metric set.
    sweep_cache = CKPT_DIR / "sweeps"

    def _cached_sweep(pool, axis, fn):
        f = sweep_cache / f"{pool}__{axis}.pkl"
        if f.exists() and not refresh:
            with open(f, "rb") as fh:
                blob = pickle.load(fh)
            if set(METRICS).issubset(blob.get("metrics") or ()):
                print(f"  [sweep cache] {pool}/{axis}")
                return blob["mean"], blob["std"]
            print(f"  [sweep cache] {pool}/{axis} stale (metric set changed) -- redoing")
        mean, std = fn()
        sweep_cache.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump({"metrics": tuple(METRICS), "mean": mean, "std": std}, fh)
        return mean, std

    for pool_name, pds in all_pools.items():
        print(f"\n[sweep] {pool_name} (N={len(pds)}) on mixed-val")
        input_sweeps[pool_name], input_sweeps_std[pool_name] = _cached_sweep(
            pool_name, "input",
            lambda _p=pds: noisy_input_sweep(
                _p, dummy, MIXED_VAL, MIXED_TASKS, BLOCK, BS,
                INPUT_RATES, mcfg.vocab_size,
            ))
        weight_sweeps[pool_name], weight_sweeps_std[pool_name] = _cached_sweep(
            pool_name, "weight",
            lambda _p=pds: noisy_weight_sweep(
                _p, dummy, MIXED_VAL, MIXED_TASKS, BLOCK, BS,
                WEIGHT_EPSILONS,
            ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scopes = scope_keys(N)

    rob_curves = ([(f"Simplex / {m}", pool, key) for m, (pool, key) in METHOD_MAP.items()]
                  + list(BASELINE_LINES))
    rob_footer = ['Task name mapping:'] + [f"  task_{ti}: {nm}" for ti, nm in enumerate(TASK_NAMES)]
    write_robustness_table(
        rob_curves, scopes,
        {
            "input":  (INPUT_RATES,     input_sweeps),
            "weight": (WEIGHT_EPSILONS, weight_sweeps),
        },
        str(RESULTS_DIR / "robustness_table.txt"),
        footer_lines=rob_footer,
    )
    write_robustness_stats_table(
        rob_curves, scopes,
        {
            "input":  (INPUT_RATES,     input_sweeps,  input_sweeps_std),
            "weight": (WEIGHT_EPSILONS, weight_sweeps, weight_sweeps_std),
        },
        str(RESULTS_DIR / "robustness_stats_table.txt"),
        footer_lines=rob_footer,
    )

    def _clean_from_sweep(pool_name, ens_key):
        return {s: {m: float(input_sweeps[pool_name][ens_key][s][m][0]) for m in METRICS}
                for s in scopes}

    simplex_results = {f"Simplex {m}": _clean_from_sweep(pool, key)
                       for m, (pool, key) in METHOD_MAP.items()}
    baseline_results = {name: _clean_from_sweep(pool, key)
                        for name, pool, key in BASELINE_LINES}
    summary_blocks = list(simplex_results.items()) + list(baseline_results.items())
    write_summary_table(summary_blocks, scopes,
                        str(RESULTS_DIR / "summary_table.txt"),
                        footer_lines=rob_footer)

    print("\nvertex diagonal per task (clean NLL / ECE / Acc / Top3 / Top5)\n")
    print_wide_table(vertex_rows, N, task_names=TASK_DISPLAY)

    print("\nclean mixed-val per task (from input sweep @ p=0)\n")
    clean_rows = [(name, with_mean(by_scope, N))
                  for name, by_scope in list(simplex_results.items()) + list(baseline_results.items())]
    print_wide_table(clean_rows, N, task_names=TASK_DISPLAY)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-only", action="store_true",
                   help="download + tokenize + encode the corpora, then stop")
    p.add_argument("--refresh", action="store_true",
                   help="ignore caches and recompute everything")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_main(data_only=args.data_only, refresh=args.refresh)
