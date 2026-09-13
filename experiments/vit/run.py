"""orchestrate the ViT merging experiment: train/calibrate Nimbus, run the merge
and ensemble baselines, and evaluate clean (radar) and under input/weight noise
(robustness). writes results/*.txt tables (via tables.py) for the figures; no
JSON cache, a run regenerates the tables.

pools are streamed lazily (one member materialized at a time, see eval.py);
baselines are built, evaluated, and freed one at a time.

    python -m experiments.vit.run              # full run, reusing any cache
    python -m experiments.vit.run --refresh    # ignore caches and recompute
"""
import argparse
import gc

import numpy as np
import torch

from nimbus.seeds import device, set_seed
from nimbus.ensemble import sample_face_ws
from nimbus.simplex import dg_to_cpu, dg_from_cpu

from .lora_bump import build_bump_params_lora, bp_lora_to_state, bp_lora_from_state
from .lora_simplex import LoRASimplexInterpolator

from . import config as cfg
from . import model as M
from . import data as data
from . import baselines as B
from . import eval as E
from . import train as T
from . import tables as RT
from .data import CyclingLoader


# ViT-scaled budgets.
RADAR_MAX_BATCHES = 8
# robustness streams the full clean/face pool (N_FACE_ENS).
ROB_POOL = cfg.N_FACE_ENS
ROB_MAX_BATCHES = 6
ROB_TRIALS = cfg.N_TRIALS


def _free():
    gc.collect(); torch.cuda.empty_cache()


def setup():
    set_seed(cfg.GLOBAL_SEED)
    tasks = cfg.TASK_NAMES
    print(f"[setup] N={cfg.N} tasks={tasks}")
    from .vertices_lora import load_lora_vertices
    vts = load_lora_vertices(tasks)       # base + rank-r LoRA, trained by vertices_lora.py
    print("[setup] using base + LoRA vertices (materialized)")
    heads = [M.build_zero_shot_head(t, data.classnames(t)) for t in tasks]
    loaders = [data.task_loaders(t, batch_size=cfg.BATCH_SIZE) for t in tasks]
    train_loaders = [CyclingLoader(d["train"]) for d in loaders]
    dummy = M.make_dummy(vts[0])
    # decode eval batches once (reused across every pool member / eps / trial).
    eval_batches = E.prepare_eval_batches(loaders, tasks, max(RADAR_MAX_BATCHES, ROB_MAX_BATCHES))
    print(f"[setup] pre-decoded {sum(len(v) for v in eval_batches.values())} eval batches")
    return dict(tasks=tasks, vts=vts, heads=heads, loaders=loaders,
                train_loaders=train_loaders, dummy=dummy, eval_batches=eval_batches)


# --- Nimbus train/cache ----------------------------------------------------

def _nimbus_ckpt():
    cfg.SIMPLEX_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return (cfg.SIMPLEX_CKPT_DIR /
            f"simplex_D{cfg.dtag(cfg.DEGREES)}_r{cfg.RANK}{cfg.VERTEX_SUFFIX}.pt")


def train_or_load_nimbus(S, refresh=False):
    """phase-1 bumps + phase-2 gates for the shipped (cfg.DEGREES, cfg.RANK), cached."""
    path = _nimbus_ckpt()
    _, a2k, abd = build_bump_params_lora(S["vts"], cfg.DEGREES, cfg.RANK, device=device,
                                         target_suffixes=cfg.LORA_TARGET_SUFFIXES)
    if path.exists() and not refresh:
        st = torch.load(path, map_location=device, weights_only=False)
        if st["D"] == list(sorted(cfg.DEGREES)) and st["r"] == cfg.RANK:
            print(f"[nimbus] loaded {path.name}")
            return bp_lora_from_state(st["bp"], device), a2k, abd, dg_from_cpu(st["dg"], device)
    bp, a2k, abd, dg = T.train_simplex_vit(S["vts"], S["heads"], S["train_loaders"],
                                           degrees=cfg.DEGREES, r=cfg.RANK)
    T.calibrate_simplex_vit(S["vts"], bp, a2k, abd, dg, S["heads"],
                            S["train_loaders"], degrees=cfg.DEGREES)
    torch.save({"bp": bp_lora_to_state(bp), "dg": dg_to_cpu(dg),
                "D": list(sorted(cfg.DEGREES)), "r": cfg.RANK}, path)
    print(f"[nimbus] cached {path.name}")
    return bp, a2k, abd, dg


def nimbus_accessor(S, kind, refresh=False):
    """
    lazy pool accessor (get_member, n, gates) over the Nimbus manifold.

    kind="face":   face-stratified ws, the Nimbus pool the ensembles are read from.
    kind="linear": gate the bezier bumps off so interp(w) reduces to the linear
    vertex blend w @ theta (the linear interpolation baseline). gates are log-gates
    (bump strength = exp(gate)), so off = -inf. sampled with uniform dirichlet
    weights over the simplex.
    """
    bp, a2k, abd, dg = train_or_load_nimbus(S, refresh=refresh)
    gates = ({k: torch.full_like(v, float("-inf")) for k, v in dg.items()}
             if kind == "linear" else dg)
    vp = [M.vision_param_dict(v) for v in S["vts"]]
    interp = LoRASimplexInterpolator(vp, bp, a2k, abd, S["dummy"], degree_gates=gates)
    del vp; torch.cuda.empty_cache()   # interp holds vertex_flat
    interp.cache_bump_flat()
    if kind == "face":
        ws = sample_face_ws(cfg.N, cfg.DEGREES, cfg.N_FACE_ENS,
                            seed=cfg.GLOBAL_SEED, device=device)
    else:   # linear: uniform dirichlet over the simplex
        rng = np.random.default_rng(cfg.GLOBAL_SEED)
        ws = [torch.from_numpy(rng.dirichlet(np.ones(cfg.N))).float().to(device)
              for _ in range(cfg.N_ENS)]

    def get(i):
        return {k: v.detach().clone() for k, v in interp(ws[i]).items()}
    return get, len(ws), gates


# --- eval wrappers ---------------------------------------------------------

def eval_clean(get, n, S, max_batches=RADAR_MAX_BATCHES):
    return E.eval_pool(get, n, S["dummy"], S["eval_batches"], S["heads"], S["tasks"],
                       max_batches=max_batches, seed=cfg.GLOBAL_SEED, log=print)


def eval_robust(get, n, S, kind):
    eps = cfg.INPUT_EPSILONS if kind == "input" else cfg.WEIGHT_EPSILONS
    n_use = min(n, ROB_POOL)
    return E.noise_sweep(get, n_use, S["dummy"], S["eval_batches"], S["heads"], S["tasks"],
                         eps, kind, n_trials=ROB_TRIALS, seed=cfg.GLOBAL_SEED,
                         max_batches=ROB_MAX_BATCHES, log=print)


# --- headline run ----------------------------------------------------------

def _eval_into(res, name, get, n, S):
    res["clean"][name] = eval_clean(get, n, S)
    for kind in ("input", "weight"):
        res["robust"][f"{name}_{kind}"] = eval_robust(get, n, S, kind)
    print(f"[baseline {name}] all-scope acc(logit)="
          f"{res['clean'][name]['logit_avg']['all']['acc']:.4f}", flush=True)


def run_headline(S, refresh=False):
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    D, r = cfg.DEGREES, cfg.RANK
    res = {"tasks": S["tasks"], "D": list(sorted(D)), "r": r, "clean": {}, "robust": {}}

    # --- Nimbus ---
    face_get, n_face, dg = nimbus_accessor(S, "face", refresh=refresh)
    res["gates"] = {str(k): float(v) for k, v in dg.items()}
    res["clean"]["nimbus_face"] = eval_clean(face_get, n_face, S)
    for kind in ("input", "weight"):
        res["robust"][f"nimbus_{kind}"] = eval_robust(face_get, n_face, S, kind)
    print(f"[nimbus] all-scope acc(logit)={res['clean']['nimbus_face']['logit_avg']['all']['acc']:.4f}",
          flush=True)
    _free()

    # --- merge baselines (single merged tower each; only fisher needs data) ---
    for name, builder in [("task_arithmetic", lambda: B.task_arithmetic_merge(S["vts"])),
                          ("ties", lambda: B.ties_merge(S["vts"])),
                          ("fisher", lambda: B.fisher_merge(S["vts"], S["heads"], S["train_loaders"]))]:
        model = builder()
        get, n = E.single_member(M.vision_param_dict(model))
        _eval_into(res, name, get, n, S)
        del model, get; _free()

    # --- baseline ensemble: M = |D| key-twin LoRA members, matched to Nimbus
    # (total learnable params == Nimbus's, total steps == Nimbus phase-1 / |D|).
    # eval_pool over the M-member pool yields the logit/softmax/param-avg ensembles. ---
    n_members = len(set(D))
    n_steps = cfg.baseline_ens_steps(D)
    print(f"[baseline-ens] M={n_members} members x {n_steps} steps, rank r={r} "
          f"(Nimbus phase-1 = {cfg.nimbus_phase1_steps(D)} steps)")
    ens_pds = B.train_baseline_ensemble(S["tasks"], S["heads"], S["train_loaders"],
                                        n_models=n_members, n_steps=n_steps, r=r,
                                        refresh=refresh)
    _eval_into(res, "baseline_ens", (lambda i, mem=ens_pds: mem[i]), len(ens_pds), S)
    del ens_pds; _free()

    # --- linear interpolation: linear vertex blends w @ theta over the simplex
    # (the same interpolator with the bezier bumps gated off). the bumps were just
    # (re)built above, so this reuses them. ---
    lin_get, n_lin, _ = nimbus_accessor(S, "linear")
    _eval_into(res, "linear_interp", lin_get, n_lin, S)
    _free()

    # --- |D|+N ensemble: the |D| members above (cached, reused verbatim) plus N
    # extra members at the same per-member budget (not compute- or parameter-matched
    # to Nimbus). ---
    _eval_dpn_into(res, S, refresh=refresh)

    RT.write_tables(res, cfg.RESULTS_DIR)
    print(f"[headline] wrote tables to {cfg.RESULTS_DIR}")
    return res


def _eval_dpn_into(res, S, refresh=False):
    """build the |D|+N pool and evaluate it into res (clean + input/weight noise)."""
    D, r = cfg.DEGREES, cfg.RANK
    n_base, n_extra = len(set(D)), cfg.N
    n_steps = cfg.baseline_ens_steps(D)
    print(f"[baseline-ens +N] {n_base} cached members + {n_extra} extra x {n_steps} "
          f"steps (total {n_base + n_extra} = |D|+N), rank r={r}")
    pds = B.train_baseline_ensemble(S["tasks"], S["heads"], S["train_loaders"],
                                    n_models=n_base, n_steps=n_steps, r=r)
    pds += B.train_baseline_ensemble(S["tasks"], S["heads"], S["train_loaders"],
                                     n_models=n_extra, n_steps=n_steps, r=r,
                                     stage="baseline_ens_dpn", refresh=refresh)
    _eval_into(res, "baseline_ens_dpn", (lambda i, mem=pds: mem[i]), len(pds), S)
    del pds; _free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None,
                    help="override the training/eval batch size (to fit smaller GPUs)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached bumps/gates and baseline members and recompute")
    args = ap.parse_args()
    if args.batch:
        cfg.BATCH_SIZE = args.batch   # setup() reads this for loaders and eval batches
    S = setup()
    run_headline(S, refresh=args.refresh)


if __name__ == "__main__":
    main()
