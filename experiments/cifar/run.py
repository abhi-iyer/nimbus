import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nimbus.ensemble import (
    build_simplex_pool, sample_face_ws, weight_average_params,
)
from nimbus.metrics import eval_metrics_logits, eval_metrics_probs, scope_keys
from nimbus.seeds import set_seed, device
from nimbus.simplex import (
    bp_from_cpu, bp_to_cpu, build_bump_params,
    clone_bp, clone_dg, dg_from_cpu, dg_to_cpu,
)
from nimbus.tables import (
    write_robustness_stats_table, write_robustness_table, write_summary_table,
)

from .baselines import (
    fisher_merge, noisy_input_sweep, noisy_weight_sweep,
    ties_merge, train_asam, train_pcgrad, train_swa,
)
from .config import (
    BATCH_SIZE, CAL_CHUNKS, CAL_EPOCHS, CAL_LR, EPOCHS, EXPERIMENT_CONFIGS,
    GLOBAL_SEED, LR, N_CAL_SAMPLES, N_ENS, N_FACE_ENS, RESULTS_DIR,
    baseline_ens_epochs, baseline_single4x_epochs, baseline_single4x_param_mult,
    stage_seed,
)
from .data import CIFAR100_CL, CL_DataLoader
from .model import load_state, make_convnext, make_convnext_single4x
from .train import (
    average_simplex_model, calibrate_simplex, collect_logits_ensemble,
    collect_logits_model, collect_probs_ensemble, eval_simplex_all, log_eval,
    train_one_epoch, train_simplex,
)


def _ckpt_path(out_dir, name):
    """path to a named checkpoint under out_dir/ckpts."""
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)
    return os.path.join(out_dir, "ckpts", f"{name}.pt")


def _save_sd(model, path):
    """save a model state dict to `path` as cpu tensors."""
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)


def _load_sd_into(model, path):
    """load a state dict from `path` into `model`."""
    sd = torch.load(path, map_location=device)
    load_state(model, sd)
    return model.to(device).eval()


def _train_basic_loop(model, train_loader, all_classes, epochs, total_steps=None,
                      schedule=True):
    """
    train with AdamW. schedule=True uses OneCycle warmup for the vertices;
    schedule=False uses constant lr (config.LR) for the jointly-trained baselines.
    """
    import torch.optim as optim
    if total_steps is None:
        total_steps = epochs
    if schedule:
        opt = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
        sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-3, total_steps=total_steps, pct_start=0.2)
    else:
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        sched = None
    train_loader.reset_shuffle_state()
    for _ in tqdm(range(epochs), desc="train"):
        train_one_epoch(model, opt, sched, F.cross_entropy, train_loader, all_classes)
    return model


def run_experiment(N, degrees, n_baseline, out_dir, refresh=False):
    """train vertices, baselines, and the Nimbus simplex for one N, then write the tables."""
    set_seed(GLOBAL_SEED)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)

    train_loader = CL_DataLoader(
        CIFAR100_CL(split="train", seed=GLOBAL_SEED),
        batch_size=BATCH_SIZE, shuffle=True, seed=GLOBAL_SEED,
    )
    test_loader = CL_DataLoader(
        CIFAR100_CL(split="test", seed=GLOBAL_SEED),
        batch_size=128, shuffle=True, seed=GLOBAL_SEED,
    )

    num_classes_total = N * 10
    all_tasks = torch.randperm(
        num_classes_total,
        generator=torch.Generator().manual_seed(GLOBAL_SEED))
    task_split = [all_tasks[i*10:(i+1)*10] for i in range(N)]
    NUM_CLASSES = max(all_tasks).item() + 1
    all_classes = [c.item() for t in task_split for c in t]

    print(f"\n{'='*80}\nexperiment: N={N}, degrees={degrees}, "
          f"N_BASELINE={n_baseline}, out_dir={out_dir}\n{'='*80}")

    vertex_ckpt = _ckpt_path(out_dir, "vertex")
    vertex_models = []
    if os.path.exists(vertex_ckpt) and not refresh:
        print(f"  [ckpt] loading vertex models from {vertex_ckpt}")
        sds = torch.load(vertex_ckpt, map_location=device)
        for vi in range(N):
            m = make_convnext(seed=vi + 1, num_classes=NUM_CLASSES)
            load_state(m, sds[vi])
            vertex_models.append(m.to(device).eval())
    else:
        for vi, tasks in enumerate(task_split):
            s = stage_seed(N, "vertex", vi)
            set_seed(s)
            model = make_convnext(seed=s, num_classes=NUM_CLASSES)
            model = _train_basic_loop(model, train_loader, tasks.tolist(), EPOCHS)
            vertex_models.append(model.eval())
            lg, lb = collect_logits_model(model, test_loader, tasks.tolist())
            print(f"Model {vi} on task {tasks.tolist()}: acc={eval_metrics_logits(lg, lb)['acc']:.4f}")
        torch.save([{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                    for m in vertex_models], vertex_ckpt)
        print(f"  [ckpt] saved vertex models to {vertex_ckpt}")

    # baseline ensemble: |D| jointly-trained ConvNeXt members, compute-matched to
    # Nimbus phase-1.
    baseline_epochs = baseline_ens_epochs(N, degrees)

    baseline_ckpt = _ckpt_path(out_dir, "baseline")
    baseline_models = []
    if os.path.exists(baseline_ckpt) and not refresh:
        print(f"  [ckpt] loading baseline ensemble from {baseline_ckpt}")
        sds = torch.load(baseline_ckpt, map_location=device)
        assert len(sds) == n_baseline, f"ckpt has {len(sds)} members, expected {n_baseline}"
        for ai in range(n_baseline):
            m = make_convnext(seed=stage_seed(N, "baseline", ai), num_classes=NUM_CLASSES)
            load_state(m, sds[ai])
            baseline_models.append(m.to(device).eval())
    else:
        print(f"\ntraining baseline ensemble: {n_baseline} members x {baseline_epochs} "
              f"epochs (compute-matched to Nimbus phase-1, D={degrees})")
        for ai in range(n_baseline):
            print(f"\ntraining baseline member {ai+1}/{n_baseline}...")
            s = stage_seed(N, "baseline", ai)
            set_seed(s)
            am = make_convnext(seed=s, num_classes=NUM_CLASSES)
            am = _train_basic_loop(am, train_loader, all_classes, baseline_epochs,
                                   schedule=False)
            lg, lb = collect_logits_model(am, test_loader, all_classes)
            print(f"  baseline {ai+1} acc={eval_metrics_logits(lg, lb)['acc']:.4f}")
            baseline_models.append(am)
        torch.save([{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                    for m in baseline_models], baseline_ckpt)
        print(f"  [ckpt] saved baseline ensemble to {baseline_ckpt}")

    # |D|+N ensemble: N extra full-width members at the same per-member budget as
    # the |D| baseline.
    n_extra = N
    baseline_extra_ckpt = _ckpt_path(out_dir, "baseline_extra")
    baseline_extra_models = []
    if os.path.exists(baseline_extra_ckpt) and not refresh:
        print(f"  [ckpt] loading |D|+N extra members from {baseline_extra_ckpt}")
        sds = torch.load(baseline_extra_ckpt, map_location=device)
        assert len(sds) == n_extra, f"ckpt has {len(sds)} members, expected {n_extra}"
        for ai in range(n_extra):
            m = make_convnext(seed=stage_seed(N, "baseline_extra", ai), num_classes=NUM_CLASSES)
            load_state(m, sds[ai])
            baseline_extra_models.append(m.to(device).eval())
    else:
        print(f"\ntraining |D|+N ensemble: {n_extra} extra members x {baseline_epochs} epochs")
        for ai in range(n_extra):
            print(f"\ntraining extra member {ai+1}/{n_extra}...")
            s = stage_seed(N, "baseline_extra", ai)
            set_seed(s)
            am = make_convnext(seed=s, num_classes=NUM_CLASSES)
            am = _train_basic_loop(am, train_loader, all_classes, baseline_epochs,
                                   schedule=False)
            baseline_extra_models.append(am)
        torch.save([{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                    for m in baseline_extra_models], baseline_extra_ckpt)
        print(f"  [ckpt] saved |D|+N extra members to {baseline_extra_ckpt}")

    baseline_dpn_models = baseline_models + baseline_extra_models   # |D|+N members

    # single4x: one wide model at 4x Nimbus phase-1 compute (2*|D|x params times
    # 2x sample-passes).
    single4x_epochs = baseline_single4x_epochs(N, degrees)
    single4x_mult = baseline_single4x_param_mult(degrees)
    single4x_ckpt = _ckpt_path(out_dir, "single4x")
    if os.path.exists(single4x_ckpt) and not refresh:
        print(f"  [ckpt] loading single4x from {single4x_ckpt}")
        single4x_model = make_convnext_single4x(seed=stage_seed(N, "single4x"),
                                                num_classes=NUM_CLASSES,
                                                param_mult=single4x_mult)
        single4x_model = _load_sd_into(single4x_model, single4x_ckpt)
    else:
        print(f"\ntraining single4x: {single4x_mult}x params, {single4x_epochs} epochs "
              f"(2x Nimbus sample-passes)")
        s = stage_seed(N, "single4x")
        set_seed(s)
        single4x_model = make_convnext_single4x(seed=s, num_classes=NUM_CLASSES,
                                                param_mult=single4x_mult)
        single4x_model = _train_basic_loop(single4x_model, train_loader, all_classes,
                                           single4x_epochs, schedule=False)
        _save_sd(single4x_model, single4x_ckpt)
        print(f"  [ckpt] saved single4x to {single4x_ckpt}")

    fisher_ckpt = _ckpt_path(out_dir, "fisher")
    if os.path.exists(fisher_ckpt) and not refresh:
        print(f"  [ckpt] loading Fisher merge from {fisher_ckpt}")
        fisher_model = make_convnext(seed=stage_seed(N, "fisher"), num_classes=NUM_CLASSES)
        fisher_model = _load_sd_into(fisher_model, fisher_ckpt)
    else:
        print("\nrunning Fisher merge...")
        set_seed(stage_seed(N, "fisher"))
        fisher_model = fisher_merge(vertex_models, train_loader, task_split, n_batches=50)
        _save_sd(fisher_model, fisher_ckpt)
        print(f"  [ckpt] saved Fisher merge to {fisher_ckpt}")

    ties_ckpt = _ckpt_path(out_dir, "ties")
    if os.path.exists(ties_ckpt) and not refresh:
        print(f"  [ckpt] loading TIES merge from {ties_ckpt}")
        ties_model = make_convnext(seed=stage_seed(N, "ties"), num_classes=NUM_CLASSES)
        ties_model = _load_sd_into(ties_model, ties_ckpt)
    else:
        print("running TIES merge...")
        ties_model = ties_merge(vertex_models, keep_fraction=0.8)
        _save_sd(ties_model, ties_ckpt)
        print(f"  [ckpt] saved TIES merge to {ties_ckpt}")

    pcgrad_ckpt = _ckpt_path(out_dir, "pcgrad")
    if os.path.exists(pcgrad_ckpt) and not refresh:
        print(f"  [ckpt] loading PCGrad from {pcgrad_ckpt}")
        pcgrad_model = make_convnext(seed=stage_seed(N, "pcgrad"), num_classes=NUM_CLASSES)
        pcgrad_model = _load_sd_into(pcgrad_model, pcgrad_ckpt)
    else:
        print("training PCGrad...")
        s = stage_seed(N, "pcgrad")
        set_seed(s)
        pcgrad_model = make_convnext(seed=s, num_classes=NUM_CLASSES)
        train_loader.reset_shuffle_state()
        pcgrad_model = train_pcgrad(pcgrad_model, train_loader, test_loader, task_split,
                                    epochs=EPOCHS, lr=5e-3, seed=s)
        _save_sd(pcgrad_model, pcgrad_ckpt)
        print(f"  [ckpt] saved PCGrad to {pcgrad_ckpt}")

    asam_ckpt = _ckpt_path(out_dir, "asam")
    if os.path.exists(asam_ckpt) and not refresh:
        print(f"  [ckpt] loading ASAM from {asam_ckpt}")
        asam_model = make_convnext(seed=stage_seed(N, "asam"), num_classes=NUM_CLASSES)
        asam_model = _load_sd_into(asam_model, asam_ckpt)
    else:
        print("training ASAM...")
        s = stage_seed(N, "asam")
        set_seed(s)
        asam_model = make_convnext(seed=s, num_classes=NUM_CLASSES)
        asam_model = train_asam(asam_model, train_loader, all_classes,
                                epochs=EPOCHS, lr=8e-3, rho=0.1)
        _save_sd(asam_model, asam_ckpt)
        print(f"  [ckpt] saved ASAM to {asam_ckpt}")

    swa_ckpt = _ckpt_path(out_dir, "swa")
    if os.path.exists(swa_ckpt) and not refresh:
        print(f"  [ckpt] loading SWA from {swa_ckpt}")
        swa_model = make_convnext(seed=stage_seed(N, "swa"), num_classes=NUM_CLASSES)
        swa_model = _load_sd_into(swa_model, swa_ckpt)
    else:
        print("training SWA...")
        s = stage_seed(N, "swa")
        set_seed(s)
        swa_model = make_convnext(seed=s, num_classes=NUM_CLASSES)
        swa_model = train_swa(swa_model, train_loader, all_classes,
                              epochs=EPOCHS, lr=8e-3, swa_start_epoch=75)
        _save_sd(swa_model, swa_ckpt)
        print(f"  [ckpt] saved SWA to {swa_ckpt}")

    baseline_results = {}

    def _ens_metrics(models, prefix, n):
        """clean-eval logit-ens, softmax-ens, and param-avg for a pool of `models`."""
        per_model = []
        for m in models:
            logits, labels = collect_logits_model(m, test_loader, all_classes)
            per_model.append(([l.cpu() for l in logits], labels))
        labels0 = per_model[0][1]
        nb = len(per_model[0][0])
        logit_ens = [sum(m[0][i] for m in per_model) / n for i in range(nb)]
        baseline_results[f"{prefix} logit ens"] = eval_metrics_logits(
            logit_ens, labels0, task_split=task_split)
        softmax_ens = [sum(F.softmax(m[0][i], dim=-1) for m in per_model) / n
                       for i in range(nb)]
        baseline_results[f"{prefix} softmax ens"] = eval_metrics_probs(
            softmax_ens, labels0, task_split=task_split)
        avg_sd = weight_average_params([m.state_dict() for m in models])
        avg_model = deepcopy(models[0]); avg_model.load_state_dict(avg_sd, strict=True)
        avg_model = avg_model.to(device).eval()
        pa_logits, pa_labels = collect_logits_model(avg_model, test_loader, all_classes)
        baseline_results[f"{prefix} param avg"] = eval_metrics_logits(
            pa_logits, pa_labels, task_split=task_split)
        del avg_model

    # baseline ensemble (|D| members) and the |D|+N ensemble, across logit,
    # softmax, and param-avg columns.
    _ens_metrics(baseline_models, "Baseline", n_baseline)
    _ens_metrics(baseline_dpn_models, "Baseline+N", len(baseline_dpn_models))

    # single-model baselines: single4x plus the parameter merges and training tricks.
    for name, m in [("single4x", single4x_model),
                    ("Fisher merge", fisher_model),
                    ("TIES merge", ties_model), ("PCGrad", pcgrad_model),
                    ("ASAM", asam_model), ("SWA", swa_model)]:
        lg, lb = collect_logits_model(m, test_loader, all_classes)
        baseline_results[name] = eval_metrics_logits(lg, lb, task_split=task_split)

    for k in sorted(baseline_results.keys()):
        v = baseline_results[k]["all"]
        print(f"  {k}: acc={v['acc']:.3f}  top3={v['top3']:.3f}  top5={v['top5']:.3f}  "
              f"nll={v['nll']:.3f}  ece={v['ece']:.3f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    simplex_pre_ckpt = _ckpt_path(out_dir, "simplex_pre")
    if os.path.exists(simplex_pre_ckpt) and not refresh:
        print(f"  [ckpt] loading simplex_pre from {simplex_pre_ckpt}")
        _, a2k, abd = build_bump_params(vertex_models, degrees)
        d = torch.load(simplex_pre_ckpt, map_location=device)
        bp_phase1 = bp_from_cpu(d["bp"])
        dg_phase1 = dg_from_cpu(d["dg"])
    else:
        print("\n" + "="*60 + "\n[phase 1: matched]\n" + "="*60)
        bp_phase1, a2k, abd, dg_phase1 = train_simplex(
            vertex_models=vertex_models,
            train_loader=train_loader,
            test_loader=test_loader,
            task_split=task_split,
            epochs=EPOCHS, lr=LR, seed=stage_seed(N, "simplex_pre"),
            degrees=degrees,
        )
        torch.save({"bp": bp_to_cpu(bp_phase1), "dg": dg_to_cpu(dg_phase1)}, simplex_pre_ckpt)
        print(f"  [ckpt] saved simplex_pre to {simplex_pre_ckpt}")

    simplex_post_ckpt = _ckpt_path(out_dir, "simplex_post")
    if os.path.exists(simplex_post_ckpt) and not refresh:
        print(f"  [ckpt] loading simplex_post from {simplex_post_ckpt}")
        d = torch.load(simplex_post_ckpt, map_location=device)
        bp = bp_from_cpu(d["bp"])
        dg = dg_from_cpu(d["dg"])
    else:
        print("\n" + "="*60 + "\n[phase 2: nll+both / uniform_degree]\n" + "="*60)
        bp, dg = clone_bp(bp_phase1), clone_dg(dg_phase1)
        calibrate_simplex(
            vertex_models, bp, a2k, abd, dg,
            train_loader, test_loader, task_split,
            CAL_EPOCHS, CAL_LR, stage_seed(N, "simplex_post"), degrees,
            n_samples=N_CAL_SAMPLES, chunks=CAL_CHUNKS,
        )
        torch.save({"bp": bp_to_cpu(bp), "dg": dg_to_cpu(dg)}, simplex_post_ckpt)
        print(f"  [ckpt] saved simplex_post to {simplex_post_ckpt}")

    simplex_results = eval_simplex_all(
        vertex_models, bp, a2k, abd, dg,
        test_loader, all_classes,
        N_ENS, N_FACE_ENS, N, degrees, GLOBAL_SEED,
        task_split=task_split,
    )
    log_eval("[simplex]", simplex_results, dg)

    # linear interpolation: bumps gated off (log-gate -inf, so exp(gate)=0), giving
    # the pure linear vertex blend interp(w) = w @ theta, sampled with uniform
    # dirichlet weights.
    dg_off = {k: torch.full_like(v, float("-inf")) for k, v in dg.items()}
    lin_rng = np.random.default_rng(GLOBAL_SEED)
    lin_ws = [torch.from_numpy(lin_rng.dirichlet(np.ones(N))).float().to(device)
              for _ in range(N_ENS)]
    ll, lly = collect_logits_ensemble(vertex_models, bp, a2k, abd, test_loader,
                                      all_classes, degree_gates=dg_off, ws=lin_ws)
    lp, lpy = collect_probs_ensemble(vertex_models, bp, a2k, abd, test_loader,
                                     all_classes, degree_gates=dg_off, ws=lin_ws)
    lin_avg = average_simplex_model(vertex_models, bp, a2k, abd, degree_gates=dg_off,
                                    n_samples=500, seed=GLOBAL_SEED)
    la_l, la_y = collect_logits_model(lin_avg, test_loader, all_classes)
    baseline_results["Linear logit ens"]   = eval_metrics_logits(ll, lly, task_split=task_split)
    baseline_results["Linear softmax ens"] = eval_metrics_probs(lp, lpy, task_split=task_split)
    baseline_results["Linear param avg"]   = eval_metrics_logits(la_l, la_y, task_split=task_split)
    del lin_avg

    MODEL_NAMES = ("logit ens uniform degree", "softmax ens uniform degree", "param avg")
    SCOPES = scope_keys(N)

    method_blocks = []
    for model_name in MODEL_NAMES:
        method_blocks.append((f"Simplex {model_name}", simplex_results[model_name]))
    for k in sorted(baseline_results.keys()):
        method_blocks.append((k, baseline_results[k]))

    task_legend = ["Task composition (CIFAR-100 class indices):"]
    for ti, t in enumerate(task_split):
        task_legend.append(f"  task_{ti}: {sorted(c.item() for c in t)}")

    write_summary_table(
        method_blocks, SCOPES,
        os.path.join(out_dir, "summary_table.txt"),
        footer_lines=task_legend,
    )

    rob_dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in rob_dummy.parameters():
        p.requires_grad_(False)

    # separate dummy for the wide single4x model.
    rob_single4x_dummy = deepcopy(single4x_model).to(device).eval()
    for p in rob_single4x_dummy.parameters():
        p.requires_grad_(False)

    rng_int = np.random.default_rng(GLOBAL_SEED)
    interior_ws = [torch.from_numpy(rng_int.dirichlet(np.ones(N))).float().to(device)
                   for _ in range(N_ENS)]
    face_ws = sample_face_ws(N, degrees, N_FACE_ENS, seed=GLOBAL_SEED, device=device)

    print("\nbuilding simplex param dict pools...")
    simplex_pools = {
        "interior": build_simplex_pool(vertex_models, bp, a2k, abd, dg, interior_ws, rob_dummy),
        "face":     build_simplex_pool(vertex_models, bp, a2k, abd, dg, face_ws, rob_dummy),
        # linear interpolation: uniform (interior) ws with the bumps gated off.
        "lin_interior": build_simplex_pool(vertex_models, bp, a2k, abd, dg_off, interior_ws, rob_dummy),
    }

    def _to_pds(models):
        return [{k: v.detach().clone().to(device) for k, v in m.state_dict().items()}
                for m in models]

    baseline_pds = _to_pds(baseline_models)
    baseline_dpn_pds = _to_pds(baseline_dpn_models)
    single4x_pds = _to_pds([single4x_model])
    fisher_pds = _to_pds([fisher_model])
    ties_pds = _to_pds([ties_model])
    pcgrad_pds = _to_pds([pcgrad_model])
    asam_pds = _to_pds([asam_model])
    swa_pds = _to_pds([swa_model])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epsilons = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    sweep_specs = [
        ("interior",     simplex_pools["interior"], rob_dummy),
        ("face",         simplex_pools["face"],     rob_dummy),
        ("lin_interior", simplex_pools["lin_interior"], rob_dummy),
        ("baseline",     baseline_pds,              rob_dummy),
        ("baseline_dpn", baseline_dpn_pds,          rob_dummy),
        ("single4x",     single4x_pds,              rob_single4x_dummy),
        ("fisher",       fisher_pds,                rob_dummy),
        ("ties",         ties_pds,                  rob_dummy),
        ("pcgrad",       pcgrad_pds,                rob_dummy),
        ("asam",         asam_pds,                  rob_dummy),
        ("swa",          swa_pds,                   rob_dummy),
    ]

    input_sweeps, weight_sweeps = {}, {}
    input_sweeps_std, weight_sweeps_std = {}, {}
    for pool_name, pds, dummy_for_pool in sweep_specs:
        print(f"\nsweeping {pool_name} (N={len(pds)})...")
        input_sweeps[pool_name], input_sweeps_std[pool_name] = noisy_input_sweep(
            pds, dummy_for_pool, test_loader, all_classes,
            epsilons, n_trials=3, seed=GLOBAL_SEED, task_split=task_split)
        weight_sweeps[pool_name], weight_sweeps_std[pool_name] = noisy_weight_sweep(
            pds, dummy_for_pool, test_loader, all_classes,
            epsilons, n_trials=3, seed=GLOBAL_SEED, task_split=task_split)

    METHOD_ROWS = [
        ("logit ens uniform degree",   "face",     "logit_avg"),
        ("softmax ens uniform degree", "face",     "softmax_avg"),
        ("param avg",                  "interior", "weight_avg"),
    ]

    BASELINE_LINES = [
        ("Linear logit ens",       "lin_interior", "logit_avg"),
        ("Linear softmax ens",     "lin_interior", "softmax_avg"),
        ("Linear param avg",       "lin_interior", "weight_avg"),
        ("Baseline logit ens",     "baseline",     "logit_avg"),
        ("Baseline softmax ens",   "baseline",     "softmax_avg"),
        ("Baseline param avg",     "baseline",     "weight_avg"),
        ("Baseline+N logit ens",   "baseline_dpn", "logit_avg"),
        ("Baseline+N softmax ens", "baseline_dpn", "softmax_avg"),
        ("Baseline+N param avg",   "baseline_dpn", "weight_avg"),
        ("single4x",               "single4x",     "single"),
        ("Fisher merge",           "fisher",       "single"),
        ("TIES merge",             "ties",         "single"),
        ("PCGrad",                 "pcgrad",       "single"),
        ("ASAM",                   "asam",         "single"),
        ("SWA",                    "swa",          "single"),
    ]

    curves = [(f"Simplex / {m}", pk, ek) for m, pk, ek in METHOD_ROWS] + list(BASELINE_LINES)

    rob_footer = ["", "Task composition (CIFAR-100 class indices):"]
    for ti, t in enumerate(task_split):
        rob_footer.append(f"  task_{ti}: {sorted(c.item() for c in t)}")

    write_robustness_table(
        curves, SCOPES,
        sweeps_by_kind={"input": (epsilons, input_sweeps),
                        "weight": (epsilons, weight_sweeps)},
        out_path=os.path.join(out_dir, "robustness_table.txt"),
        footer_lines=rob_footer,
    )

    write_robustness_stats_table(
        curves, SCOPES,
        sweeps_by_kind={"input": (epsilons, input_sweeps, input_sweeps_std),
                        "weight": (epsilons, weight_sweeps, weight_sweeps_std)},
        out_path=os.path.join(out_dir, "robustness_stats_table.txt"),
        footer_lines=rob_footer,
    )

    del vertex_models, baseline_models, baseline_extra_models, baseline_dpn_models
    del single4x_model
    del fisher_model, ties_model, pcgrad_model, asam_model, swa_model
    del bp, dg, bp_phase1, dg_phase1
    del simplex_pools
    del baseline_pds, baseline_dpn_pds, single4x_pds
    del fisher_pds, ties_pds, pcgrad_pds, asam_pds, swa_pds
    del input_sweeps, weight_sweeps, input_sweeps_std, weight_sweeps_std
    del rob_dummy, rob_single4x_dummy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """run each configured experiment, optionally filtered by --N."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=None,
                    help="run only the config with this N value")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached checkpoints and recompute")
    args = ap.parse_args()

    for N, degrees, n_baseline in EXPERIMENT_CONFIGS:
        if args.N is not None and N != args.N:
            continue
        out_dir = str(RESULTS_DIR / f"N={N}")
        run_experiment(N=N, degrees=degrees, n_baseline=n_baseline, out_dir=out_dir,
                       refresh=args.refresh)


if __name__ == "__main__":
    main()
