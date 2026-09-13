"""
nimbus ablations on CIFAR-100 (N=3).

for each variant, train (or load) a simplex, sample N_W weights from the strict
interior (dirichlet(1)), and report mean / worst per-W test NLL.

variants:
  full       shared P_2, P_3 ; KL on  ; phase 2 on   (== existing simplex_post)
  pairwise   shared P_2 only ; KL on  ; phase 2 on
  interior   shared P_3 only ; KL on  ; phase 2 on
  separate   P_{12,13,23,123} (per-face); KL on ; phase 2 on
  no_kl      shared P_2, P_3 ; KL off ; phase 2 on
  no_phase2  shared P_2, P_3 ; KL on  ; phase 2 off  (== existing simplex_pre)

usage:
  python -m misc.ablations.simplex_design                 # all variants
  python -m misc.ablations.simplex_design --variant full  # one variant
  python -m misc.ablations.simplex_design --n_w 200       # finer simplex sampling
"""
import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from tqdm import tqdm

from nimbus.seeds import set_seed, device
from nimbus.simplex import (
    SimplexInterpolator, build_bump_params,
    bp_from_cpu, bp_to_cpu, clone_bp, clone_dg, dg_from_cpu, dg_to_cpu,
)

from experiments.cifar.config import (
    BATCH_SIZE, CAL_CHUNKS, CAL_EPOCHS, CAL_LR, EPOCHS, GLOBAL_SEED, LR,
    N_CAL_SAMPLES, RESULTS_DIR, stage_seed,
)
from experiments.cifar.data import CIFAR100_CL, CL_DataLoader
from experiments.cifar.model import load_state, make_convnext
from experiments.cifar.train import calibrate_simplex, train_simplex


N = 3
ABL_DIR = Path(__file__).resolve().parent / "simplex_design_results"

# (name, degrees, share_by_degree, use_kl, do_phase2)
VARIANTS = [
    ("full",      [2, 3], True,  True,  True),
    ("pairwise",  [2],    True,  True,  True),
    ("interior",  [3],    True,  True,  True),
    ("separate",  [2, 3], False, True,  True),
    ("no_kl",     [2, 3], True,  False, True),
    ("no_phase2", [2, 3], True,  True,  False),
]


def existing_ckpt(name):
    base = RESULTS_DIR / f"N={N}" / "ckpts"
    if name == "full":
        return base / "simplex_post.pt"
    if name == "no_phase2":
        return base / "simplex_pre.pt"
    return None


def ablation_seed(name):
    return stage_seed(N, "simplex_pre") + 1_000 + abs(hash(name)) % 10_000


def load_or_train(name, degrees, share_by_degree, use_kl, do_phase2,
                  vertex_models, train_loader, test_loader, task_split,
                  ckpt_path, refresh=False):
    src = existing_ckpt(name)
    if src is not None and src.exists() and not refresh:
        print(f"  [reusing existing checkpoint] {src}")
        d = torch.load(src, map_location=device)
    elif os.path.exists(ckpt_path) and not refresh:
        print(f"  [reusing ablation checkpoint] {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
    else:
        d = None

    _, a2k, abd = build_bump_params(vertex_models, degrees, share_by_degree=share_by_degree)
    if d is not None:
        return bp_from_cpu(d["bp"]), a2k, abd, dg_from_cpu(d["dg"])

    seed = ablation_seed(name)
    print(f"  [training] seed={seed} degrees={degrees} share_by_degree={share_by_degree} "
          f"use_kl={use_kl} do_phase2={do_phase2}")
    bp_pre, a2k, abd, dg_pre = train_simplex(
        vertex_models, train_loader, test_loader, task_split,
        EPOCHS, LR, seed, degrees,
        share_by_degree=share_by_degree, use_kl=use_kl,
    )
    if not do_phase2:
        bp, dg = bp_pre, dg_pre
    else:
        bp, dg = clone_bp(bp_pre), clone_dg(dg_pre)
        calibrate_simplex(
            vertex_models, bp, a2k, abd, dg,
            train_loader, test_loader, task_split,
            CAL_EPOCHS, CAL_LR, seed + 7777, degrees,
            n_samples=N_CAL_SAMPLES, chunks=CAL_CHUNKS, use_kl=use_kl,
        )
    torch.save({"bp": bp_to_cpu(bp), "dg": dg_to_cpu(dg)}, ckpt_path)
    print(f"  [saved] {ckpt_path}")
    return bp, a2k, abd, dg


@torch.no_grad()
def per_w_clean_nll(vertex_models, bp, a2k, abd, dg, test_loader, all_classes,
                    n_w, seed):
    Nv = len(vertex_models)
    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    interp = SimplexInterpolator(vertex_params, bp, a2k, abd, dummy, degree_gates=dg)
    interp.cache_bump_flat()

    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(n_w):
        w = rng.dirichlet(np.ones(Nv))
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        ws.append(torch.from_numpy(w).float().to(device))

    params_per_w = [interp(w) for w in ws]

    dl = deepcopy(test_loader)
    dl.set_epoch_data(all_classes)

    nll_sums = np.zeros(n_w, dtype=np.float64)
    n_total = 0
    for x, y in tqdm(dl, desc="per-W NLL"):
        x = x.to(device); y = y.to(device)
        for i, params in enumerate(params_per_w):
            logits = functional_call(dummy, params, (x,))
            nll_sums[i] += F.cross_entropy(logits, y, reduction="sum").item()
        n_total += y.size(0)
    return (nll_sums / max(n_total, 1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    help="one of: all, full, pairwise, interior, separate, no_kl, no_phase2")
    ap.add_argument("--n_w", type=int, default=100,
                    help="number of interior weights sampled per variant")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached ablation checkpoints and retrain")
    args = ap.parse_args()

    set_seed(GLOBAL_SEED)
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    (ABL_DIR / "ckpts").mkdir(exist_ok=True)

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

    vertex_ckpt = RESULTS_DIR / f"N={N}" / "ckpts" / "vertex.pt"
    if not vertex_ckpt.exists():
        raise FileNotFoundError(
            f"vertex checkpoint not found at {vertex_ckpt}. "
            f"Run `python -m experiments.cifar.run --N {N}` first.")
    sds = torch.load(vertex_ckpt, map_location=device)
    vertex_models = []
    for vi in range(N):
        m = make_convnext(seed=vi + 1, num_classes=NUM_CLASSES)
        load_state(m, sds[vi])
        vertex_models.append(m.to(device).eval())

    results_path = ABL_DIR / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name, degrees, share_by_degree, use_kl, do_phase2 in VARIANTS:
        if args.variant != "all" and args.variant != name:
            continue
        print(f"\n=== variant: {name} ===")
        bp, a2k, abd, dg = load_or_train(
            name, degrees, share_by_degree, use_kl, do_phase2,
            vertex_models, train_loader, test_loader, task_split,
            ckpt_path=str(ABL_DIR / "ckpts" / f"{name}.pt"),
            refresh=args.refresh,
        )
        nlls = per_w_clean_nll(
            vertex_models, bp, a2k, abd, dg, test_loader, all_classes,
            n_w=args.n_w, seed=GLOBAL_SEED,
        )
        results[name] = {
            "mean_nll": float(np.mean(nlls)),
            "worst_nll": float(np.max(nlls)),
            "n_w": args.n_w,
            "degrees": list(degrees),
            "share_by_degree": share_by_degree,
            "use_kl": use_kl,
            "do_phase2": do_phase2,
        }
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  mean_nll={results[name]['mean_nll']:.4f}  "
              f"worst_nll={results[name]['worst_nll']:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== ablation table  (N={N}, CIFAR-100, n_w={args.n_w}) ===")
    print(f"{'variant':<12} {'mean clean NLL':>16} {'worst clean NLL':>18}")
    print(f"{'-'*12} {'-'*16:>16} {'-'*18:>18}")
    rows = []
    for name, *_ in VARIANTS:
        if name not in results:
            continue
        r = results[name]
        line = f"{name:<12} {r['mean_nll']:>16.4f} {r['worst_nll']:>18.4f}"
        rows.append(line)
        print(line)
    (ABL_DIR / "summary.txt").write_text(
        f"Nimbus ablations (N={N}, CIFAR-100, n_w={args.n_w})\n"
        f"{'variant':<12} {'mean clean NLL':>16} {'worst clean NLL':>18}\n"
        + "\n".join(rows) + "\n"
    )


if __name__ == "__main__":
    main()
