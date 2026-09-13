"""
ablation: how seed-sensitive are Nimbus's robustness curves?

runs the full CIFAR N=3 nimbus pipeline (vertices -> simplex -> calibration) for
N independent seeds and reports the spread of the multitask ("all") NLL vs
epsilon curves at the face pool's logit average.

    python -m misc.ablations.seed_sensitivity                 # 5 seeds (42..46)
    python -m misc.ablations.seed_sensitivity --seeds 42 43   # explicit seeds
    python -m misc.ablations.seed_sensitivity --refresh       # recompute the cache
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from nimbus.seeds import set_seed, device
from nimbus.ensemble import sample_face_ws, build_simplex_pool
from nimbus.simplex import (
    build_bump_params, bp_to_cpu, bp_from_cpu, dg_to_cpu, dg_from_cpu,
)

from experiments.cifar.config import (
    BATCH_SIZE, EPOCHS, LR, CAL_EPOCHS, CAL_LR, N_CAL_SAMPLES, CAL_CHUNKS,
    N_FACE_ENS, GLOBAL_SEED,
)
from experiments.cifar.data import CIFAR100_CL, CL_DataLoader
from experiments.cifar.model import make_convnext, load_state
from experiments.cifar.train import train_simplex, calibrate_simplex, train_one_epoch
from experiments.cifar.baselines import noisy_input_sweep, noisy_weight_sweep

N = 3
DEGREES = [2, 3]
# same epsilon grid as the main robustness figures
EPS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0])
MODE = "logit_avg"      # nimbus face-pool ensembling mode reported
SCOPE = "all"           # multitask NLL
ABL_ROOT = Path(__file__).resolve().parent


# --- seeds ------------------------------------------------------------------
def vseed(master, vi):      return master * 100 + vi
def simplex_seed(master):   return master * 100 + 50
def cal_seed(master):       return master * 100 + 77


# --- data setup (all seeded off master) -------------------------------------
def make_setup(master_seed):
    base_train = CIFAR100_CL(split="train", seed=master_seed)
    base_test = CIFAR100_CL(split="test", seed=master_seed)
    train_loader = CL_DataLoader(base_train, batch_size=BATCH_SIZE, shuffle=True, seed=master_seed)
    test_loader = CL_DataLoader(base_test, batch_size=128, shuffle=True, seed=master_seed)
    all_tasks = torch.randperm(N * 10, generator=torch.Generator().manual_seed(master_seed))
    task_split = [all_tasks[i * 10:(i + 1) * 10] for i in range(N)]
    NUM_CLASSES = int(max(all_tasks).item()) + 1
    all_classes = [c.item() for t in task_split for c in t]
    return dict(base_train=base_train, train_loader=train_loader, test_loader=test_loader,
                task_split=task_split, NUM_CLASSES=NUM_CLASSES, all_classes=all_classes)


# --- single-task vertex training loop ---------------------------------------
def basic_train(model, loader, classes, epochs):
    """AdamW + OneCycle over `classes` for `epochs`."""
    opt = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-3, total_steps=epochs, pct_start=0.2)
    loader.reset_shuffle_state()
    for _ in tqdm(range(epochs), desc="train", leave=False):
        train_one_epoch(model, opt, sched, F.cross_entropy, loader, classes)
    return model.eval()


def train_task_vertices(train_loader, task_split, NUM_CLASSES, master_seed, epochs=EPOCHS):
    """N single-task vertices (vertex i sees only task i's 10 classes)."""
    verts = []
    for vi, t in enumerate(task_split):
        s = vseed(master_seed, vi)
        set_seed(s)
        m = make_convnext(seed=s, num_classes=NUM_CLASSES)
        print(f"  [vertex {vi}] task classes {t.tolist()}")
        m = basic_train(m, train_loader, t.tolist(), epochs)
        verts.append(m.eval())
    return verts


# --- Nimbus (phase 1 + calibration) -----------------------------------------
def train_nimbus(vertex_models, train_loader, test_loader, task_split, master_seed,
                 phase1_epochs=EPOCHS, cal_epochs=CAL_EPOCHS):
    bp, a2k, abd, dg = train_simplex(
        vertex_models, train_loader, test_loader, task_split,
        phase1_epochs, LR, simplex_seed(master_seed), DEGREES,
        share_by_degree=True, use_kl=True)
    calibrate_simplex(
        vertex_models, bp, a2k, abd, dg, train_loader, test_loader, task_split,
        cal_epochs, CAL_LR, cal_seed(master_seed), DEGREES,
        n_samples=N_CAL_SAMPLES, chunks=CAL_CHUNKS, use_kl=True)
    return bp, a2k, abd, dg


# --- robustness curve extraction --------------------------------------------
def _rob_dummy(vertex_models):
    d = deepcopy(vertex_models[0]).to(device).eval()
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def face_pool_pds(vertex_models, bp, a2k, abd, dg, master_seed, n_face=N_FACE_ENS):
    dummy = _rob_dummy(vertex_models)
    face_ws = sample_face_ws(N, DEGREES, n_face, seed=master_seed, device=device)
    pds = build_simplex_pool(vertex_models, bp, a2k, abd, dg, face_ws, dummy)
    return pds, dummy


def nll_curves(pds, dummy, test_loader, all_classes, task_split, master_seed,
               mode=MODE, eps=EPS, n_trials=3):
    """multitask (all-scope) NLL vs epsilon for input and weight noise."""
    in_mean, _ = noisy_input_sweep(pds, dummy, test_loader, all_classes, eps,
                                   n_trials=n_trials, seed=master_seed, task_split=task_split)
    wt_mean, _ = noisy_weight_sweep(pds, dummy, test_loader, all_classes, eps,
                                    n_trials=n_trials, seed=master_seed, task_split=task_split)
    return {"input": in_mean[mode][SCOPE]["nll"].tolist(),
            "weight": wt_mean[mode][SCOPE]["nll"].tolist()}


# --- checkpoint helpers -----------------------------------------------------
def save_vertices(vertex_models, path):
    torch.save([{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for m in vertex_models], path)


def load_vertices(path, NUM_CLASSES):
    sds = torch.load(path, map_location=device)
    out = []
    for vi, sd in enumerate(sds):
        m = make_convnext(seed=vi + 1, num_classes=NUM_CLASSES)
        load_state(m, sd)
        out.append(m.to(device).eval())
    return out


def save_nimbus(bp, dg, path):
    torch.save({"bp": bp_to_cpu(bp), "dg": dg_to_cpu(dg)}, path)


def load_nimbus(path, vertex_models):
    d = torch.load(path, map_location=device)
    _, a2k, abd = build_bump_params(vertex_models, DEGREES, share_by_degree=True)
    return bp_from_cpu(d["bp"]), a2k, abd, dg_from_cpu(d["dg"])


# --- plotting + txt ---------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    return plt


def plot_bands(band, out_pdf, title, eps=EPS):
    """band = {'input': {'mean','std','lo','hi'}, 'weight': {...}}, arrays over eps."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("input", "weight")):
        m = np.array(band[kind]["mean"]); s = np.array(band[kind]["std"])
        ax.plot(eps, m, color="#6a3d9a", lw=2.4, label="Nimbus (mean)")
        ax.fill_between(eps, m - s, m + s, color="#6a3d9a", alpha=0.20,
                        label=r"$\pm$1 std across seeds")
        ax.set_title(f"{kind} noise", fontweight="bold")
        ax.set_xlabel(r"noise $\varepsilon$"); ax.set_ylabel("multitask NLL")
        ax.set_xlim(left=0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)
    print("saved", out_pdf)


KEEP_EPS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def _keep_eps(e):
    """true if this epsilon is one of the reported rows."""
    return any(abs(e - k) < 1e-6 for k in KEEP_EPS)


def _fmt2(x):
    """round to <=2 decimals, dropping trailing zeros but keeping one (e.g. 1.0, 11.8)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print("saved", path)


# --- seed-sensitivity driver ------------------------------------------------
def run_seed(master_seed, ckdir, cfg, refresh=False):
    """per-seed nimbus input/weight NLL curves (cached)."""
    curve_ck = ckdir / f"seed{master_seed}_curve.json"
    if curve_ck.exists() and not refresh:
        print(f"  [reuse] {curve_ck}")
        return json.loads(curve_ck.read_text())

    set_seed(master_seed)
    S = make_setup(master_seed)

    vpath = ckdir / f"seed{master_seed}_vertices.pt"
    if vpath.exists() and not refresh:
        print(f"  [reuse vertices] {vpath}")
        verts = load_vertices(vpath, S["NUM_CLASSES"])
    else:
        verts = train_task_vertices(S["train_loader"], S["task_split"],
                                    S["NUM_CLASSES"], master_seed, epochs=cfg["vert_epochs"])
        save_vertices(verts, vpath)

    npath = ckdir / f"seed{master_seed}_nimbus.pt"
    if npath.exists() and not refresh:
        print(f"  [reuse nimbus] {npath}")
        bp, a2k, abd, dg = load_nimbus(npath, verts)
    else:
        bp, a2k, abd, dg = train_nimbus(verts, S["train_loader"], S["test_loader"],
                                        S["task_split"], master_seed,
                                        phase1_epochs=cfg["phase1_epochs"],
                                        cal_epochs=cfg["cal_epochs"])
        save_nimbus(bp, dg, npath)

    pds, dummy = face_pool_pds(verts, bp, a2k, abd, dg, master_seed, n_face=cfg["n_face"])
    curve = nll_curves(pds, dummy, S["test_loader"], S["all_classes"],
                       S["task_split"], master_seed, eps=cfg["eps"], n_trials=cfg["n_trials"])
    save_json(curve, curve_ck)
    del verts, bp, dg, pds, dummy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curve


def aggregate(curves):
    """per-eps mean / std / min / max across the seed curves, for each kind."""
    band = {}
    for kind in ("input", "weight"):
        M = np.array([c[kind] for c in curves])          # [n_seed, n_eps]
        band[kind] = {"mean": M.mean(0).tolist(), "std": M.std(0, ddof=0).tolist(),
                      "lo": M.min(0).tolist(), "hi": M.max(0).tolist(),
                      "per_seed": M.tolist()}
    return band


def write_txt(band, seeds, out_txt, eps):
    lines = []
    titles = {"input": "**Input noise NLL loss**:", "weight": "**Weight noise NLL loss**:"}
    for kind in ("input", "weight"):
        b = band[kind]
        header = ["$\\varepsilon$", "NLL mean", "NLL std"]
        rows = [[_fmt2(e), _fmt2(m), _fmt2(s)]
                for e, m, s in zip(eps, b["mean"], b["std"])
                if _keep_eps(e)]
        lines += [titles[kind], "", _table(header, rows), ""]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines).rstrip() + "\n")
    print("saved", out_txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached curves/vertices/nimbus and recompute")
    args = ap.parse_args()

    seeds = args.seeds or [GLOBAL_SEED + i for i in range(args.n_seeds)]
    cfg = dict(vert_epochs=EPOCHS, phase1_epochs=EPOCHS, cal_epochs=CAL_EPOCHS,
               n_face=N_FACE_ENS, eps=EPS, n_trials=3)

    out_dir = ABL_ROOT / "seed_sensitivity_results"
    ckdir = out_dir / "ckpts"
    ckdir.mkdir(parents=True, exist_ok=True)

    curves = []
    for s in seeds:
        print(f"\n=== seed {s} ===")
        curves.append(run_seed(s, ckdir, cfg, refresh=args.refresh))

    band = aggregate(curves)
    save_json({"seeds": seeds, "degrees": DEGREES, "eps": cfg["eps"].tolist(),
               "band": band}, out_dir / "results.json")
    write_txt(band, seeds, out_dir / "seed_sensitivity.txt", cfg["eps"])
    plot_bands(band, out_dir / "seed_sensitivity.pdf",
               f"CIFAR N=3 -- Nimbus robustness across {len(seeds)} seeds", eps=cfg["eps"])


if __name__ == "__main__":
    main()
