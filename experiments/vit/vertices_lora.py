"""train 8 single-task LoRA vertices (base CLIP + rank-r LoRA adapter) and report
per-task acc + NLL for each.

early stopping on a held-out val split with constant LR (short warmup, no decay);
keeps the best-val checkpoint. adapters are cached; materialization into towers
happens in a later step.

    python -m experiments.vit.vertices_lora
"""
import json
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from nimbus.seeds import device, set_seed
from . import config as cfg
from . import model as M
from .baselines import base_tower, _make_adapter, _adapter_pd

R = 64
VLR = 1e-4                      # constant after warmup, no decay
WARMUP = 100
MAX_STEPS = 16000               # max training steps
EVAL_EVERY = 200
PATIENCE = 8                    # stop after 8 evals (1600 steps) with no val-acc improvement
VAL_BATCHES = 16
VERT_DIR = cfg.SIMPLEX_CKPT_DIR
OUT = cfg.RESULTS_DIR / "vertices_lora_validation.json"


@torch.no_grad()
def eval_on_task(dummy, pd, head, batches, proj, scale):
    """per-task acc + mean-CE NLL of a single model."""
    correct = tot = 0
    nll_sum = 0.0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = M.head_logits(M.encode(dummy, pd, x), head, proj, scale)
        nll_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        tot += len(y)
    return correct / tot, nll_sum / tot


def _prep_batches(loader, n):
    out = []
    for bi, (x, y) in enumerate(loader):
        if bi >= n:
            break
        out.append((x.clone(), y.clone()))     # CPU; moved to device in eval_on_task
    return out


def _load_into(adapter, st):
    for n, bo in adapter.items():
        s = st[n]
        if s["kind"] == "lora":
            bo.B.data.copy_(s["B"].to(device)); bo.A.data.copy_(s["A"].to(device))
        else:
            bo.weight.data.copy_(s["weight"].to(device))


def materialize_lora_vertex(task, r=R):
    """
    a full vision tower = base + trained LoRA adapter for task, in the shape the
    rest of the pipeline expects (eval mode, requires_grad=True). requires the
    adapter to be trained/cached by this module first.
    """
    ckpt = VERT_DIR / f"vertex_lora_{task}_r{r}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing LoRA vertex adapter {ckpt} -- train via "
                                f"`python -m experiments.vit.vertices_lora` first")
    base = base_tower()
    adapter = _make_adapter(base, r)
    _load_into(adapter, torch.load(ckpt, map_location=device, weights_only=False)["state"])
    vt = deepcopy(base)                          # independent tower (base is on device)
    sd = dict(vt.state_dict())
    with torch.no_grad():
        for n, bo in adapter.items():
            sd[n] = sd[n] + bo.materialize()     # base weight + LoRA/dense delta
    vt.load_state_dict(sd)
    vt.eval()
    for p in vt.parameters():
        p.requires_grad_(True)                   # bump-builder enumerates via requires_grad
    return vt


def load_lora_vertices(task_names, r=R):
    from .baselines import free_base_tower
    vts = [materialize_lora_vertex(t, r) for t in task_names]
    free_base_tower()          # baselines reload it lazily
    return vts


def train_lora_vertex(dummy, base_sd, head, stream, val_batches, proj, scale, seed, tag):
    ckpt = VERT_DIR / f"{tag}.pt"
    base = base_tower()
    if ckpt.exists():
        adapter = _make_adapter(base, R)
        d = torch.load(ckpt, map_location=device, weights_only=False)
        _load_into(adapter, d["state"])
        return adapter, d.get("best_val_acc", float("nan")), d.get("best_step", -1)
    set_seed(seed)
    adapter = _make_adapter(base, R)
    opt = torch.optim.AdamW([p for bo in adapter.values() for p in bo.params()],
                            lr=VLR, betas=(0.9, 0.95), weight_decay=0.0)
    best_acc, best_state, best_step, since = -1.0, None, 0, 0
    pbar = tqdm(range(MAX_STEPS), desc=tag)
    for it in pbar:
        for g in opt.param_groups:
            g["lr"] = VLR * min(1.0, (it + 1) / WARMUP)     # warmup then constant
        x, y = next(stream)
        loss = F.cross_entropy(
            M.head_logits(M.encode(dummy, _adapter_pd(base_sd, adapter), x), head, proj, scale), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (it + 1) % EVAL_EVERY == 0:
            vacc, vnll = eval_on_task(dummy, _adapter_pd(base_sd, adapter), head, val_batches, proj, scale)
            if vacc > best_acc:
                best_acc, best_step, since = vacc, it + 1, 0
                best_state = {n: bo.state() for n, bo in adapter.items()}
            else:
                since += 1
            pbar.set_postfix_str(f"vacc={vacc:.3f} vnll={vnll:.2f} best={best_acc:.3f}@{best_step}")
            if since >= PATIENCE:
                break
    _load_into(adapter, best_state)
    torch.save({"state": best_state, "best_val_acc": best_acc, "best_step": best_step}, ckpt)
    return adapter, best_acc, best_step


def main():
    from .run import setup                       # lazy import
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    S = setup()
    dummy, tasks, heads = S["dummy"], S["tasks"], S["heads"]
    proj, scale = M.base_projection_scale()
    base_sd = {n: p.detach() for n, p in base_tower().state_dict().items()}
    print(f"[vertex-lora] r={R}, lr={VLR} (warmup {WARMUP}, no decay), max_steps={MAX_STEPS}, "
          f"early-stop patience={PATIENCE}x{EVAL_EVERY} on val", flush=True)

    results = {"r": R, "lr": VLR, "max_steps": MAX_STEPS, "tasks": {}}
    print(f"{'task':>10} | {'LoRA acc':>9} {'nll':>6} | {'stop@':>6}", flush=True)
    for i, t in enumerate(tasks):
        test_b = S["eval_batches"][i]
        val_b = _prep_batches(S["loaders"][i]["val"], VAL_BATCHES)
        adapter, bva, bstep = train_lora_vertex(dummy, base_sd, heads[i], S["train_loaders"][i],
                                                val_b, proj, scale, seed=cfg.GLOBAL_SEED + 100 + i,
                                                tag=f"vertex_lora_{t}_r{R}")
        lora_acc, lora_nll = eval_on_task(dummy, _adapter_pd(base_sd, adapter), heads[i], test_b, proj, scale)
        results["tasks"][t] = {"lora_acc": lora_acc, "lora_nll": lora_nll,
                               "best_val_acc": bva, "stop_step": bstep}
        print(f"{t:>10} | {lora_acc:>9.4f} {lora_nll:>6.3f} | {bstep:>6}", flush=True)
        OUT.write_text(json.dumps(results, indent=2))
        torch.cuda.empty_cache()

    macc = sum(v["lora_acc"] for v in results["tasks"].values()) / len(tasks)
    print(f"[vertex-lora] done. mean LoRA vertex acc={macc:.4f}", flush=True)


if __name__ == "__main__":
    main()
