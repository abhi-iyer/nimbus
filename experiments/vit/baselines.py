"""baselines for the ViT merging experiment.

merge baselines operate directly on the vertex towers, no training of their own:
  * task arithmetic : theta_base + scale * sum_i (theta_i - theta_base)
  * TIES            : trim -> elect sign -> disjoint-merge the task vectors
  * Fisher          : diagonal-Fisher-weighted average of the vertices
all are defined relative to the pretrained base tower.

the baseline ensemble is the one baseline that trains: |D| key-twin LoRA
members, each an independent multitask fine-tune of base CLIP, param- and
compute-matched to Nimbus.
"""
import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from nimbus.seeds import device, set_seed

from .lora_bump import LoRABump, DenseBump, is_lora_target

from .config import CLIP_BASE, CKPT_DIR, LORA_TARGET_SUFFIXES, SIMPLEX_LR
from .model import base_projection_scale, head_logits, encode, make_dummy
from .train import stage_seed


_BASE_TOWER = None


def base_tower():
    """pretrained CLIP vision tower, the shared init."""
    global _BASE_TOWER
    if _BASE_TOWER is None:
        from transformers import CLIPVisionModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _BASE_TOWER = CLIPVisionModel.from_pretrained(
                CLIP_BASE, use_safetensors=True).to(device).eval()
    return _BASE_TOWER


def free_base_tower():
    """drop the cached base tower from gpu (reloaded lazily on next base_tower())."""
    global _BASE_TOWER
    _BASE_TOWER = None
    torch.cuda.empty_cache()


def _float_sd(model):
    return {n: p.detach().float() for n, p in model.state_dict().items()
            if p.is_floating_point()}


def _load_merged(template, merged_sd):
    """load a merged float state dict into a fresh tower (cast to orig dtype)."""
    m = deepcopy(template).to(device).eval()
    full = m.state_dict()
    for n in full:
        if n in merged_sd:
            full[n] = merged_sd[n].to(full[n].dtype)
    m.load_state_dict(full, strict=True)
    return m


# --- merge baselines (no training) ---

def task_arithmetic_merge(vertices, scale=0.3):
    base = _float_sd(base_tower())
    sds = [_float_sd(v) for v in vertices]
    merged = {}
    for n, b in base.items():
        tv = sum(sd[n] - b for sd in sds)
        merged[n] = b + scale * tv
    return _load_merged(vertices[0], merged)


def ties_merge(vertices, keep_fraction=0.2):
    """
    TIES relative to the pretrained base. keep_fraction is the top-magnitude
    fraction of each task vector to keep (rest trimmed to 0).
    """
    base = _float_sd(base_tower())
    sds = [_float_sd(v) for v in vertices]
    N = len(vertices)
    merged = {}
    for n, b in base.items():
        tvecs = torch.stack([sd[n] - b for sd in sds], dim=0)        # [N, *shape]
        flat = tvecs.reshape(N, -1)
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
    return _load_merged(vertices[0], merged)


def compute_diagonal_fisher(vertex, loader, head, proj, scale, n_batches=50):
    """diagonal Fisher of a vertex on its task, via per-task-head CE gradients."""
    m = deepcopy(vertex).to(device).train()
    fisher = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    n = 0
    for x, y in loader:
        if n >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        m.zero_grad()
        logits = head_logits(m(pixel_values=x).pooler_output, head, proj, scale)
        F.cross_entropy(logits, y).backward()
        for name, p in m.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().pow(2)
        n += 1
    for name in fisher:
        fisher[name] /= max(n, 1)
    return fisher


def fisher_merge(vertices, heads, train_loaders, n_batches=50):
    proj, scale = base_projection_scale()
    N = len(vertices)
    print("[fisher] computing diagonal Fishers...")
    fishers = [compute_diagonal_fisher(vertices[i], train_loaders[i], heads[i],
                                       proj, scale, n_batches) for i in range(N)]
    sds = [v.state_dict() for v in vertices]
    merged = {}
    for name, val in sds[0].items():
        if name in fishers[0] and val.is_floating_point():
            num = sum(fishers[i][name] * sds[i][name].detach() for i in range(N))
            den = sum(fishers[i][name] for i in range(N)).clamp(min=1e-8)
            merged[name] = num / den
        else:
            merged[name] = val.detach().clone()
    return _load_merged(vertices[0], merged)


# --- baseline ensemble: |D| key-twin LoRA members ---
# each member is one Nimbus bump-key's parameterization applied as an independent
# fine-tune of base CLIP: rank-r LoRA on the targeted matrices + dense deltas on
# everything else, trained with per-task CE on the interleaved task stream (no
# distillation).

def _make_adapter(base, r):
    """
    a single key-twin adapter over base's trainable params (LoRA on the targeted
    matrices, dense on the rest).
    """
    tmpl = [(n, p) for n, p in base.named_parameters() if p.requires_grad]
    return {n: (LoRABump.init(tuple(p.shape), r, device)
                if is_lora_target(n, p, LORA_TARGET_SUFFIXES)
                else DenseBump.zeros(p.shape, device))
            for n, p in tmpl}


def _adapter_pd(base_sd, adapter):
    """full vision-tower param dict for encode = base weights + materialized adapter."""
    pd = dict(base_sd)
    for n, bo in adapter.items():
        pd[n] = base_sd[n] + bo.materialize()
    return pd


def _load_adapter(adapter, st):
    for n, bo in adapter.items():
        s = st[n]
        if s["kind"] == "lora":
            bo.B.data.copy_(s["B"].to(device)); bo.A.data.copy_(s["A"].to(device))
        else:
            bo.weight.data.copy_(s["weight"].to(device))


def train_baseline_member(task_names, heads, train_loaders, n_steps, r, seed, tag,
                          refresh=False):
    """
    train one baseline-ensemble member (rank-r LoRA + dense multitask fine-tune
    of base CLIP), cached.

    returns a detached full param dict for streaming eval.
    """
    from .data import CyclingLoader
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{tag}.pt"
    base = base_tower()
    dummy = make_dummy(base)
    base_sd = {n: p.detach() for n, p in base.state_dict().items()}
    proj, scale = base_projection_scale()
    N = len(task_names)

    if ckpt.exists() and not refresh:
        adapter = _make_adapter(base, r)          # init irrelevant here (overwritten)
        _load_adapter(adapter, torch.load(ckpt, map_location=device))
        with torch.no_grad():
            return {n: v.detach().clone() for n, v in _adapter_pd(base_sd, adapter).items()}

    # seed before building the adapter so the per-member LoRA init is
    # deterministic per member.
    set_seed(seed)
    adapter = _make_adapter(base, r)
    streams = [CyclingLoader(tl) for tl in train_loaders]
    # match Nimbus bump training: constant SIMPLEX_LR, AdamW(0.9, 0.95), no
    # warmup or decay.
    opt = torch.optim.AdamW([p for bo in adapter.values() for p in bo.params()],
                            lr=SIMPLEX_LR, betas=(0.9, 0.95), weight_decay=0.0)
    pbar = tqdm(range(n_steps), desc=tag)
    ema = None
    for it in pbar:
        i = it % N                                   # interleave tasks
        x, y = next(streams[i])
        logits = head_logits(encode(dummy, _adapter_pd(base_sd, adapter), x),
                             heads[i], proj, scale)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if it % 50 == 0:
            pbar.set_postfix_str(f"loss={ema:.3f}")
    torch.save({n: bo.state() for n, bo in adapter.items()}, ckpt)
    with torch.no_grad():
        return {n: v.detach().clone() for n, v in _adapter_pd(base_sd, adapter).items()}


def train_baseline_ensemble(task_names, heads, train_loaders, n_models, n_steps, r,
                            stage="baseline_ens", refresh=False):
    """
    M independent key-twin members (the |D| pool is param/compute-matched to
    Nimbus).

    stage (str) selects the seed namespace and checkpoint tag, so the |D|+N extra
    members train under "baseline_ens_dpn" without colliding with the cached
    "baseline_ens_{i}" members.
    """
    return [train_baseline_member(task_names, heads, train_loaders, n_steps, r,
                                  seed=stage_seed(stage, i), tag=f"{stage}_{i}",
                                  refresh=refresh)
            for i in range(n_models)]
