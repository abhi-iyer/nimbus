"""Nimbus bump training (phase 1) + gate calibration (phase 2) for CLIP ViT.

tasks have disjoint label spaces and per-task zero-shot heads, so a face is not
concatenated into one shared-vocab tensor: for each face the interpolated encoder
runs once over the face's images, the pooled embeddings are split per task, and
CE + KD are computed with each task's own head.

bumps come from build_bump_params_lora(..., target_suffixes=ViT matrices) and the
interpolator is LoRASimplexInterpolator (materializes B @ A). face sampling, w
sampling, KD temperature, and the two-phase schedule match the LLM leg.
"""
import hashlib
import math
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nimbus.seeds import device, set_seed
from nimbus.simplex import sample_w_for_degree

from .lora_bump import build_bump_params_lora, count_bump_params
from .lora_simplex import LoRASimplexInterpolator

from .config import (
    GLOBAL_SEED, LORA_TARGET_SUFFIXES,
    SIMPLEX_ITERS_PER_DEGREE, SIMPLEX_LR,
    CAL_ITERS, CAL_LR, N_CAL_SAMPLES, CAL_BATCHES,
    DISTILL_TEMPERATURE,
)
from .model import (
    encode, head_logits, base_projection_scale, vision_param_dict, make_dummy,
)


DEVICE = device


def stage_seed(*parts):
    """deterministic per-stage seed from a variadic key, via a stable sha256 digest."""
    key = "|".join(map(str, parts)).encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return GLOBAL_SEED + (digest % (2 ** 31))


def _vertex_params(vertices):
    return [vision_param_dict(vm) for vm in vertices]


@torch.no_grad()
def _teacher_logits(vertex, x, head, proj, scale):
    return head_logits(vertex(pixel_values=x).pooler_output, head, proj, scale)


def _flat_to_paramdict(interp, i):
    """
    vertex i's param dict reconstructed from interp.vertex_flat, so teacher
    forwards reuse it via the dummy and the vertex modules can be offloaded to cpu
    during training.
    """
    flat = interp.vertex_flat[i]
    pd, off = {}, 0
    for n, sh, sz in zip(interp.param_names, interp.param_shapes, interp.param_sizes):
        pd[n] = flat[off:off + sz].view(sh); off += sz
    pd.update(interp.fixed_sd)
    return pd


@torch.no_grad()
def _teacher_logits_flat(dummy, interp, i, x, head, proj, scale):
    return head_logits(encode(dummy, _flat_to_paramdict(interp, i), x), head, proj, scale)


def train_simplex_vit(vertices, heads, loaders, degrees, r, seed=None):
    """
    phase 1: fit the LoRA bumps by per-task CE + KD distillation.

    degrees are fit sequentially (higher degrees are residuals on the finalized
    lower degrees), so bumps cannot be warm-started key-by-key; extend S by a clean
    retrain.
    """
    if seed is None:
        seed = stage_seed("simplex_pre")
    set_seed(seed)
    lr = SIMPLEX_LR
    n_iters = SIMPLEX_ITERS_PER_DEGREE
    N_ = len(vertices)
    T = DISTILL_TEMPERATURE
    proj, scale = base_projection_scale()

    dummy = make_dummy(vertices[0])
    vertex_params = _vertex_params(vertices)
    bp, a2k, abd = build_bump_params_lora(
        vertices, degrees, r, device=DEVICE, target_suffixes=LORA_TARGET_SUFFIXES)
    dg = {k: nn.Parameter(torch.tensor(0.0, device=DEVICE)) for k in bp}
    interp = LoRASimplexInterpolator(vertex_params, bp, a2k, abd, dummy, degree_gates=dg)
    del vertex_params
    # offload the vertex modules to cpu during the training loop; teacher forwards
    # reuse interp.vertex_flat via the dummy. restored to gpu before returning.
    for v in vertices:
        v.to("cpu")
    torch.cuda.empty_cache()

    print(f"[vit-simplex] N={N_}, degrees={list(degrees)}, rank r={r}")
    print(f"[vit-simplex] {len(bp)} bump set(s), {count_bump_params(bp):,} params")

    streams = [iter(loaders[i]) for i in range(N_)]

    def next_batch(i):
        try:
            return next(streams[i])
        except StopIteration:
            streams[i] = iter(loaders[i])
            return next(streams[i])

    trained_keys = set()
    for rr_group in [[rr] for rr in sorted(degrees)]:
        rng = np.random.default_rng(seed + hash(tuple(rr_group)) % (2 ** 31))
        train_keys = {a2k[a] for rr in rr_group for a in abd[rr]}
        for k, pd in bp.items():
            for bo in pd.values():
                bo.requires_grad_(k in train_keys)
        for k, g in dg.items():
            g.requires_grad_(k in trained_keys)

        train_params = [p for k in train_keys for bo in bp[k].values() for p in bo.params()]
        gate_params = [g for g in dg.values() if g.requires_grad]
        groups = [{"params": train_params, "lr": lr}]
        if gate_params:
            groups.append({"params": gate_params, "lr": lr})
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

        all_faces = [(rr, face) for rr in rr_group for face in combinations(range(N_), rr)]
        n_per_step = max(1, min(len(all_faces), N_))
        do_subsample = n_per_step < len(all_faces)
        print(f"\n--- degrees {rr_group}: {sum(p.numel() for p in train_params):,} "
              f"bump params, {len(all_faces)} face(s), {n_per_step}/step ---")

        pbar = tqdm(range(n_iters), desc=f"deg {rr_group} r{r}")
        loss_ema = None
        for it in pbar:
            batch = [next_batch(i) for i in range(N_)]
            xs = [b[0] for b in batch]
            ys = [b[1] for b in batch]

            step_faces = ([all_faces[i] for i in rng.choice(len(all_faces),
                          size=n_per_step, replace=False)] if do_subsample else all_faces)

            opt.zero_grad(set_to_none=True)
            interp.cache_bump_flat()
            step_loss = 0.0
            for rr, face in step_faces:
                w, _ = sample_w_for_degree(N_, rr, rng=rng, device=DEVICE, face=face)
                denom = len(face) * len(step_faces)
                for i in face:
                    # backward per task, recomputing interp(w) each time so each
                    # encoder graph is freed before the next.
                    stu_i = head_logits(encode(dummy, interp(w), xs[i]), heads[i], proj, scale)
                    ce = F.cross_entropy(stu_i, ys[i])
                    teach = _teacher_logits_flat(dummy, interp, i, xs[i], heads[i], proj, scale)
                    kd = F.kl_div(F.log_softmax(stu_i / T, dim=-1),
                                  F.softmax(teach / T, dim=-1),
                                  reduction="batchmean") * (T * T)
                    loss_i = (ce + kd) / denom
                    loss_i.backward()
                    step_loss += float(loss_i)
            opt.step()
            loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss
            if it % 50 == 0:
                pbar.set_postfix_str(f"loss={loss_ema:.3f}")
        trained_keys.update(train_keys)

    for pd in bp.values():             # drop the trained-bump grad buffers before the
        for bo in pd.values():         # calibration phase, which freezes the bumps
            for p in bo.params():
                p.grad = None
    for v in vertices:                 # restore to gpu for the caller (eval / merges)
        v.to(DEVICE)
    torch.cuda.empty_cache()
    return bp, a2k, abd, dg


def calibrate_simplex_vit(vertices, bp, a2k, abd, dg, heads, loaders, degrees, seed=None):
    """
    phase 2: calibrate degree gates (bumps frozen) against the logit/softmax
    ensemble NLL + matched-KD objective, per task. mutates dg in place.
    """
    if seed is None:
        seed = stage_seed("simplex_post")
    set_seed(seed)
    cal_lr = CAL_LR
    cal_iters = CAL_ITERS
    N_ = len(vertices)
    T = DISTILL_TEMPERATURE
    proj, scale = base_projection_scale()

    dummy = make_dummy(vertices[0])
    vertex_params = _vertex_params(vertices)
    interp = LoRASimplexInterpolator(vertex_params, bp, a2k, abd, dummy, degree_gates=dg)
    del vertex_params; torch.cuda.empty_cache()

    for pd in bp.values():
        for bo in pd.values():
            bo.requires_grad_(False)
    for g in dg.values():
        g.requires_grad_(True)
    train = [g for g in dg.values() if g.requires_grad]
    if not train:
        print("[cal] no gates, skipping")
        return
    opt = torch.optim.AdamW(train, lr=cal_lr, betas=(0.9, 0.95), weight_decay=0.0)

    all_faces = [(rr, face) for rr in sorted(degrees) for face in combinations(range(N_), rr)]
    cnt = {}
    for rr, _ in all_faces:
        cnt[rr] = cnt.get(rr, 0) + 1
    fw = np.array([1.0 / cnt[rr] for rr, _ in all_faces])
    fw = fw / fw.sum()

    streams = [iter(loaders[i]) for i in range(N_)]

    def next_batch(i):
        try:
            return next(streams[i])
        except StopIteration:
            streams[i] = iter(loaders[i])
            return next(streams[i])

    rng = np.random.default_rng(seed + 1234)

    # fixed calibration set + precomputed teacher softmaxes; the gates never touch
    # the frozen vertices, so every teacher KD target is constant across iters.
    cal_x, cal_y = [], []
    for i in range(N_):
        xs_i, ys_i = zip(*(next_batch(i) for _ in range(CAL_BATCHES)))
        cal_x.append(list(xs_i)); cal_y.append(list(ys_i))
    with torch.no_grad():
        # teach_p[i][b] = softmax(vertex_i on task-i's batch b / T), shape [B, C_i].
        # the KD term is matched: task i is only distilled from its own specialist.
        teach_p = [[F.softmax(_teacher_logits(vertices[i], cal_x[i][b], heads[i], proj, scale) / T, dim=-1)
                    for b in range(CAL_BATCHES)] for i in range(N_)]

    for v in vertices:                 # teachers cached; offload modules for the
        v.to("cpu")                    # gate-training loop (student uses vertex_flat)
    torch.cuda.empty_cache()

    print(f"[cal] gates: {len(train)} params, {cal_iters} iters, lr={cal_lr}, "
          f"{N_CAL_SAMPLES} samples/step, "
          f"teachers cached on {CAL_BATCHES} batches/task")

    pbar = tqdm(range(cal_iters), desc="cal-gates")
    loss_ema = None
    for it in pbar:
        b = int(rng.integers(CAL_BATCHES))
        opt.zero_grad(set_to_none=True)
        interp.cache_bump_flat()

        face_idx = rng.choice(len(all_faces), size=N_CAL_SAMPLES, p=fw)
        sampled = []
        for fi in face_idx:
            rr, face = all_faces[fi]
            w, _ = sample_w_for_degree(N_, rr, rng=rng, device=DEVICE, face=face)
            sampled.append(((rr, face), w))

        # per-task: ensemble (over sampled w) logit/softmax NLL + matched-KD.
        step_loss = 0.0
        for i in range(N_):
            x_i, y_i = cal_x[i][b], cal_y[i][b]
            stu_logits = [head_logits(encode(dummy, interp(w), x_i), heads[i], proj, scale)
                          for (_, _), w in sampled]   # each [B, C_i]
            stacked = torch.stack(stu_logits)
            avg_logits = stacked.mean(0)
            avg_probs = F.softmax(stacked, dim=-1).mean(0)
            logit_nll = F.cross_entropy(avg_logits, y_i)
            soft_nll = -torch.log(avg_probs.gather(1, y_i[:, None]).clamp(min=1e-12)).mean()
            loss_i = (logit_nll + soft_nll) / 2

            kl_sum = 0.0
            in_face_count = 0
            for s_idx, ((rr, face), w) in enumerate(sampled):
                if i not in face:
                    continue
                in_face_count += 1

                lp = F.log_softmax(stu_logits[s_idx] / T, dim=-1)
                kl_sum += F.kl_div(lp, teach_p[i][b], reduction="batchmean") * (T * T)

            loss_i = loss_i + kl_sum / max(in_face_count, 1)   # count=0 -> kl_sum is 0.0
            (loss_i / N_).backward()
            
            step_loss += float(loss_i) / N_
        
        opt.step()
        loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss
        if it % 25 == 0:
            gate_str = " ".join(f"g{k}={math.exp(dg[k].item()):.2f}"
                                for k in sorted(dg, key=lambda x: x if isinstance(x, int) else 0))
            pbar.set_postfix_str(f"loss={loss_ema:.3f} | {gate_str}")

    for v in vertices:                 # restore to gpu for the caller (eval / merges)
        v.to(DEVICE)
    torch.cuda.empty_cache()
