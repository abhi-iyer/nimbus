import math
from copy import deepcopy
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from tqdm import tqdm

from nimbus.metrics import (
    scope_keys, new_scoped_states, accumulate_scoped, finalize_scoped,
)
from nimbus.seeds import device, set_seed
from nimbus.simplex import (
    SimplexInterpolator, build_bump_params, sample_w_for_degree,
)

from .config import (
    N, TASK_NAMES, CKPT_DIR, BATCH_SIZE,
    VERTEX_ITERS, VERTEX_LR, VERTEX_WARMUP,
    VERTEX_EVAL_EVERY, VERTEX_PATIENCE, VERTEX_VAL_BATCHES,
    SIMPLEX_ITERS_PER_DEGREE, SIMPLEX_LR,
    CAL_ITERS, CAL_LR, N_CAL_SAMPLES, CAL_CHUNKS,
    DEGREES, DISTILL_TEMPERATURE, GLOBAL_SEED,
)
from .data import CyclingLoader, make_loader, val_batches
from .model import NanoGPT


DEVICE = device


STAGE_OFFSET = {
    'vertex':            0,
    'simplex_pre':     100,
    'simplex_post':    200,
    'baseline':           300,   # the |D| compute+param-matched ensemble members
    'baseline_extra':     400,   # the |D|+N extra members
}


def stage_seed(name, idx=0):
    return GLOBAL_SEED + STAGE_OFFSET[name] + idx


@torch.no_grad()
def eval_metrics(model, val_array, block_size, batch_size, max_batches=30,
                 task_ids=None):
    was_training = model.training
    model.eval()
    scopes = scope_keys(N) if task_ids is not None else ['all']
    states = new_scoped_states(scopes, DEVICE)
    any_b = False
    for x, y, t in val_batches(val_array, block_size, batch_size, max_batches, task_ids):
        logits, _ = model(x, None)
        accumulate_scoped(logits, y, t, states)
        any_b = True
    if was_training:
        model.train()
    return finalize_scoped(states, any_b)


@torch.no_grad()
def eval_nll(model, val_array, block_size, batch_size, max_batches=30):
    return eval_metrics(model, val_array, block_size, batch_size, max_batches)['all']['nll']


def train_vertex_models(train_arrays, val_arrays, mcfg, refresh=False):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for ti, name in enumerate(TASK_NAMES):
        ckpt = CKPT_DIR / f"vertex_{name}.pt"
        m = NanoGPT(mcfg).to(DEVICE)
        if ckpt.exists() and not refresh:
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m.eval()
            print(f"[vertex {ti}:{name}] loaded {ckpt}")
            models.append(m)
            continue

        s = stage_seed('vertex', ti)
        set_seed(s)
        loader = CyclingLoader(make_loader(
            train_arrays[name], mcfg.block_size, BATCH_SIZE,
            shuffle=True, seed=s,
        ))
        opt = torch.optim.AdamW(m.parameters(), lr=VERTEX_LR,
                                betas=(0.9, 0.95), weight_decay=0.0)
        m.train()
        pbar = tqdm(range(VERTEX_ITERS), desc=f"vertex {ti}:{name}")
        loss_ema = None
        best_vl, best_state, best_it, since = float("inf"), None, 0, 0
        for it in pbar:
            x, y = next(loader)
            x, y = x.to(DEVICE), y.to(DEVICE)
            # constant LR after warmup.
            cur_lr = VERTEX_LR * min(1.0, (it + 1) / max(1, VERTEX_WARMUP))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            _, loss = m(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            l = loss.item()
            loss_ema = l if loss_ema is None else 0.95 * loss_ema + 0.05 * l
            if it % 100 == 0:
                pbar.set_postfix(loss=f"{loss_ema:.3f}", lr=f"{cur_lr:.1e}")
            if (it + 1) % VERTEX_EVAL_EVERY == 0:
                vl = eval_nll(m, val_arrays[name], mcfg.block_size, BATCH_SIZE,
                              max_batches=VERTEX_VAL_BATCHES)
                m.train()
                if vl < best_vl - 1e-4:
                    best_vl, best_it, since = vl, it + 1, 0
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                else:
                    since += 1
                pbar.set_postfix(loss=f"{loss_ema:.3f}", val=f"{vl:.4f}",
                                 best=f"{best_vl:.4f}@{best_it}")
                if since >= VERTEX_PATIENCE:
                    print(f"[vertex {ti}:{name}] early stop at {it+1} "
                          f"(no val gain for {VERTEX_PATIENCE} checks)")
                    break
        if best_state is not None:            # restore the best-val weights, not the last
            m.load_state_dict(best_state)
        torch.save(m.state_dict(), ckpt)
        # same batch count as the early-stopping metric.
        vl = eval_nll(m, val_arrays[name], mcfg.block_size, BATCH_SIZE,
                      max_batches=VERTEX_VAL_BATCHES)
        print(f"[vertex {ti}:{name}] val_nll={vl:.4f} (best {best_vl:.4f} @ it={best_it})")
        m.eval()
        models.append(m)
    return models


def train_simplex(vertex_models, train_arrays, mcfg, degrees=DEGREES, seed=None):
    """
    phase 1: fit the dense bezier bumps by per-face CE + KD distillation.

    degrees are fit sequentially, each key a residual on the finalized lower keys.
    """
    if seed is None:
        seed = stage_seed('simplex_pre')
    lr = SIMPLEX_LR
    n_iters = SIMPLEX_ITERS_PER_DEGREE
    set_seed(seed)
    N_ = len(vertex_models)
    T = DISTILL_TEMPERATURE

    dummy = deepcopy(vertex_models[0]).to(DEVICE).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    for vm in vertex_models:
        vm.eval()

    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    bp, a2k, abd = build_bump_params(vertex_models, degrees)
    for pd in bp.values():
        for n, p in pd.items():
            pd[n] = nn.Parameter(p.data.to(DEVICE))

    dg = {k: nn.Parameter(torch.tensor(0.0, device=DEVICE)) for k in bp}
    interp = SimplexInterpolator(vertex_params, bp, a2k, abd, dummy, degree_gates=dg)

    n_per_set = sum(p.numel() for p in vertex_params[0].values())
    print(f"[simplex] N={N_}, degrees={list(degrees)} (binary, per_degree, matched)")
    print(f"[simplex] {len(bp)} bump set(s) x {n_per_set:,} params each")
    for r in sorted(degrees):
        print(f"  degree {r}: {len(abd[r])} alpha(s)")

    loaders = [
        CyclingLoader(make_loader(train_arrays[n], mcfg.block_size, BATCH_SIZE,
                                  shuffle=True, seed=seed + i + 1000))
        for i, n in enumerate(TASK_NAMES)
    ]

    degree_groups = [[r] for r in sorted(degrees)]
    trained_keys = set()
    for dg_group in degree_groups:
        rng = np.random.default_rng(seed + hash(tuple(dg_group)) % (2 ** 31))

        train_keys = set()
        for r in dg_group:
            for a in abd[r]:
                train_keys.add(a2k[a])
        for k, pd in bp.items():
            for p in pd.values():
                p.requires_grad_(k in train_keys)
        for k, g in dg.items():
            g.requires_grad_(k in trained_keys)

        train_params = [p for k in train_keys for p in bp[k].values()]
        gate_params = [g for g in dg.values() if g.requires_grad]
        groups = [{"params": train_params, "lr": lr}]
        if gate_params:
            groups.append({"params": gate_params, "lr": lr})
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

        all_faces = [(r, face) for r in dg_group for face in combinations(range(N_), r)]
        n_per_step = max(1, min(len(all_faces), N_))
        do_subsample = n_per_step < len(all_faces)

        n_train = sum(p.numel() for p in train_params)
        gate_info = f" + {len(gate_params)} gate(s)" if gate_params else ""
        print(f"\n--- degrees {dg_group}: {n_train:,} bump params{gate_info}, "
              f"{len(all_faces)} face(s), {n_per_step}/step ---")

        pbar = tqdm(range(n_iters), desc=f"deg {dg_group}")
        loss_ema = None
        for it in pbar:
            batches = [next(l) for l in loaders]
            xs = [b[0].to(DEVICE) for b in batches]
            ys = [b[1].to(DEVICE) for b in batches]

            if do_subsample:
                fidx = rng.choice(len(all_faces), size=n_per_step, replace=False)
                step_faces = [all_faces[i] for i in fidx]
            else:
                step_faces = all_faces

            opt.zero_grad(set_to_none=True)
            interp.cache_bump_flat()

            step_loss = 0.0
            for r, face in step_faces:
                w, _ = sample_w_for_degree(N_, r, rng=rng, device=DEVICE, face=face)
                x_f = torch.cat([xs[i] for i in face])
                y_f = torch.cat([ys[i] for i in face])
                cp = interp(w)
                stu_logits, _ = functional_call(dummy, cp, (x_f, None))
                V = stu_logits.size(-1)
                face_loss = F.cross_entropy(stu_logits.view(-1, V), y_f.view(-1))

                with torch.no_grad():
                    teach = torch.cat(
                        [vertex_models[i](xs[i])[0] for i in face], dim=0
                    )
                    teach_p = F.softmax(teach / T, dim=-1)
                stu_lp = F.log_softmax(stu_logits / T, dim=-1)
                kd = F.kl_div(
                    stu_lp.reshape(-1, V),
                    teach_p.reshape(-1, V),
                    reduction="batchmean",
                ) * (T * T)
                face_loss = face_loss + kd

                face_loss = face_loss / len(step_faces)
                face_loss.backward()
                step_loss += face_loss.item()
            opt.step()
            loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss
            if it % 50 == 0:
                gate_str = ""
                if dg:
                    gate_str = " | " + " ".join(
                        f"g{k}={math.exp(dg[k].item()):.2f}"
                        for k in sorted(dg, key=lambda x: x if isinstance(x, int) else 0)
                    )
                pbar.set_postfix_str(f"loss={loss_ema:.3f}{gate_str}")

        trained_keys.update(train_keys)

    return bp, a2k, abd, dg


def calibrate_simplex(vertex_models, bp, a2k, abd, dg, train_arrays, mcfg,
                      degrees=DEGREES, seed=None):
    """
    phase 2: fit the per-degree log-gates (bumps frozen) against the
    logit/softmax ensemble NLL + matched-KD objective.
    """
    if seed is None:
        seed = stage_seed('simplex_post')
    cal_lr = CAL_LR
    cal_iters = CAL_ITERS
    N_ = len(vertex_models)
    set_seed(seed)
    T = DISTILL_TEMPERATURE

    dummy = deepcopy(vertex_models[0]).to(DEVICE).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    for vm in vertex_models:
        vm.eval()

    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    interp = SimplexInterpolator(vertex_params, bp, a2k, abd, dummy, degree_gates=dg)

    for pd in bp.values():
        for p in pd.values():
            p.requires_grad_(False)
    for g in dg.values():
        g.requires_grad_(True)

    train = list(dg.values())
    if not train:
        print("[cal] no gates, skipping")
        return
    opt = torch.optim.AdamW(train, lr=cal_lr, betas=(0.9, 0.95), weight_decay=0.0)

    all_faces = [(r, face) for r in sorted(degrees) for face in combinations(range(N_), r)]
    cnt = {}
    for r, _ in all_faces:
        cnt[r] = cnt.get(r, 0) + 1
    fw = np.array([1.0 / cnt[r] for r, _ in all_faces])
    fw = fw / fw.sum()

    loaders = [
        CyclingLoader(make_loader(train_arrays[n], mcfg.block_size, BATCH_SIZE,
                                  shuffle=True, seed=seed + i + 5000))
        for i, n in enumerate(TASK_NAMES)
    ]
    rng = np.random.default_rng(seed + 1234)

    n_train = sum(p.numel() for p in train)
    print(f"[cal] nll+both + matched KL (uniform_degree): {n_train:,} gate params, "
          f"{cal_iters} iters, lr={cal_lr}, {N_CAL_SAMPLES} samples/step, "
          f"chunks={CAL_CHUNKS}")

    pbar = tqdm(range(cal_iters), desc="cal-gates")
    loss_ema = None
    for it in pbar:
        batches = [next(l) for l in loaders]
        xs_full = [b[0].to(DEVICE) for b in batches]
        ys_full = [b[1].to(DEVICE) for b in batches]

        opt.zero_grad(set_to_none=True)
        interp.cache_bump_flat()

        face_indices = rng.choice(len(all_faces), size=N_CAL_SAMPLES, p=fw)
        sampled = []
        for fi in face_indices:
            r, face = all_faces[fi]
            w, _ = sample_w_for_degree(N_, r, rng=rng, device=DEVICE, face=face)
            sampled.append(((r, face), w))

        x_chunks = [torch.chunk(x, CAL_CHUNKS, dim=0) for x in xs_full]
        y_chunks = [torch.chunk(y, CAL_CHUNKS, dim=0) for y in ys_full]
        n_chunks = len(x_chunks[0])

        step_loss = 0.0
        for c in range(n_chunks):
            xs = [x_chunks[i][c] for i in range(N_)]
            ys = [y_chunks[i][c] for i in range(N_)]
            x_all = torch.cat(xs)
            y_all = torch.cat(ys)

            all_logits = [functional_call(dummy, interp(w), (x_all, None))[0]
                          for (_, _), w in sampled]
            stacked = torch.stack(all_logits)
            avg_logits = stacked.mean(dim=0)
            avg_probs = F.softmax(stacked, dim=-1).mean(dim=0)
            V = avg_logits.size(-1)

            logit_nll = F.cross_entropy(avg_logits.reshape(-1, V), y_all.reshape(-1))
            soft_nll = -torch.log(
                avg_probs.reshape(-1, V).gather(1, y_all.reshape(-1, 1)).clamp(min=1e-12)
            ).mean()
            loss = (logit_nll + soft_nll) / 2

            sizes = [x.size(0) for x in xs]
            offsets = [sum(sizes[:j]) for j in range(N_)]

            kl_sum = torch.tensor(0.0, device=DEVICE)
            for s_idx, ((r, face), w) in enumerate(sampled):
                face_slices = [all_logits[s_idx][offsets[i]:offsets[i] + sizes[i]]
                               for i in face]
                student_logits_face = torch.cat(face_slices, dim=0)
                student_log_probs = F.log_softmax(student_logits_face / T, dim=-1)

                with torch.no_grad():
                    teacher_logits = torch.cat(
                        [vertex_models[i](xs[i])[0] for i in face], dim=0
                    )
                    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                kd_matched = F.kl_div(
                    student_log_probs.reshape(-1, V),
                    teacher_probs.reshape(-1, V),
                    reduction="batchmean",
                ) * (T * T)

                kl_sum = kl_sum + kd_matched

            loss = loss + kl_sum / N_CAL_SAMPLES
            (loss / n_chunks).backward()
            step_loss += loss.item() / n_chunks

        opt.step()
        loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss
        if it % 25 == 0:
            gate_str = " | " + " ".join(
                f"g{k}={math.exp(dg[k].item()):.2f}"
                for k in sorted(dg, key=lambda x: x if isinstance(x, int) else 0)
            )
            pbar.set_postfix_str(f"loss={loss_ema:.3f}{gate_str}")
