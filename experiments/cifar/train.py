from copy import deepcopy
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
from tqdm import tqdm

from nimbus.metrics import eval_metrics_logits, eval_metrics_probs
from nimbus.seeds import set_seed, device
from nimbus.simplex import (
    SimplexInterpolator, build_bump_params, sample_w_for_degree,
)

from .config import DISTILL_TEMPERATURE


def train_one_epoch(model, optimizer, scheduler, loss_fn, loader, task_labels):
    """
    train one epoch over `task_labels`. supports closure-based optimizers and
    steps the scheduler once at the end.
    """
    model.train()
    if hasattr(loader, "set_epoch_data"):
        loader.set_epoch_data(task_labels)
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        def closure():
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            return loss

        import inspect
        if "closure" in inspect.signature(optimizer.step).parameters:
            loss = optimizer.step(closure=closure)
        else:
            loss = closure()
            optimizer.step()
        losses.append(loss.item())
    if scheduler is not None:
        scheduler.step()
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_logits_model(model, test_loader, task_labels):
    """collect per-batch logits and labels for `model` over `task_labels`."""
    model = model.to(device).eval()
    dl = deepcopy(test_loader)
    dl.set_epoch_data(task_labels)
    logits_list, labels_list = [], []
    for x, y in dl:
        logits_list.append(model(x.to(device)).cpu())
        labels_list.append(y)
    return logits_list, labels_list


@torch.no_grad()
def collect_logits_ensemble(vertex_models, bump_params, alpha_to_key, alphas_by_degree,
                            test_loader, task_labels, n_samples=50, seed=42,
                            degree_gates=None, dirichlet_alpha=1.0, ws=None):
    """logit-average of the interpolated models at weights `ws` over `task_labels`."""
    N = len(vertex_models)
    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )
    interp.cache_bump_flat()
    if ws is None:
        rng = np.random.default_rng(seed)
        ws = [torch.from_numpy(rng.dirichlet(dirichlet_alpha * np.ones(N))).float().to(device)
              for _ in range(n_samples)]
    n = len(ws)
    dl = deepcopy(test_loader)
    dl.set_epoch_data(task_labels)
    logits_list, labels_list = [], []
    for x, y in dl:
        x = x.to(device)
        avg_logits = None
        for w in ws:
            params = interp(w)
            logits = functional_call(dummy, params, (x,))
            avg_logits = logits if avg_logits is None else avg_logits + logits
        logits_list.append((avg_logits / n).cpu())
        labels_list.append(y)
    return logits_list, labels_list


@torch.no_grad()
def collect_probs_ensemble(vertex_models, bump_params, alpha_to_key, alphas_by_degree,
                           test_loader, task_labels, n_samples=50, seed=42,
                           degree_gates=None, dirichlet_alpha=1.0, ws=None):
    """softmax-average of the interpolated models at weights `ws` over `task_labels`."""
    N = len(vertex_models)
    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )
    interp.cache_bump_flat()
    if ws is None:
        rng = np.random.default_rng(seed)
        ws = [torch.from_numpy(rng.dirichlet(dirichlet_alpha * np.ones(N))).float().to(device)
              for _ in range(n_samples)]
    n = len(ws)
    dl = deepcopy(test_loader)
    dl.set_epoch_data(task_labels)
    probs_list, labels_list = [], []
    for x, y in dl:
        x = x.to(device)
        avg_probs = None
        for w in ws:
            params = interp(w)
            probs = F.softmax(functional_call(dummy, params, (x,)), dim=-1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        probs_list.append((avg_probs / n).cpu())
        labels_list.append(y)
    return probs_list, labels_list


@torch.no_grad()
def average_simplex_model(vertex_models, bump_params, alpha_to_key, alphas_by_degree,
                          degree_gates=None, n_samples=500, seed=42, dirichlet_alpha=1.0):
    """build a model from the parameter average over n_samples dirichlet weights."""
    N = len(vertex_models)
    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)
    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )
    interp.cache_bump_flat()
    rng = np.random.default_rng(seed)
    avg_sd = None
    for _ in range(n_samples):
        w = torch.from_numpy(rng.dirichlet(dirichlet_alpha * np.ones(N))).float().to(device)
        params = interp(w)
        if avg_sd is None:
            avg_sd = {k: v.detach().clone() for k, v in params.items()}
        else:
            for k, v in params.items():
                if v.is_floating_point():
                    avg_sd[k].add_(v.detach())
    for v in avg_sd.values():
        if v.is_floating_point():
            v.div_(n_samples)
    for k, v in dummy.state_dict().items():
        if k not in avg_sd:
            avg_sd[k] = v.clone()
    m = deepcopy(vertex_models[0])
    m.load_state_dict(avg_sd, strict=True)
    return m.to(device).eval()


def train_simplex(vertex_models, train_loader, test_loader, task_split,
                  epochs, lr, seed, degrees,
                  share_by_degree=True, use_kl=True):
    """
    nimbus phase 1: fit the bezier bumps.

    trains bumps one degree at a time in ascending order, with cross-entropy plus
    optional KL distillation to the vertex teachers. gates for already-finalized
    degrees are trained alongside; the current degree's gate stays at init.
    constant lr.

    returns (bump_params, alpha_to_key, alphas_by_degree, degree_gates).
    """
    set_seed(seed)
    N = len(vertex_models)
    T = DISTILL_TEMPERATURE

    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)

    @torch.compile(mode="default")
    def compiled_forward(params, x):
        return functional_call(dummy, params, (x,))

    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]
    bump_params, alpha_to_key, alphas_by_degree = build_bump_params(
        vertex_models, degrees, share_by_degree=share_by_degree)

    degree_gates = {
        key: nn.Parameter(torch.tensor(0.0, device=device))
        for key in bump_params
    }

    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )

    param_size = sum(p.numel() for p in vertex_params[0].values())
    print(f"N={N}, degrees={degrees}")
    print(f"unique bump sets: {len(bump_params)}, params/set: {param_size:,}")
    for r in sorted(degrees):
        print(f"  degree {r}: {len(alphas_by_degree[r])} term(s)")

    dataloaders = []
    for tasks in task_split:
        train_loader.reset_shuffle_state()
        dl = deepcopy(train_loader)
        dl.set_epoch_data(tasks)
        dataloaders.append(dl)

    loss_fn = nn.CrossEntropyLoss()
    all_classes_local = [c.item() for t in task_split for c in t]

    degree_groups = [[r] for r in sorted(degrees)]
    trained_keys = set()

    for dg in degree_groups:
        rng = np.random.default_rng(seed + hash(tuple(dg)) % (2**31))

        trainable_keys = set()
        for r in dg:
            for alpha in alphas_by_degree[r]:
                trainable_keys.add(alpha_to_key[alpha])

        for key, pdict in bump_params.items():
            for p in pdict.values():
                p.requires_grad_(key in trainable_keys)

        for key, g in degree_gates.items():
            g.requires_grad_(key in trained_keys)

        trainable_params = [p for key in trainable_keys for p in bump_params[key].values()]
        gate_params = [g for g in degree_gates.values() if g.requires_grad]
        opt_groups = [{"params": trainable_params, "lr": lr}]
        if gate_params:
            opt_groups.append({"params": gate_params, "lr": lr})
        opt = optim.AdamW(opt_groups, weight_decay=0.0)

        all_faces = []
        for r in dg:
            for face in combinations(range(N), r):
                all_faces.append((r, face))

        n_faces_per_step = max(1, min(len(all_faces), N))
        do_subsample = n_faces_per_step < len(all_faces)

        n_train = sum(p.numel() for p in trainable_params)
        gate_info = f" + {len(gate_params)} gates" if gate_params else ""
        print(f"\n--- degrees {dg}: {n_train:,} trainable params{gate_info}, "
              f"{len(all_faces)} face(s), {n_faces_per_step} per step ---")

        log_every = max(1, epochs // 5)

        for epoch in tqdm(range(epochs), desc=f"deg {dg}"):
            iters = [iter(dl) for dl in dataloaders]
            total_loss, n_batches = 0.0, 0

            while True:
                batches = []
                for it in iters:
                    try:
                        batches.append(next(it))
                    except StopIteration:
                        batches = None
                        break
                if batches is None:
                    break

                xs = [b[0].to(device) for b in batches]
                ys = [b[1].to(device) for b in batches]

                if do_subsample:
                    fidx = rng.choice(len(all_faces), size=n_faces_per_step, replace=False)
                    step_faces = [all_faces[i] for i in fidx]
                else:
                    step_faces = all_faces

                opt.zero_grad(set_to_none=True)
                interp.cache_bump_flat()

                step_loss = 0.0
                for r, face in step_faces:
                    w, _ = sample_w_for_degree(N, r, rng=rng, device=device, face=face)
                    x_face = torch.cat([xs[i] for i in face])
                    y_face = torch.cat([ys[i] for i in face])
                    cp = interp(w)
                    student_logits = compiled_forward(cp, x_face)
                    face_loss = loss_fn(student_logits, y_face)

                    if use_kl:
                        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                        with torch.no_grad():
                            teacher_logits = torch.cat(
                                [vertex_models[i](xs[i]) for i in face], dim=0
                            )
                            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                        kd = F.kl_div(student_log_probs, teacher_probs,
                                      reduction="batchmean") * (T * T)
                        face_loss = face_loss + kd

                    face_loss = face_loss / len(step_faces)
                    face_loss.backward()
                    step_loss += face_loss.item()
                opt.step()
                total_loss += step_loss
                n_batches += 1

            if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    w_c = torch.full((N,), 1.0 / N, device=device)
                    interp.cache_bump_flat()
                    cp = interp(w_c)
                m = deepcopy(vertex_models[0])
                m.load_state_dict({k: v.detach().clone() for k, v in cp.items()}, strict=True)
                m = m.to(device).eval()
                lg, lb = collect_logits_model(m, test_loader, all_classes_local)
                acc_all = eval_metrics_logits(lg, lb)["acc"]
                accs = []
                for t in task_split:
                    lg_t, lb_t = collect_logits_model(m, test_loader, t)
                    accs.append(eval_metrics_logits(lg_t, lb_t)["acc"])
                parts = [f"T{i}={a:.3f}" for i, a in enumerate(accs)]
                gate_str = ""
                if gate_params:
                    gate_str = " | " + ", ".join(
                        f"g{k}={torch.exp(degree_gates[k]).item():.3f}"
                        for k in sorted(degree_gates.keys(),
                                        key=lambda x: x if isinstance(x, int) else 0))
                tqdm.write(f"  [ep {epoch+1}] loss={total_loss/max(n_batches,1):.4f} | "
                           f"all={acc_all:.4f} | {', '.join(parts)}{gate_str}")

        trained_keys.update(trainable_keys)

    return bump_params, alpha_to_key, alphas_by_degree, degree_gates


def calibrate_simplex(vertex_models, bump_params, alpha_to_key, alphas_by_degree,
                      degree_gates, train_loader, test_loader, task_split,
                      epochs, lr, seed, degrees,
                      n_samples=5, chunks=1, use_kl=True):
    """
    nimbus phase 2: calibrate the degree gates with bumps frozen.

    trains only the gates on an ensemble objective (mean of logit-average and
    softmax-average nll) plus optional matched KL distillation. constant lr.
    """
    N = len(vertex_models)
    set_seed(seed)
    T = DISTILL_TEMPERATURE

    dummy = deepcopy(vertex_models[0]).to(device).eval()
    for p in dummy.parameters():
        p.requires_grad_(False)

    @torch.compile(mode="default")
    def compiled_forward(params, x):
        return functional_call(dummy, params, (x,))

    vertex_params = [
        {n: p.detach().clone() for n, p in vm.named_parameters() if p.requires_grad}
        for vm in vertex_models
    ]

    interp = SimplexInterpolator(
        vertex_params, bump_params, alpha_to_key, alphas_by_degree, dummy,
        degree_gates=degree_gates,
    )

    for pdict in bump_params.values():
        for p in pdict.values():
            p.requires_grad_(False)
    for g in degree_gates.values():
        g.requires_grad_(True)

    trainable = list(degree_gates.values())
    if not trainable:
        print("  calibrate: no gates, skipping")
        return

    opt = optim.AdamW([{"params": trainable, "lr": lr}], weight_decay=0.0)

    all_faces = []
    for r in sorted(degrees):
        for face in combinations(range(N), r):
            all_faces.append((r, face))
    deg_counts = {}
    for r, _ in all_faces:
        deg_counts[r] = deg_counts.get(r, 0) + 1
    face_weights = np.array([1.0 / deg_counts[r] for r, _ in all_faces])
    face_weights /= face_weights.sum()

    dataloaders = []
    for tasks in task_split:
        train_loader.reset_shuffle_state()
        dl = deepcopy(train_loader)
        dl.set_epoch_data(tasks)
        dataloaders.append(dl)

    all_classes_local = [c.item() for t in task_split for c in t]
    rng = np.random.default_rng(seed + 777)

    n_train = sum(p.numel() for p in trainable)
    print(f"\n--- calibrate [nll+both, uniform_degree]: "
          f"{n_train:,} gate params, {epochs} ep, lr={lr}, {n_samples} samples/step ---")

    log_every = max(1, epochs // 5)

    for epoch in tqdm(range(epochs), desc="cal-gates"):
        iters = [iter(dl) for dl in dataloaders]
        total_loss, n_batches = 0.0, 0

        while True:
            batches = []
            for it in iters:
                try:
                    batches.append(next(it))
                except StopIteration:
                    batches = None
                    break
            if batches is None:
                break

            xs_full = [b[0].to(device) for b in batches]
            ys_full = [b[1].to(device) for b in batches]

            opt.zero_grad(set_to_none=True)
            interp.cache_bump_flat()

            face_indices = rng.choice(len(all_faces), size=n_samples, p=face_weights)
            sampled = []
            for fi in face_indices:
                r, face = all_faces[fi]
                w, _ = sample_w_for_degree(N, r, rng=rng, device=device, face=face)
                sampled.append(((r, face), w))

            x_chunks = [torch.chunk(x, chunks, dim=0) for x in xs_full]
            y_chunks = [torch.chunk(y, chunks, dim=0) for y in ys_full]
            n_chunks = len(x_chunks[0])

            step_loss = 0.0
            for c in range(n_chunks):
                xs = [x_chunks[i][c] for i in range(N)]
                ys = [y_chunks[i][c] for i in range(N)]
                x_all = torch.cat(xs)
                y_all = torch.cat(ys)

                all_logits = [compiled_forward(interp(w), x_all)
                              for (_, _), w in sampled]

                stacked = torch.stack(all_logits)
                avg_logits = stacked.mean(dim=0)
                avg_probs = F.softmax(stacked, dim=-1).mean(dim=0)

                logit_nll = F.cross_entropy(avg_logits, y_all)
                soft_nll = -torch.log(
                    avg_probs[torch.arange(len(y_all), device=device), y_all].clamp(min=1e-12)
                ).mean()
                loss = (logit_nll + soft_nll) / 2

                if use_kl:
                    sizes = [x.size(0) for x in xs]
                    offsets = [sum(sizes[:j]) for j in range(len(xs))]

                    kl_sum = torch.tensor(0.0, device=device)
                    for s_idx, ((r, face), w) in enumerate(sampled):
                        face_slices = [all_logits[s_idx][offsets[i]:offsets[i] + sizes[i]]
                                       for i in face]
                        student_logits_face = torch.cat(face_slices)
                        student_log_probs = F.log_softmax(student_logits_face / T, dim=-1)

                        with torch.no_grad():
                            teacher_logits = torch.cat(
                                [vertex_models[i](xs[i]) for i in face])
                            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                        kd_matched = F.kl_div(student_log_probs, teacher_probs,
                                              reduction="batchmean") * (T * T)

                        kl_sum = kl_sum + kd_matched

                    loss = loss + kl_sum / n_samples

                (loss / n_chunks).backward()
                step_loss += loss.item() / n_chunks

            opt.step()
            total_loss += step_loss
            n_batches += 1

        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                w_c = torch.full((N,), 1.0 / N, device=device)
                interp.cache_bump_flat()
                cp = interp(w_c)
            m = deepcopy(vertex_models[0])
            m.load_state_dict({k: v.detach().clone() for k, v in cp.items()}, strict=True)
            m = m.to(device).eval()
            lg, lb = collect_logits_model(m, test_loader, all_classes_local)
            acc_all = eval_metrics_logits(lg, lb)["acc"]
            accs = []
            for t in task_split:
                lg_t, lb_t = collect_logits_model(m, test_loader, t)
                accs.append(eval_metrics_logits(lg_t, lb_t)["acc"])
            parts = [f"T{i}={a:.3f}" for i, a in enumerate(accs)]
            gate_str = " | " + ", ".join(
                f"g{k}={torch.exp(degree_gates[k]).item():.3f}"
                for k in sorted(degree_gates.keys(),
                                key=lambda x: x if isinstance(x, int) else 0))
            tqdm.write(f"  [cal ep {epoch+1}] loss={total_loss/max(n_batches,1):.4f} | "
                       f"all={acc_all:.4f} | {', '.join(parts)}{gate_str}")


def eval_simplex_all(vertex_models, bp, a2k, abd, dg,
                     test_loader, all_classes, n_ens, n_face_ens,
                     N, degrees, seed, task_split):
    """evaluate the face-pool logit/softmax ensembles and the parameter average."""
    from nimbus.ensemble import sample_face_ws

    results = {}
    ws = sample_face_ws(N, degrees, n_face_ens, seed=seed, device=device)
    fl, fy = collect_logits_ensemble(
        vertex_models, bp, a2k, abd,
        test_loader, all_classes, degree_gates=dg, ws=ws)
    fp, fpy = collect_probs_ensemble(
        vertex_models, bp, a2k, abd,
        test_loader, all_classes, degree_gates=dg, ws=ws)
    results["logit ens uniform degree"] = eval_metrics_logits(fl, fy, task_split=task_split)
    results["softmax ens uniform degree"] = eval_metrics_probs(fp, fpy, task_split=task_split)
    avg_m = average_simplex_model(vertex_models, bp, a2k, abd,
                                  degree_gates=dg, n_samples=500, seed=seed)
    avg_l, avg_y = collect_logits_model(avg_m, test_loader, all_classes)
    results["param avg"] = eval_metrics_logits(avg_l, avg_y, task_split=task_split)
    del avg_m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def log_eval(label, results, dg):
    """print all-scope metrics for each result block, with gate values."""
    gate_str = " | gates: " + ", ".join(
        f"g{k}={torch.exp(dg[k]).item():.3f}"
        for k in sorted(dg.keys(), key=lambda x: x if isinstance(x, int) else 0))
    print(label + gate_str)
    for name, by_scope in results.items():
        v = by_scope["all"]
        print(f"  {name:<26}: acc={v['acc']:.3f}  top3={v['top3']:.3f}  "
              f"top5={v['top5']:.3f}  nll={v['nll']:.3f}  ece={v['ece']:.3f}")
