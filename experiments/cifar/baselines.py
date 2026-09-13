from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
from tqdm import tqdm

from nimbus.ensemble import (
    flatten_trainable, unflatten_trainable, weight_average_params,
)
from nimbus.metrics import scope_keys
from nimbus.robustness import (
    empty_sweep_results, finalize_sweep, merge_trial, merge_trial_stats,
    perturb_weight_vector, sweep_one_pass,
)
from nimbus.seeds import set_seed, device

from .perturbations import perturb_input_l2


class ASAM(optim.Optimizer):
    """adaptive sharpness-aware minimization wrapping a base optimizer."""

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._asam_norm()
        rho = self.param_groups[0]["rho"]
        scale = rho / (grad_norm + 1e-12)
        params = self.param_groups[0]["params"]
        for p in params:
            if p.grad is None:
                continue
            self.state[p]["old_p"] = p.data.clone()
            e_w = p.data.abs() * p.grad * scale.to(p)
            p.data.add_(e_w)
        self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        params = self.param_groups[0]["params"]
        for p in params:
            state = self.state.get(p, None)
            if (state is not None) and ("old_p" in state):
                p.data.copy_(state["old_p"])
                del state["old_p"]
        self.base_optimizer.step()
        self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "ASAM requires a closure"
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step()
        loss = closure()
        self.second_step()
        return loss

    @torch.no_grad()
    def _asam_norm(self):
        params = self.param_groups[0]["params"]
        dev = params[0].device
        chunks = []
        for p in params:
            if p.grad is None:
                continue
            chunks.append((p.data.abs() * p.grad).norm(p=2).to(dev))
        if not chunks:
            return torch.tensor(0.0, device=dev)
        return torch.norm(torch.stack(chunks), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def compute_diagonal_fisher(model, train_loader, task_labels, n_batches=50):
    """diagonal fisher of `model` on `task_labels` from squared grads over up to n_batches."""
    model = deepcopy(model).to(device).train()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    dl = deepcopy(train_loader)
    dl.set_epoch_data(task_labels)
    n = 0
    for x, y in dl:
        if n >= n_batches:
            break
        model.zero_grad()
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach().pow(2)
        n += 1
    for name in fisher:
        fisher[name] /= max(n, 1)
    return fisher


def fisher_merge(vertex_models, train_loader, task_split, n_batches=50):
    """fisher-weighted parameter average of the vertex models."""
    N = len(vertex_models)
    print("computing diagonal fishers...")
    fishers = [compute_diagonal_fisher(vm, train_loader, t.tolist(), n_batches)
               for vm, t in zip(vertex_models, task_split)]

    sd0 = vertex_models[0].state_dict()
    merged_sd = {}
    for name, val in sd0.items():
        if name in fishers[0] and val.is_floating_point():
            num = sum(fishers[i][name] * vertex_models[i].state_dict()[name].detach()
                      for i in range(N))
            den = sum(fishers[i][name] for i in range(N)).clamp(min=1e-8)
            merged_sd[name] = num / den
        else:
            merged_sd[name] = val.clone()

    m = deepcopy(vertex_models[0])
    m.load_state_dict(merged_sd, strict=True)
    return m.to(device).eval()


def ties_merge(vertex_models, keep_fraction=0.8):
    """
    TIES merge: task vectors relative to the vertex mean, trimmed to the top
    magnitudes, sign-elected, then disjoint-averaged.
    """
    N = len(vertex_models)
    sds = [vm.state_dict() for vm in vertex_models]

    base_sd = {}
    for name, val in sds[0].items():
        if val.is_floating_point():
            base_sd[name] = sum(sd[name].detach().float() for sd in sds) / N

    merged_sd = {name: val.clone() for name, val in sds[0].items()}

    for name, base in base_sd.items():
        tvecs = torch.stack([sds[i][name].detach().float() - base for i in range(N)], dim=0)

        flat = tvecs.reshape(N, -1)
        thresh = torch.quantile(flat.abs(), 1.0 - keep_fraction, dim=1)
        for _ in range(tvecs.dim() - 1):
            thresh = thresh.unsqueeze(-1)
        trimmed = tvecs * (tvecs.abs() >= thresh).float()

        sign_sum = trimmed.sign().sum(dim=0)
        elected = sign_sum.sign()
        elected[elected == 0] = 1.0

        agree = (trimmed.sign() == elected.unsqueeze(0)) | (trimmed == 0)
        disjoint = trimmed * agree.float()

        counts = (disjoint != 0).float().sum(dim=0).clamp(min=1)
        merged_tv = disjoint.sum(dim=0) / counts
        merged_sd[name] = (base + merged_tv).to(sds[0][name].dtype)

    m = deepcopy(vertex_models[0])
    m.load_state_dict(merged_sd, strict=True)
    return m.to(device).eval()


def train_pcgrad(model, train_loader, test_loader, task_split, epochs, lr, seed):
    """PCGrad: project conflicting task gradients apart before each step. constant lr."""
    set_seed(seed)
    N = len(task_split)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    dataloaders = []
    for tasks in task_split:
        train_loader.reset_shuffle_state()
        dl = deepcopy(train_loader)
        dl.set_epoch_data(tasks)
        dataloaders.append(dl)

    param_list = [p for p in model.parameters() if p.requires_grad]

    for epoch in tqdm(range(epochs), desc="PCGrad"):
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

            task_grads = []
            task_loss_sum = 0.0
            for x, y in batches:
                opt.zero_grad()
                loss = loss_fn(model(x.to(device)), y.to(device))
                loss.backward()
                task_grads.append(
                    torch.cat([p.grad.detach().flatten() for p in param_list])
                )
                task_loss_sum += loss.item()

            pc_grads = [g.clone() for g in task_grads]
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    dot = torch.dot(pc_grads[i], task_grads[j])
                    if dot < 0:
                        norm_sq = task_grads[j].pow(2).sum().clamp(min=1e-8)
                        pc_grads[i] = pc_grads[i] - (dot / norm_sq) * task_grads[j]

            final_grad = sum(pc_grads)
            opt.zero_grad()
            offset = 0
            for p in param_list:
                n = p.numel()
                p.grad = final_grad[offset:offset + n].reshape(p.shape).clone()
                offset += n
            opt.step()

            total_loss += task_loss_sum / N
            n_batches += 1

    return model.eval()


def train_swa(model, train_loader, all_classes, epochs, lr, swa_start_epoch=75):
    """SWA: constant-lr training, averaging state snapshots from swa_start_epoch on."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    snapshots = []
    train_loader.reset_shuffle_state()
    for ep in tqdm(range(epochs), desc="SWA"):
        model.train()
        train_loader.set_epoch_data(all_classes)
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        if ep >= swa_start_epoch:
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    avg_sd = weight_average_params(snapshots)
    model.load_state_dict(avg_sd, strict=True)
    return model.to(device).eval()


def train_asam(model, train_loader, all_classes, epochs, lr, rho=0.1):
    """ASAM training at constant lr."""
    opt = ASAM(model.parameters(), base_optimizer_cls=optim.AdamW, lr=lr, rho=rho, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    train_loader.reset_shuffle_state()
    for _ in tqdm(range(epochs), desc="ASAM"):
        model.train()
        train_loader.set_epoch_data(all_classes)
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)

            def closure():
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                return loss

            opt.step(closure=closure)
    return model.eval()


def cifar_forward(dummy, pd, x, **kwargs):
    """functional forward of `dummy` with param dict `pd`."""
    return functional_call(dummy, pd, (x,))


def cifar_batches(test_loader, task_labels, task_split):
    """yield (x, y, task_id) test batches; task_id is None when task_split is None."""
    dl = deepcopy(test_loader)
    dl.set_epoch_data(list(task_labels))
    if task_split is None:
        for x, y in dl:
            yield x.to(device), y.to(device), None
        return
    N = len(task_split)
    cls_to_task = {}
    for ti in range(N):
        for c in task_split[ti]:
            ci = int(c.item()) if hasattr(c, "item") else int(c)
            cls_to_task[ci] = ti
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        t = torch.tensor([cls_to_task[int(yy.item())] for yy in y],
                         device=y.device, dtype=torch.long)
        yield x, y, t


def _build_avg_pd(param_dicts, dummy):
    """weight-average of param dicts, filling missing keys from `dummy`."""
    sd0 = dummy.state_dict()
    avg = weight_average_params(param_dicts)
    out = dict(avg)
    for k, v in sd0.items():
        if k not in out:
            out[k] = v.clone()
    return out


@torch.no_grad()
def noisy_input_sweep(param_dicts, dummy, test_loader, task_labels,
                      epsilons, n_trials=3, seed=42, task_split=None):
    """sweep input-noise epsilons, returning mean and std metric sweeps over n_trials."""
    N = len(task_split) if task_split is not None else 0
    scopes = scope_keys(N)
    avg_pd = _build_avg_pd(param_dicts, dummy)
    results_mean = empty_sweep_results(scopes)
    results_std = empty_sweep_results(scopes)

    for eps in tqdm(epsilons, desc="input noise"):
        trial_acc = empty_sweep_results(scopes)
        for trial in range(n_trials):
            set_seed(seed + hash((float(eps), trial)) % (2**32))

            def transform(x, y):
                return perturb_input_l2(x, float(eps)), y

            batches = cifar_batches(test_loader, task_labels, task_split)
            trial_one = sweep_one_pass(
                pool_pds=param_dicts,
                weight_avg_pd=avg_pd,
                dummy=dummy,
                batches=batches,
                forward_fn=cifar_forward,
                scopes=scopes,
                batch_transform=transform,
            )
            wrapped = {m: {s: {met: [trial_one[m][s][met]] for met in trial_one[m][s]}
                           for s in scopes} for m in trial_one}
            merge_trial(trial_acc, wrapped, scopes)

        merge_trial_stats(results_mean, results_std, trial_acc, scopes)

    return finalize_sweep(results_mean), finalize_sweep(results_std)


@torch.no_grad()
def noisy_weight_sweep(param_dicts, dummy, test_loader, task_labels,
                       epsilons, n_trials=3, seed=42, task_split=None):
    """sweep weight-noise epsilons, returning mean and std metric sweeps over n_trials."""
    N = len(task_split) if task_split is not None else 0
    scopes = scope_keys(N)
    trainable_names = [n for n, _ in dummy.named_parameters()]
    shapes = [dummy.state_dict()[n].shape for n in trainable_names]
    sizes = [dummy.state_dict()[n].numel() for n in trainable_names]
    avg_pd = _build_avg_pd(param_dicts, dummy)
    results_mean = empty_sweep_results(scopes)
    results_std = empty_sweep_results(scopes)

    for eps in tqdm(epsilons, desc="weight noise"):
        trial_acc = empty_sweep_results(scopes)
        for trial in range(n_trials):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + hash((float(eps), trial)) % (2**32))

            perturbed_pds = []
            for pd in param_dicts:
                flat = flatten_trainable(pd, trainable_names)
                pf = perturb_weight_vector(flat, float(eps), generator=gen)
                perturbed_pds.append(unflatten_trainable(pf, pd, trainable_names, shapes, sizes))

            avg_flat = flatten_trainable(avg_pd, trainable_names)
            avg_pf = perturb_weight_vector(avg_flat, float(eps), generator=gen)
            perturbed_avg = unflatten_trainable(avg_pf, avg_pd, trainable_names, shapes, sizes)

            batches = cifar_batches(test_loader, task_labels, task_split)
            trial_one = sweep_one_pass(
                pool_pds=perturbed_pds,
                weight_avg_pd=perturbed_avg,
                dummy=dummy,
                batches=batches,
                forward_fn=cifar_forward,
                scopes=scopes,
            )
            wrapped = {m: {s: {met: [trial_one[m][s][met]] for met in trial_one[m][s]}
                           for s in scopes} for m in trial_one}
            merge_trial(trial_acc, wrapped, scopes)

        merge_trial_stats(results_mean, results_std, trial_acc, scopes)

    return finalize_sweep(results_mean), finalize_sweep(results_std)
