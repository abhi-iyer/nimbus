"""classification metrics (accuracy, top-k, nll, ece), scoped by task."""

import torch
import torch.nn.functional as F


METRICS = ("acc", "top3", "top5", "nll", "ece")
ECE_N_BINS = 15

NAN_METRICS = {k: float("nan") for k in METRICS}


def scope_keys(N):
    """scope names: 'all' plus one per task."""
    return ["all"] + [f"task_{i}" for i in range(N)]


def scope_mask(labels, task_cls):
    """boolean mask of labels belonging to the given task classes."""
    task_set = torch.tensor([int(c) for c in task_cls],
                            device=labels.device, dtype=labels.dtype)
    return (labels.unsqueeze(-1) == task_set.unsqueeze(0)).any(dim=-1)


@torch.no_grad()
def compute_ece(confidences, correctness, n_bins=ECE_N_BINS):
    """expected calibration error over n_bins equal-width confidence bins."""
    bin_bounds = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidences >= bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum().float() * (confidences[m].mean() - correctness[m].mean()).abs()
    return (ece / len(confidences)).item()


@torch.no_grad()
def metrics_from_logits(logits, labels):
    """acc, top3, top5, nll, ece from logits."""
    if labels.numel() == 0:
        return dict(NAN_METRICS)
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    nll_tok = F.cross_entropy(logits, labels, reduction="none")
    nll = nll_tok.mean().item()
    ece = compute_ece(conf, correct)
    top3 = (labels.unsqueeze(-1) == logits.topk(3, dim=-1).indices).any(-1).float().mean().item()
    top5 = (labels.unsqueeze(-1) == logits.topk(5, dim=-1).indices).any(-1).float().mean().item()
    return {"acc": correct.mean().item(), "top3": top3, "top5": top5, "nll": nll, "ece": ece}


@torch.no_grad()
def metrics_from_probs(probs, labels):
    """acc, top3, top5, nll, ece from probabilities."""
    if labels.numel() == 0:
        return dict(NAN_METRICS)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    nll_tok = -probs[torch.arange(len(labels)), labels].clamp(min=1e-12).log()
    nll = nll_tok.mean().item()
    ece = compute_ece(conf, correct)
    top3 = (labels.unsqueeze(-1) == probs.topk(3, dim=-1).indices).any(-1).float().mean().item()
    top5 = (labels.unsqueeze(-1) == probs.topk(5, dim=-1).indices).any(-1).float().mean().item()
    return {"acc": correct.mean().item(), "top3": top3, "top5": top5, "nll": nll, "ece": ece}


@torch.no_grad()
def eval_metrics_logits(logits_list, labels_list, task_split=None):
    """metrics over concatenated logits, optionally split per task."""
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    all_mets = metrics_from_logits(logits, labels)
    if task_split is None:
        return all_mets
    out = {"all": all_mets}
    for ti, cls in enumerate(task_split):
        m = scope_mask(labels, cls.tolist() if hasattr(cls, "tolist") else cls)
        out[f"task_{ti}"] = metrics_from_logits(logits[m], labels[m])
    return out


@torch.no_grad()
def eval_metrics_probs(probs_list, labels_list, task_split=None):
    """metrics over concatenated probabilities, optionally split per task."""
    probs = torch.cat(probs_list)
    labels = torch.cat(labels_list)
    all_mets = metrics_from_probs(probs, labels)
    if task_split is None:
        return all_mets
    out = {"all": all_mets}
    for ti, cls in enumerate(task_split):
        m = scope_mask(labels, cls.tolist() if hasattr(cls, "tolist") else cls)
        out[f"task_{ti}"] = metrics_from_probs(probs[m], labels[m])
    return out


def new_metric_state(device):
    """fresh streaming-accumulation state for one scope."""
    return {
        "nll_sum": 0.0,
        "correct1": 0,
        "correct3": 0,
        "correct5": 0,
        "n_tok": 0,
        "bin_conf":  torch.zeros(ECE_N_BINS, device=device),
        "bin_acc":   torch.zeros(ECE_N_BINS, device=device),
        "bin_count": torch.zeros(ECE_N_BINS, device=device),
        "_edges":    torch.linspace(0.0, 1.0, ECE_N_BINS + 1, device=device),
    }


def new_scoped_states(scopes, device):
    """one metric state per scope."""
    return {s: new_metric_state(device) for s in scopes}


def finalize_metrics(state):
    """reduce an accumulation state to acc, top3, top5, nll, ece."""
    n = max(1, state["n_tok"])
    nll = state["nll_sum"] / n
    bin_count = state["bin_count"]
    total = bin_count.sum().clamp(min=1)
    ece = 0.0
    for b in range(ECE_N_BINS):
        c = bin_count[b].item()
        if c > 0:
            avg_conf = state["bin_conf"][b].item() / c
            avg_acc  = state["bin_acc"][b].item() / c
            ece += (c / total.item()) * abs(avg_conf - avg_acc)
    return {
        "nll":  nll,
        "ece":  ece,
        "acc":  state["correct1"] / n,
        "top3": state["correct3"] / n,
        "top5": state["correct5"] / n,
    }


def finalize_scoped(states, any_batch):
    """finalize every scope, or nan metrics if no batch was seen."""
    return {s: (finalize_metrics(st) if any_batch else dict(NAN_METRICS))
            for s, st in states.items()}


def accumulate_into(logits, y, state, mask=None):
    """accumulate one logit batch into a metric state."""
    V = logits.size(-1)
    logits = logits.reshape(-1, V)
    y = y.reshape(-1)
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.sum().item() == 0:
            return
        logits = logits[mask]
        y = y[mask]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
    probs = log_probs.exp()
    max_p, pred = probs.max(dim=-1)
    correct = (pred == y)
    k5 = min(5, V)
    top5_idx = probs.topk(k5, dim=-1).indices
    match = (top5_idx == y.unsqueeze(1))
    correct3 = match[:, :min(3, k5)].any(dim=-1)
    correct5 = match.any(dim=-1)
    state["nll_sum"]  += nll.sum().item()
    state["correct1"] += correct.sum().item()
    state["correct3"] += correct3.sum().item()
    state["correct5"] += correct5.sum().item()
    state["n_tok"]    += y.numel()
    edges = state["_edges"]
    bidx = torch.bucketize(max_p, edges[1:-1]).clamp(max=ECE_N_BINS - 1)
    state["bin_conf"].scatter_add_(0,  bidx, max_p.float())
    state["bin_acc"].scatter_add_(0,   bidx, correct.float())
    state["bin_count"].scatter_add_(0, bidx, torch.ones_like(max_p, dtype=torch.float32))


def accumulate_from_probs_into(probs, y, state, mask=None):
    """accumulate one probability batch into a metric state."""
    V = probs.size(-1)
    probs = probs.reshape(-1, V).clamp(min=1e-12)
    y = y.reshape(-1)
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.sum().item() == 0:
            return
        probs = probs[mask]
        y = y[mask]
    max_p, pred = probs.max(dim=-1)
    correct = (pred == y)
    nll = -probs.gather(1, y.unsqueeze(1)).squeeze(1).log()
    k5 = min(5, V)
    top5_idx = probs.topk(k5, dim=-1).indices
    match = (top5_idx == y.unsqueeze(1))
    correct3 = match[:, :min(3, k5)].any(dim=-1)
    correct5 = match.any(dim=-1)
    state["nll_sum"]  += nll.sum().item()
    state["correct1"] += correct.sum().item()
    state["correct3"] += correct3.sum().item()
    state["correct5"] += correct5.sum().item()
    state["n_tok"]    += y.numel()
    edges = state["_edges"]
    bidx = torch.bucketize(max_p, edges[1:-1]).clamp(max=ECE_N_BINS - 1)
    state["bin_conf"].scatter_add_(0,  bidx, max_p.float())
    state["bin_acc"].scatter_add_(0,   bidx, correct.float())
    state["bin_count"].scatter_add_(0, bidx, torch.ones_like(max_p, dtype=torch.float32))


def accumulate_scoped(logits_or_probs, y, task_ids, states, from_probs=False):
    """accumulate into the 'all' scope and each task scope by task id."""
    fn = accumulate_from_probs_into if from_probs else accumulate_into
    fn(logits_or_probs, y, states["all"])
    if task_ids is None:
        return
    for scope, st in states.items():
        if scope == "all":
            continue
        ti = int(scope.split("_")[1])
        mask = (task_ids == ti)
        fn(logits_or_probs, y, st, mask=mask)
