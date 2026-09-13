"""LoRA-parameterized correction bumps.

targeted 2D weight matrices get a low-rank bump B @ A (B is [out, r], A is
[r, in]); 1D params and non-targeted matrices keep a dense bump
nn.Parameter(torch.zeros_like(p)). vertex models are untouched.

each bump wraps a uniform interface:
  materialize() -> full-shape tensor (the delta added to the vertex blend)
  params()      -> the nn.Parameter(s) to optimize
  requires_grad_(flag)
  state()       -> cpu state for caching
"""
import torch
import torch.nn as nn

from nimbus.simplex import enumerate_alphas

from .config import LORA_TARGET_SUFFIXES, LORA_INIT_STD


def is_lora_target(name, param, target_suffixes=LORA_TARGET_SUFFIXES):
    """
    a bump is LoRA iff the param is a 2D matrix with a targeted name.

    target_suffixes (tuple) defaults to the ViT/CLIP vision-tower matrices.
    """
    return param.dim() == 2 and name.endswith(tuple(target_suffixes))


class DenseBump:
    """full-shape dense bump."""

    kind = "dense"

    def __init__(self, weight):
        self.weight = weight  # nn.Parameter

    @classmethod
    def zeros(cls, shape, device):
        return cls(nn.Parameter(torch.zeros(shape, device=device)))

    def materialize(self):
        return self.weight

    def params(self):
        return [self.weight]

    def requires_grad_(self, flag):
        self.weight.requires_grad_(flag)
        return self

    def state(self):
        return {"kind": "dense", "weight": self.weight.data.detach().cpu().clone()}

    def clone(self):
        return DenseBump(nn.Parameter(self.weight.data.clone()))


class LoRABump:
    """
    low-rank bump B @ A for a [out, in] matrix.

    B inits to zero and A to small normal noise, so the bump is 0 at init.
    """

    kind = "lora"

    def __init__(self, B, A):
        self.B = B  # [out, r]
        self.A = A  # [r, in]

    @classmethod
    def init(cls, shape, r, device, init_std=LORA_INIT_STD):
        out, in_ = shape
        B = nn.Parameter(torch.zeros(out, r, device=device))
        A = nn.Parameter(torch.empty(r, in_, device=device).normal_(0.0, init_std))
        return cls(B, A)

    def materialize(self):
        return self.B @ self.A

    def params(self):
        return [self.B, self.A]

    def requires_grad_(self, flag):
        self.B.requires_grad_(flag)
        self.A.requires_grad_(flag)
        return self

    @property
    def rank(self):
        return self.B.shape[1]

    def state(self):
        return {
            "kind": "lora",
            "B": self.B.data.detach().cpu().clone(),
            "A": self.A.data.detach().cpu().clone(),
        }

    def clone(self):
        return LoRABump(nn.Parameter(self.B.data.clone()),
                        nn.Parameter(self.A.data.clone()))


def build_bump_params_lora(vertex_models, degrees, r, device, share_by_degree=True,
                           target_suffixes=LORA_TARGET_SUFFIXES):
    """
    LoRA analogue of nimbus.simplex.build_bump_params.

    each bump_params[key][name] is a DenseBump or LoRABump; the (key, name)
    layout matches the dense version, so the interpolator sees an identical flat
    param map. target_suffixes (tuple) selects which 2D matrices are low-rank.

    returns (bump_params, alpha_to_key, alphas_by_degree).
    """
    N = len(vertex_models)
    template = [(n, p) for n, p in vertex_models[0].named_parameters() if p.requires_grad]

    alphas_by_degree = {rr: enumerate_alphas(N, rr) for rr in sorted(degrees)}
    all_alphas = [a for rr in sorted(degrees) for a in alphas_by_degree[rr]]
    if share_by_degree:
        alpha_to_key = {a: sum(a) for a in all_alphas}
    else:
        alpha_to_key = {a: a for a in all_alphas}
    unique_keys = list(dict.fromkeys(alpha_to_key[a] for a in all_alphas))

    def make_bump(name, p):
        if is_lora_target(name, p, target_suffixes):
            return LoRABump.init(tuple(p.shape), r, device)
        return DenseBump.zeros(p.shape, device)

    bump_params = {
        key: {n: make_bump(n, p) for n, p in template}
        for key in unique_keys
    }
    return bump_params, alpha_to_key, alphas_by_degree


def bp_lora_to_state(bp):
    return {k: {n: bo.state() for n, bo in pd.items()} for k, pd in bp.items()}


def bp_lora_from_state(state, device):
    out = {}
    for k, pd in state.items():
        out[k] = {}
        for n, st in pd.items():
            if st["kind"] == "lora":
                out[k][n] = LoRABump(nn.Parameter(st["B"].to(device)),
                                     nn.Parameter(st["A"].to(device)))
            else:
                out[k][n] = DenseBump(nn.Parameter(st["weight"].to(device)))
    return out


def count_bump_params(bp):
    """total trainable scalars across one full set of bumps (all keys)."""
    return sum(p.numel() for pd in bp.values() for bo in pd.values() for p in bo.params())
