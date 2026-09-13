"""CLIP ViT-B/32 backbone, zero-shot heads, and the functional-call forward.

interpolation operates over the vision tower (vision_model.*) only. the
visual_projection, logit_scale, and per-task zero-shot heads are loaded once
from base CLIP and held fixed/shared across all vertices. the forward is:

    pooled  = vision_tower(pixel_values).pooler_output      # [B, 768]   (interpolated)
    z       = normalize(pooled @ projection.T)              # [B, 512]   (fixed proj)
    logits  = logit_scale * z @ head_t.T                    # [B, C_t]   (fixed head_t)

encode() goes through functional_call with the interpolated param dict;
head_logits() is the fixed tail, applied per task.
"""
import warnings
from copy import deepcopy

import torch
from torch.func import functional_call

from nimbus.seeds import device

from .config import (
    CLIP_BASE, TASK_SPECS, HEAD_DIR,
)


_BASE = None


def _load_base():
    """
    load base CLIP (text encoder, tokenizer, visual_projection, logit_scale)
    once, cached as a module-level singleton.
    """
    global _BASE
    if _BASE is not None:
        return _BASE
    from transformers import CLIPModel, CLIPTokenizerFast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip = CLIPModel.from_pretrained(CLIP_BASE, use_safetensors=True).to(device).eval()
        tok = CLIPTokenizerFast.from_pretrained(CLIP_BASE)
    for p in clip.parameters():
        p.requires_grad_(False)
    _BASE = {
        "clip": clip,
        "tokenizer": tok,
        "projection": clip.visual_projection.weight.detach().clone(),  # [512, 768]
        "scale": float(clip.logit_scale.exp().item()),                 # 100.0
    }
    return _BASE


def base_projection_scale():
    """the fixed (projection [512,768], logit_scale float) shared by all vertices."""
    b = _load_base()
    return b["projection"], b["scale"]


# --- zero-shot heads ---

@torch.no_grad()
def build_zero_shot_head(task_name, classnames):
    """[C, 512] L2-normalized text-embedding head for task_name, cached."""
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    cache = HEAD_DIR / f"head_{task_name}.pt"
    if cache.exists():
        d = torch.load(cache, map_location=device)
        if d["classnames"] == list(classnames):
            return d["head"].to(device)
    b = _load_base()
    template = TASK_SPECS[task_name]["template"]
    prompts = [template.format(c) for c in classnames]
    toks = b["tokenizer"](prompts, padding=True, return_tensors="pt").to(device)
    W = b["clip"].get_text_features(**toks)
    W = W / W.norm(dim=-1, keepdim=True)          # [C, 512]
    torch.save({"head": W.detach().cpu(), "classnames": list(classnames)}, cache)
    return W


def make_dummy(vertex=None):
    """a frozen vision tower for functional_call (params overridden per call)."""
    vt = deepcopy(vertex) if vertex is not None else None
    if vt is None:
        raise ValueError("make_dummy needs a template vertex")
    vt = vt.to(device).eval()
    for p in vt.parameters():
        p.requires_grad_(False)
    return vt


def vision_param_dict(model):
    """interpolated param dict = the whole vision tower (detached clones)."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# --- forward ---

def encode(dummy, param_dict, pixel_values):
    """interpolated vision-tower forward -> pooled CLS embedding [B, 768]."""
    out = functional_call(dummy, param_dict, (pixel_values,))
    return out.pooler_output


def head_logits(pooled, head, projection, scale):
    """fixed tail: project pooled -> 512, normalize, cosine-sim against head."""
    z = pooled @ projection.t()                   # [B, 512]
    z = z / z.norm(dim=-1, keepdim=True)
    return scale * z @ head.t()                    # [B, C]
