"""LoRA-aware simplex interpolation for the ViT leg.

assembles each per-key flat bump vector by calling bo.materialize() (rebuilding
B @ A) instead of reshaping a stored dense tensor. everything else (vertex
blend, log-space degree coefficients, gates) is inherited unchanged.
"""
import torch

from nimbus.simplex import SimplexInterpolator


class LoRASimplexInterpolator(SimplexInterpolator):
    def cache_bump_flat(self):
        self._bump_cache = torch.stack([
            torch.cat([bo.materialize().reshape(-1)
                       for bo in self.bump_params[key].values()])
            for key in self.key_list
        ])

    def __call__(self, w):
        # re-materialize B @ A on every call so each face's backward() owns its
        # own autograd graph.
        self.cache_bump_flat()
        return super().__call__(w)
