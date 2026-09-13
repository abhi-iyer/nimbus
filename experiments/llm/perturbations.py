"""token-level input corruptions for the LLM leg."""

import torch


@torch.no_grad()
def corrupt_tokens(idx, p, vocab_size, generator=None):
    """replace each token with probability p by a uniform random token."""
    if p <= 0.0:
        return idx
    mask = torch.rand(idx.shape, device=idx.device, generator=generator) < p
    rand = torch.randint(0, vocab_size, idx.shape, device=idx.device,
                         generator=generator, dtype=idx.dtype)
    return torch.where(mask, rand, idx)
