import torch


@torch.no_grad()
def perturb_input_l2(x, epsilon):
    """add gaussian noise per image, scaled to ||noise|| = epsilon * ||x||."""
    if epsilon <= 0.0:
        return x
    flat = x.reshape(x.size(0), -1)
    x_norm = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    noise = torch.randn_like(flat)
    noise_norm = noise.norm(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (epsilon * x_norm / noise_norm).reshape(-1, *([1] * (x.dim() - 1)))
    return x + scale * noise.reshape_as(x)
