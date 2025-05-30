import torch
import torch.nn.functional as F

def sqnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    signal_power = torch.sum(b_flat ** 2, dim = 1)
    noise_power = torch.sum((a_flat - b_flat) ** 2, dim = 1) + eps
    return 10.0 * torch.log10(signal_power / noise_power).mean().item() # dB

def cosine_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    cos_sim = F.cosine_similarity(a_flat, b_flat, dim = 1, eps = eps)
    return 1.0 - cos_sim.mean().item()
