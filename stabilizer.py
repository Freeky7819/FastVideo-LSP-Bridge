import torch

def temporal_cosine_drift(latents: torch.Tensor) -> torch.Tensor:
    """
    Meri "drsenje" faze med zaporednimi latentnimi vektorji.
    latents: [B, T, D] (ali [T, D]) v float32.
    Vrne per-batch drift metriko (več = slabše).
    """
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)  # [1, T, D]
    v = torch.nn.functional.normalize(latents[:, 1:] - latents[:, :-1], dim=-1)
    c = (v[:, 1:] * v[:, :-1]).sum(dim=-1).clamp(-1, 1)
    drift = (1.0 - c).mean(dim=-1)
    return drift  # brez gradienta; za diagnostiko


def log_periodic_stabilizer(latents: torch.Tensor, freq: float = 5.5, weight: float = 0.01) -> torch.Tensor:
    """
    Penalizira varianco latentov v log-časovni fazi (mehka poravnava gibanja).
    latents: [B, T, D]
    freq: frekvenca ω (navadna spremenljivka)
    weight: utež λ
    """
    assert latents.dim() == 3, "expected [B, T, D]"
    B, T, D = latents.shape
    device = latents.device
    t = torch.arange(T, device=device, dtype=latents.dtype)
    phase = torch.sin(freq * torch.log1p(t)).view(1, T, 1)
    x = latents * phase
    var_t = x.var(dim=1, unbiased=False).mean()
    return weight * var_t
