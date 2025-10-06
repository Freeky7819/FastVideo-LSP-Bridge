def log_periodic_stabilizer(latents, freq=ω, weight=λ):
    t = torch.arange(latents.size(1), device=latents.device)
    phase = torch.sin(freq * torch.log1p(t))
    loss = weight * ((latents * phase.unsqueeze(0).unsqueeze(-1)).var(dim=1).mean())
    return loss
