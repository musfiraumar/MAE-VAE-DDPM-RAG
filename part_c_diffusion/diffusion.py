import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000

# 0-indexed: index 999 means t = T
betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, eps):
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

@torch.no_grad()
def sample(unet, labels, save_steps=None):
    unet.eval()
    labels = labels.to(DEVICE).long()
    n = len(labels)
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    saved = {}

    for t in range(T - 1, -1, -1):
        t_batch = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps_hat = unet(x, t_batch, class_labels=labels).sample
        mean = (1 / alphas[t].sqrt()) * (x - (1 - alphas[t]) / (1 - alpha_bar[t]).sqrt() * eps_hat)
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        if save_steps is not None and t in save_steps:
            saved[t] = x.clone()

    x = x.clamp(-1, 1)                         # clip only at the end
    if save_steps is not None:
        saved[0] = x
        return x, saved
    return x