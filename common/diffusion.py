# diffusion.py  (compile-safe version)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style √10000 sinusoidal embedding."""
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
    )
    angles = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:                                            # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb

# ------------------------------------------------------------
# Core latent-space diffusion model
# ------------------------------------------------------------
class LatentDiffusionModel(nn.Module):
    """Residual MLP ε-predictor for vector latents."""
    def __init__(self, latent_dim: int, hidden: int = 512,
                 depth: int = 4, time_dim: int = 128):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU()
        )
        self.in_proj  = nn.Linear(latent_dim, hidden)
        self.blocks   = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))
             for _ in range(depth)]
        )
        self.out_proj = nn.Linear(hidden, latent_dim)
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.time_proj(sinusoidal_embedding(t, self.time_dim))
        for blk in self.blocks:
            h = h + blk(h)                 # simple residual
        return self.out_proj(h)            # ε̂


# ------------------------------------------------------------
# DDPM wrapper
# ------------------------------------------------------------
class DDPM(nn.Module):
    """
    Standard DDPM machinery around the ε-predictor.
    Uses a linear β-schedule by default (1 000 steps).
    """
    def __init__(self, model: LatentDiffusionModel,
                 n_steps: int = 1_000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2):
        super().__init__()
        self.model = model
        betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.n_steps = n_steps

    # ---- forward (q) process -----------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """x_t = √α̅_t · x₀ + √(1-α̅_t) · ε"""
        sqrt_alpha_cum           = torch.sqrt(self.alphas_cum[t])          # (B,)
        sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alphas_cum[t])    # (B,)

        # broadcast to x0’s shape
        while sqrt_alpha_cum.dim() < x0.dim():
            sqrt_alpha_cum           = sqrt_alpha_cum.unsqueeze(-1)
            sqrt_one_minus_alpha_cum = sqrt_one_minus_alpha_cum.unsqueeze(-1)

        return sqrt_alpha_cum * x0 + sqrt_one_minus_alpha_cum * noise

    # ---- training loss ------------------------------------------------------
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t   = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    # ---- reverse (p) process ------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        One reverse step: x_{t-1} ← x_t.
        """
        batch = x_t.size(0)
        t_batch = torch.full((batch,), t, device=x_t.device, dtype=torch.long)
        eps_pred = self.model(x_t, t_batch)

        beta_t            = self.betas[t]
        alpha_t           = 1.0 - beta_t
        alpha_bar_t       = self.alphas_cum[t]
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

        coeff = beta_t / sqrt_one_minus_ab
        mean  = (1.0 / torch.sqrt(alpha_t)) * (x_t - coeff * eps_pred)

        if t == 0:
            return mean                                    # final clean sample
        noise  = torch.randn_like(x_t)
        sigma  = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, shape, device) -> torch.Tensor:
        """Full chain sampling (mostly for debugging)."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.p_sample(x, t)
        return x

    # --- in DDPM ---------------------------------------------------------------
    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single-shot estimate of the clean latent x0 given a noisy x_t.
        t can be an int (same for the whole batch) or a (B,) LongTensor.
        """
        if isinstance(t, int):
            t = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)

        eps_pred = self.model(x_t, t)  # ε̂_θ(x_t, t)

        alpha_bar = self.alphas_cum[t]  # (B,)
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_1mab = torch.sqrt(1.0 - alpha_bar)

        while sqrt_ab.dim() < x_t.dim():  # broadcast to x_t shape
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1mab = sqrt_1mab.unsqueeze(-1)

        x0_hat = (x_t - sqrt_1mab * eps_pred) / sqrt_ab
        return x0_hat


# ------------------------------------------------------------
# Simple latent replay buffer
# ------------------------------------------------------------
class LatentReplay(Dataset):
    def __init__(self, latents: torch.Tensor):
        self.latents = latents
    def __len__(self):             return self.latents.size(0)
    def __getitem__(self, idx):    return self.latents[idx]

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train_diffusion(model: DDPM,
                    loader: DataLoader,
                    epochs: int = 10,
                    lr: float = 2e-4,
                    grad_clip: float = 1.0,
                    device: str = "cuda"):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x0 in loader:
            x0 = x0.to(device)
            t  = torch.randint(0, model.n_steps, (x0.size(0),),
                               device=device, dtype=torch.long)

            loss = model.p_losses(x0, t)

            optim.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            total_loss += loss.item() * x0.size(0)

        mean_loss = total_loss / len(loader.dataset)
        print(f"[{epoch}/{epochs}]  loss={mean_loss:.4f}")
