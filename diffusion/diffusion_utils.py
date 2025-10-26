import torch

def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2, schedule="linear"):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    raise ValueError(f"Unsupported schedule: {schedule}")

class DiffusionSchedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T).to(device)                # (T,)
        alphas = 1.0 - betas                                    # (T,)
        alpha_bar = torch.cumprod(alphas, dim=0)                # ᾱ_t
        # handy square roots
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)             # (T,)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    def add_noise(self, x0, t, noise=None):
        """
        x0: (N,C,H,W) in [-1,1]
        t:  (N,) int64 timesteps in [0, T-1]
        returns: x_t, eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # gather per-sample scalars and reshape to (N,1,1,1)
        s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = s1 * x0 + s2 * noise
        return x_t, noise