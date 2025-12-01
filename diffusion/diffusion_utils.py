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

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)             # (T,)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        alpha_bar_prev = torch.ones_like(alpha_bar)
        alpha_bar_prev[1:] = alpha_bar[:-1]
        self.alpha_bar_prev = alpha_bar_prev

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


class DDIMSampler:
    def __init__(self, model, schedule: DiffusionSchedule, eta: float = 0.0, clip_denoised: bool = True):
        self.model = model
        self.schedule = schedule
        self.eta = eta
        self.clip_denoised = clip_denoised

    def _build_timesteps(self, num_steps: int) -> torch.Tensor:
        total = self.schedule.T
        steps = max(2, min(num_steps, total))
        base = torch.linspace(0, total - 1, steps=steps, device=self.schedule.device)
        base = torch.round(base).to(dtype=torch.long)
        base = base.unique(sorted=True)
        if base[-1].item() != 0:
            base = torch.cat([base, torch.zeros(1, device=base.device, dtype=base.dtype)], dim=0)
        if base[0].item() != total - 1:
            head = torch.tensor([total - 1], device=base.device, dtype=base.dtype)
            base = torch.cat([head, base], dim=0)
        return torch.flip(base.unique(sorted=True), dims=[0])

    @torch.no_grad()
    def sample(self, shape, num_steps: int = 50, eta: float | None = None, clip_denoised: bool | None = None):
        device = self.schedule.device
        batch = shape[0]
        eta = self.eta if eta is None else eta
        clip = self.clip_denoised if clip_denoised is None else clip_denoised

        was_training = self.model.training
        self.model.eval()

        x = torch.randn(shape, device=device)
        timesteps = self._build_timesteps(num_steps)

        for idx, t_scalar in enumerate(timesteps):
            t_val = int(t_scalar.item())
            t = torch.full((batch,), t_val, device=device, dtype=torch.long)
            eps = self.model(x, t)

            sqrt_alpha_bar_t = self.schedule.sqrt_alpha_bar[t].view(batch, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bar[t].view(batch, 1, 1, 1)

            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
            if clip:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

            if idx == len(timesteps) - 1:
                sqrt_alpha_bar_prev = torch.ones_like(sqrt_alpha_bar_t)
                alpha_bar_prev = torch.ones_like(sqrt_alpha_bar_t)
                sqrt_one_minus_alpha_bar_prev = torch.zeros_like(sqrt_one_minus_alpha_bar_t)
            else:
                prev_val = int(timesteps[idx + 1].item())
                t_prev = torch.full((batch,), prev_val, device=device, dtype=torch.long)
                sqrt_alpha_bar_prev = self.schedule.sqrt_alpha_bar[t_prev].view(batch, 1, 1, 1)
                sqrt_one_minus_alpha_bar_prev = self.schedule.sqrt_one_minus_alpha_bar[t_prev].view(batch, 1, 1, 1)
                alpha_bar_prev = sqrt_alpha_bar_prev ** 2

            alpha_bar_t = sqrt_alpha_bar_t ** 2
            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev),
                    min=0.0,
                )
            )

            dir_xt = sqrt_alpha_bar_prev * pred_x0
            noise_scale = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0))
            x = dir_xt + noise_scale * eps
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        if was_training:
            self.model.train()
        return x