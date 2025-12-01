import math
import torch
import torch.nn as nn


def _make_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        const = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -const)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.embed = SinusoidalPosEmb(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(t))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            _make_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            _make_norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time = self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + time
        h = self.block2(h)
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _make_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(h_).reshape(b, c, h * w)
        v = self.v(h_).reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        image_size: int = 64,
        time_dim: int | None = None,
        attn_resolutions: tuple[int, ...] = (16,),
    ):
        super().__init__()
        if time_dim is None:
            time_dim = base_channels * 4

        self.time_mlp = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.mid = None
        self.out = None

        current_res = image_size
        in_ch = base_channels
        skip_channels: list[int] = []

        for idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block1 = ResidualBlock(in_ch, out_ch, time_dim)
            block2 = ResidualBlock(out_ch, out_ch, time_dim)
            use_attn = current_res in attn_resolutions
            attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
            down = Downsample(out_ch) if idx < len(channel_mults) - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([block1, block2, attn, down]))
            skip_channels.extend([out_ch, out_ch])
            in_ch = out_ch
            if idx < len(channel_mults) - 1:
                current_res //= 2

        self.mid = nn.ModuleList(
            [
                ResidualBlock(in_ch, in_ch, time_dim),
                AttentionBlock(in_ch),
                ResidualBlock(in_ch, in_ch, time_dim),
            ]
        )

        current_res = current_res
        for idx, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            skip_ch2 = skip_channels.pop()
            skip_ch1 = skip_channels.pop()
            block1 = ResidualBlock(in_ch + skip_ch2, out_ch, time_dim)
            block2 = ResidualBlock(out_ch + skip_ch1, out_ch, time_dim)
            use_attn = current_res in attn_resolutions
            attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
            up = Upsample(out_ch, base_channels * channel_mults[idx - 1]) if idx > 0 else nn.Identity()
            self.ups.append(nn.ModuleList([block1, block2, attn, up]))
            in_ch = out_ch
            if idx > 0:
                current_res *= 2

        self.out = nn.Sequential(
            _make_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.in_conv(x)
        skips: list[torch.Tensor] = []

        for block1, block2, attn, down in self.downs:
            x = block1(x, t_emb)
            skips.append(x)
            x = block2(x, t_emb)
            skips.append(x)
            x = attn(x)
            x = down(x)

        mid1, mid_attn, mid2 = self.mid
        x = mid1(x, t_emb)
        x = mid_attn(x)
        x = mid2(x, t_emb)

        for block1, block2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, t_emb)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, t_emb)
            x = attn(x)
            x = up(x)

        return self.out(x)
