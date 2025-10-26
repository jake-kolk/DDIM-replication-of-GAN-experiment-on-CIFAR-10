import os, torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from .diffusion_utils import DiffusionSchedule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "output/noisy"
os.makedirs(OUT_DIR, exist_ok=True)

# CIFAR-10 in [-1,1]
tx = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

def main():
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tx)
    # use 0 workers for portability; fine for a one-off preview script
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    sched = DiffusionSchedule(T=1000, device=DEVICE)

    # take one fixed batch and visualize progressive noise
    x0, _ = next(iter(dl))
    x0 = x0.to(DEVICE)

    viz_steps = [1, 50, 100, 250, 500, 750, 999]
    with torch.no_grad():
        for t_scalar in viz_steps:
            t = torch.full((x0.size(0),), t_scalar, dtype=torch.long, device=DEVICE)
            xt, _ = sched.add_noise(x0, t)
            save_image(
                xt, os.path.join(OUT_DIR, f"cifar_noisy_t{t_scalar:03d}.png"),
                nrow=8, normalize=True, value_range=(-1, 1)
            )
    print(f"Saved noisy grids to {OUT_DIR}")

if __name__ == "__main__":
    main()