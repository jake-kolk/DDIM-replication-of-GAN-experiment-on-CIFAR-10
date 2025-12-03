import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

print(torch.cuda.is_available())  # should print True these 3 lines are to test gpu recognision
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
from diffusion import DDIMSampler, DiffusionSchedule, UNetModel

# ------------------ new imports for mixed precision ------------------
from torch.cuda.amp import autocast, GradScaler

def _parse_channel_mults(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(","))


def build_dataloader(image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    tx = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tx)
    pin = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)


def sample_images(model, schedule, args, epoch: int):
    sampler = DDIMSampler(model, schedule, eta=0.0)
    start = time.perf_counter()
    samples = sampler.sample(
        (args.sample_batch_size, 3, args.image_size, args.image_size),
        num_steps=args.num_sample_steps,
    )
    out_path = Path(args.sample_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    grid_name = out_path / f"ddim_samples_epoch_{epoch:03d}.png"
    nrow = int(math.sqrt(args.sample_batch_size))
    nrow = max(1, nrow)
    save_image(samples, grid_name.as_posix(), nrow=nrow, normalize=True, value_range=(-1, 1))
    return time.perf_counter() - start


def save_checkpoint(model, optimizer, epoch: int, args, global_step: int):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": args.base_channels,
        "channel_mults": args.channel_mults,
        "image_size": args.image_size,
        "timesteps": args.timesteps,
    }
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    torch.save(payload, ckpt_dir / f"ddim_epoch_{epoch:03d}.pth")


def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = build_dataloader(args.image_size, args.batch_size, args.num_workers)

    model = UNetModel(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        image_size=args.image_size,
    ).to(device)
    schedule = DiffusionSchedule(T=args.timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ------------------ new: initialize GradScaler ------------------
    scaler = GradScaler()

    start_epoch = 0
    global_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)

    run_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")

        accum_steps = max(1, args.accum_steps)
        optimizer.zero_grad()

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = torch.randint(0, schedule.T, (images.size(0),), device=device, dtype=torch.long)
            noise = torch.randn_like(images)
            noisy, noise = schedule.add_noise(images, t, noise)

            # ------------------ mixed precision ------------------
            with autocast():
                preds = model(noisy, t)
                loss = F.mse_loss(preds, noise)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if (i + 1) % accum_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            # show the actual (unscaled) loss in the progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % args.sample_every == 0:
            sample_dur = sample_images(model, schedule, args, epoch + 1)
            print(f"Saved DDIM preview grid in {sample_dur:.2f} s")

        save_checkpoint(model, optimizer, epoch + 1, args, global_step)
        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} finished in {epoch_time/60:.2f} min ({epoch_time:.1f} s)")

    total_time = time.perf_counter() - run_start
    print(f"Training completed in {total_time/3600:.2f} h ({total_time/60:.2f} min)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDIM on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mults", type=str, default="1,2,2,4")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=1, help="epochs between saved sample grids")
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--num-sample-steps", type=int, default=50)
    parser.add_argument("--sample-dir", type=str, default="generated_samples_ddim")
    parser.add_argument("--ckpt-dir", type=str, default="weights/ddim")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accum-steps", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cli_args.channel_mults = _parse_channel_mults(cli_args.channel_mults)
    train(cli_args)
