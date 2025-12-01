import argparse
import os
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion import DDIMSampler, DiffusionSchedule, UNetModel

DDIM_ETA = 0.0  # deterministic DDIM sampling


def load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    return state, config


def build_model(config: dict, device: torch.device) -> UNetModel:
    channel_mults = tuple(config.get("channel_mults", (1, 2, 2, 4)))
    model = UNetModel(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 3),
        base_channels=config.get("base_channels", 128),
        channel_mults=channel_mults,
        image_size=config.get("image_size", 64),
    )
    return model.to(device)


def make_output_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def count_existing(prefix: str, suffix: str, directory: Path) -> int:
    matches = [name for name in os.listdir(directory) if name.startswith(prefix) and name.endswith(suffix)]
    return len(matches)


def sample_batches(model, schedule, args, device: torch.device):
    sampler = DDIMSampler(model, schedule, eta=DDIM_ETA)
    remaining = args.num_images
    out_dir = make_output_dir(args.output_dir)
    counter = count_existing("ddim_batch_", ".png", out_dir) + 1
    total_start = time.perf_counter()
    generated = 0

    while remaining > 0:
        batch = min(args.batch_size, remaining)
        batch_start = time.perf_counter()
        samples = sampler.sample((batch, 3, args.image_size, args.image_size), num_steps=args.num_steps)
        filename = out_dir / f"ddim_batch_{counter:03d}.png"
        nrow = min(8, batch)
        save_image(samples, filename.as_posix(), nrow=nrow, normalize=True, value_range=(-1, 1))
        batch_time = time.perf_counter() - batch_start
        generated += batch
        print(f"Saved {batch} samples to {filename.name} in {batch_time:.2f} s")
        remaining -= batch
        counter += 1
    total_time = time.perf_counter() - total_start
    print(f"Generated {generated} images across {counter-1} files in {total_time/60:.2f} min ({total_time:.1f} s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 samples with a trained DDIM model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="generated_samples_ddim")
    parser.add_argument("--timesteps", type=int, default=1000, help="fallback diffusion steps if missing from checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    overall_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, config = load_checkpoint(args.checkpoint, device)
    image_size = config.get("image_size", args.image_size)
    args.image_size = image_size
    schedule = DiffusionSchedule(T=config.get("timesteps", args.timesteps), device=device)
    model = build_model(config, device)
    model.load_state_dict(state["model"])
    model.eval()
    sample_batches(model, schedule, args, device)
    total_time = time.perf_counter() - overall_start
    print(f"ddim_generate.py finished in {total_time/60:.2f} min ({total_time:.1f} s). Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
