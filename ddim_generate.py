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


def sample_single_images(model, schedule, args, device: torch.device):
    sampler = DDIMSampler(model, schedule, eta=DDIM_ETA)
    out_dir = make_output_dir(args.output_dir)
    counter = count_existing("ddim_", ".png", out_dir) + 1
    
    total_start = time.perf_counter()

    for i in range(args.num_images):
        start = time.perf_counter()
        
        # Generate ONE image at a time
        sample = sampler.sample((1, 3, args.image_size, args.image_size), num_steps=args.num_steps)
        sample = sample.squeeze(0)  # remove batch dim

        filename = out_dir / f"ddim_{counter:04d}.png"
        save_image(sample, filename.as_posix(), normalize=True, value_range=(-1, 1))

        dt = time.perf_counter() - start
        print(f"Saved image {filename.name} in {dt:.2f}s")

        counter += 1

    total_time = time.perf_counter() - total_start
    print(f"Generated {args.num_images} images in {total_time:.1f}s ({total_time/60:.2f} min)")


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
    sample_single_images(model, schedule, args, device)
    total_time = time.perf_counter() - overall_start
    print(f"ddim_generate.py finished in {total_time/60:.2f} min ({total_time:.1f} s). Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
