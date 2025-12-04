#!/usr/bin/env python3
"""
compare_after_training.py

Generates samples from a trained DDIM and a trained GAN, times sampling,
computes FID (Frechet Inception Distance) against a real dataset, and writes a report.

Usage example:
python compare.py --ddim-checkpoint weights/ddim_1/ddim_epoch_050.pth --gan-checkpoint weights/gan_1/netG_epoch_24.pth --num-samples 10 --batch-size 64 --output-dir comparison_results --real-data data/cifar-10-batches-py
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image

# --- Replace these imports with your project modules (same names you used before) ---
# from diffusion import DDIMSampler, DiffusionSchedule, UNetModel
# from gan_cifar import Generator
#
# If these modules are in another path, either adjust PYTHONPATH or edit imports below.

# --- Fallback placeholders to avoid import errors while showing code.
# Remove these placeholders and uncomment the real imports above when running in your env.
try:
    from diffusion import DDIMSampler, DiffusionSchedule, UNetModel  # type: ignore
except Exception:
    DDIMSampler = None  # type: ignore
    DiffusionSchedule = None  # type: ignore
    UNetModel = None  # type: ignore

try:
    from gan_cifar import Generator  # type: ignore
except Exception:
    Generator = None  # type: ignore
# -----------------------------------------------------------------------------------

from torchvision.models import inception_v3  # type: ignore


# -------------------------
# Utilities for FID
# -------------------------
def get_inception_model(device: torch.device):
    """
    Returns an InceptionV3 model adapted to produce 2048-d pool features.
    We set model.fc = Identity so model(x) returns the pooled features.
    """
    m = inception_v3(pretrained=True, aux_logits=False)
    # Replace final fc so forward returns the pooled features (2048-d)
    m.fc = nn.Identity()
    m.eval()
    m.to(device)
    return m


def preprocess_for_inception(x: torch.Tensor, device: torch.device):
    """
    x: tensor in range [-1, 1] or [0,1], shape (B, C, H, W)
    returns: tensor shaped (B, 3, 299, 299) normalized with ImageNet mean/std
    """
    # ensure float
    if x.dtype != torch.float32:
        x = x.float()
    # If input is [-1,1], convert to [0,1]
    if x.min() < -0.1:
        x = (x + 1.0) / 2.0
    # Resize to 299x299 expected by inception
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(device)


def get_activations(dataloader, model, device: torch.device, dims=2048):
    """
    Run images from dataloader through inception model to get activations.
    Returns numpy array shape (N, dims).
    """
    model.eval()
    act_list = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)
            imgs = preprocess_for_inception(imgs, device)
            feats = model(imgs)  # (B, dims)
            if feats.dim() == 4:
                feats = feats.squeeze(-1).squeeze(-1)
            act_list.append(feats.cpu().numpy())
    if len(act_list) == 0:
        return np.zeros((0, dims))
    return np.concatenate(act_list, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of FID via Frechet distance between two Gaussians.
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("WARN: adding eps to diagonal of cov estimates for stability")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def compute_fid_from_activations(act_real: np.ndarray, act_gen: np.ndarray):
    mu_real = np.mean(act_real, axis=0)
    mu_gen = np.mean(act_gen, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)
    sigma_gen = np.cov(act_gen, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


# -------------------------
# Sample generation helpers
# -------------------------
def make_output_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def count_existing(prefix: str, suffix: str, directory: Path) -> int:
    matches = [name for name in os.listdir(directory) if name.startswith(prefix) and name.endswith(suffix)]
    return len(matches)


def load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    return state, config


def sample_ddim(model, schedule, sampler_class, checkpoint_state, args, device: torch.device) -> Tuple[np.ndarray, float]:
    """
    Generate samples from DDIM in batches. Returns numpy array of shape (N, C, H, W) in [-1,1],
    and the total sampling time (seconds).
    """
    if sampler_class is None or DDIMSampler is None:
        raise RuntimeError("DDIMSampler not available - check imports")
    sampler = sampler_class(model, schedule, eta=args.ddim_eta)
    out_dir = make_output_dir(os.path.join(args.output_dir, "ddim_samples"))
    start_index = count_existing("ddim_", ".png", out_dir) + 1

    all_samples = []
    total_start = time.perf_counter()
    n_batches = math.ceil(args.num_samples / args.batch_size)
    for bi in range(n_batches):
        b = min(args.batch_size, args.num_samples - bi * args.batch_size)
        t0 = time.perf_counter()
        # assume sampler.sample accepts (batch, C, H, W)
        batch = sampler.sample((b, 3, args.image_size, args.image_size), num_steps=args.num_steps)
        t1 = time.perf_counter()
        # ensure CPU numpy
        batch = batch.detach().cpu()
        # Save each image individually
        for i in range(b):
            idx = start_index + bi * args.batch_size + i
            fname = out_dir / f"ddim_{idx:04d}.png"
            save_image(batch[i], fname.as_posix(), normalize=True, value_range=(-1, 1))
        all_samples.append(batch.numpy())
        print(f"DDIM: batch {bi+1}/{n_batches} generated in {t1-t0:.3f}s")
    total_time = time.perf_counter() - total_start
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples[: args.num_samples], total_time


def sample_gan(generator_class, gan_checkpoint_path: str, args, device: torch.device) -> Tuple[np.ndarray, float]:
    """
    Generate samples from a GAN generator. Returns numpy array shape (N, C, H, W) in [-1,1] and total time.
    """
    if generator_class is None:
        raise RuntimeError("Generator class not available - check imports")
    # instantiate generator. Assumes signature Generator(ngpu=...) or no args
    try:
        netG = generator_class(ngpu=1).to(device)
    except TypeError:
        netG = generator_class().to(device)
    state = torch.load(gan_checkpoint_path, map_location=device)
    # try dict or state dict
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        netG.load_state_dict(state["state_dict"])
    elif "model" in state:
        netG.load_state_dict(state["model"])
    else:
        # fallback: assume the checkpoint is a state_dict
        netG.load_state_dict(state)
    netG.eval()

    out_dir = make_output_dir(os.path.join(args.output_dir, "gan_samples"))
    existing = [f for f in os.listdir(out_dir) if f.startswith("generated_image(") and f.endswith(").png")]
    next_index = len(existing) + 1

    all_samples = []
    total_start = time.perf_counter()
    n_batches = math.ceil(args.num_samples / args.batch_size)
    noise_dim = getattr(args, "noise_dim", 100)
    for bi in range(n_batches):
        b = min(args.batch_size, args.num_samples - bi * args.batch_size)
        noise = torch.randn(b, noise_dim, 1, 1, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            fake = netG(noise).detach().cpu()  # expect [-1,1]
        t1 = time.perf_counter()
        for i in range(b):
            fname = os.path.join(out_dir, f"generated_image({next_index}).png")
            save_image(fake[i], fname, normalize=True, value_range=(-1, 1))
            next_index += 1
        all_samples.append(fake.numpy())
        print(f"GAN: batch {bi+1}/{n_batches} generated in {t1-t0:.3f}s")
    total_time = time.perf_counter() - total_start
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples[: args.num_samples], total_time


# -------------------------
# Real dataset loader
# -------------------------
def get_real_dataloader(args, image_size: int, batch_size: int):
    """
    By default, uses CIFAR-10 test set (32x32). You can also specify a folder path
    (args.real_data pointing to a directory of images).
    """
    if args.real_data == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # yields [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1,1]
        ])
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return loader
    else:
        # assume real_data is a directory of images
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = datasets.ImageFolder(root=args.real_data, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return loader


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare DDIM vs GAN after training")
    parser.add_argument("--ddim-checkpoint", type=str, required=True)
    parser.add_argument("--gan-checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=500, help="Total samples to generate per model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    parser.add_argument("--image-size", type=int, default=64, help="Model image size (H=W)")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--real-data", type=str, default="cifar10", help="Either 'cifar10' or path to folder with real images")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--inception-batch", type=int, default=16, help="batch size when computing inception features")
    parser.add_argument("--noise-dim", type=int, default=100, help="latent dim for GAN generation")
    parser.add_argument("--save-report-json", type=str, default="comparison_report.json")
    args = parser.parse_args()

    # attach some args to be used by functions
    args.num_steps = args.ddim_steps
    args.ddim_eta = args.ddim_eta
    args.output_dir = make_output_dir(args.output_dir).as_posix()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Load models / checkpoints
    print("Loading checkpoints...")
    ddim_state, ddim_config = load_checkpoint(args.ddim_checkpoint, device)
    gan_state = torch.load(args.gan_checkpoint, map_location=device)

    # Build DDIM model
    if UNetModel is None or DiffusionSchedule is None or DDIMSampler is None:
        print("WARNING: DDIM classes not available at import time. Ensure 'diffusion' module is on PYTHONPATH.")
    else:
        channel_mults = tuple(ddim_config.get("channel_mults", (1, 2, 2, 4)))
        ddim_model = UNetModel(
            in_channels=ddim_config.get("in_channels", 3),
            out_channels=ddim_config.get("out_channels", 3),
            base_channels=ddim_config.get("base_channels", 128),
            channel_mults=channel_mults,
            image_size=ddim_config.get("image_size", args.image_size),
        ).to(device)
        # load weights
        try:
            ddim_model.load_state_dict(ddim_state["model"])
        except Exception:
            # try direct state dict
            ddim_model.load_state_dict(ddim_state)

        schedule = DiffusionSchedule(T=ddim_config.get("timesteps", 1000), device=device)

    # Generate samples and time
    results = {}
    # Try to extract training-time metadata if present
    results["ddim_checkpoint"] = args.ddim_checkpoint
    results["gan_checkpoint"] = args.gan_checkpoint
    results["metadata"] = {}
    if isinstance(ddim_state, dict):
        results["metadata"]["ddim_keys"] = list(ddim_state.keys())
    if isinstance(gan_state, dict):
        results["metadata"]["gan_keys"] = list(gan_state.keys())

    # DDIM sampling
    print("Generating DDIM samples...")
    try:
        ddim_samples, ddim_time = sample_ddim(ddim_model, schedule, DDIMSampler, ddim_state, args, device)
        results["ddim_sampling_time_s"] = float(ddim_time)
        results["ddim_samples_saved_dir"] = os.path.join(args.output_dir, "ddim_samples")
    except Exception as e:
        ddim_samples = None
        print("Failed to sample DDIM:", e)
        results["ddim_sampling_error"] = str(e)

    # GAN sampling
    print("Generating GAN samples...")
    try:
        gan_samples, gan_time = sample_gan(Generator, args.gan_checkpoint, args, device)
        results["gan_sampling_time_s"] = float(gan_time)
        results["gan_samples_saved_dir"] = os.path.join(args.output_dir, "gan_samples")
    except Exception as e:
        gan_samples = None
        print("Failed to sample GAN:", e)
        results["gan_sampling_error"] = str(e)

    # Compute per-image times
    if ddim_samples is not None:
        results["ddim_per_image_s"] = results["ddim_sampling_time_s"] / float(args.num_samples)
    if gan_samples is not None:
        results["gan_per_image_s"] = results["gan_sampling_time_s"] / float(args.num_samples)

    # Prepare real dataloader
    print("Preparing real dataset loader...")
    real_loader = get_real_dataloader(args, image_size=args.image_size, batch_size=args.inception_batch)

    # Prepare inception model
    print("Loading Inception model for FID computation (this may take a while the first time)...")
    inc = get_inception_model(device)

    # Get activations for real images (use same count as num_samples)
    print("Getting activations for real images...")
    # Build a dataloader that yields exactly args.num_samples images (stop when enough)
    def limited_loader(loader, limit):
        taken = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            b = imgs.shape[0]
            if taken + b > limit:
                remain = limit - taken
                yield imgs[:remain]
                return
            yield imgs
            taken += b
            if taken >= limit:
                return

    real_subloader = limited_loader(real_loader, args.num_samples)
    act_real = get_activations(real_subloader, inc, device)

    # Get activations for generated images (DDIM and GAN)
    if ddim_samples is not None:
        print("Computing activations for DDIM samples...")
        # convert numpy -> torch tensor and create loader
        t = torch.from_numpy(ddim_samples).float()
        dd_loader = torch.utils.data.DataLoader(t, batch_size=args.inception_batch)
        act_ddim = get_activations(dd_loader, inc, device)
        fid_ddim = compute_fid_from_activations(act_real, act_ddim)
        results["fid_ddim"] = float(fid_ddim)
        print(f"FID (DDIM vs real): {fid_ddim:.3f}")
    else:
        results["fid_ddim"] = None

    if gan_samples is not None:
        print("Computing activations for GAN samples...")
        t = torch.from_numpy(gan_samples).float()
        gan_loader = torch.utils.data.DataLoader(t, batch_size=args.inception_batch)
        act_gan = get_activations(gan_loader, inc, device)
        fid_gan = compute_fid_from_activations(act_real, act_gan)
        results["fid_gan"] = float(fid_gan)
        print(f"FID (GAN vs real): {fid_gan:.3f}")
    else:
        results["fid_gan"] = None

    # Save a small summary CSV and JSON
    out_dir = Path(args.output_dir)
    json_path = out_dir / args.save_report_json
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    # Also save a minimal CSV
    import csv

    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "total_sampling_time_s", "per_image_s", "FID_vs_real"])
        if results.get("ddim_sampling_time_s") is not None:
            writer.writerow(["DDIM", results["ddim_sampling_time_s"], results["ddim_per_image_s"], results.get("fid_ddim")])
        if results.get("gan_sampling_time_s") is not None:
            writer.writerow(["GAN", results["gan_sampling_time_s"], results["gan_per_image_s"], results.get("fid_gan")])

    print("Comparison complete.")
    print(f"JSON report: {json_path}")
    print(f"CSV summary: {csv_path}")
    print("Full results keys:", list(results.keys()))


if __name__ == "__main__":
    main()
