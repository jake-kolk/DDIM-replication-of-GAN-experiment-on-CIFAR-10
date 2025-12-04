from __future__ import annotations
import argparse
import os
import random
import time
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

cudnn.benchmark = True

LOGFILE = "ganlog.txt"

def log(*args, **kwargs):
    """Prints to console AND appends to ganlog.txt."""
    message = " ".join(str(a) for a in args)
    print(message, **kwargs)
    with open(LOGFILE, "a") as f:
        f.write(message + "\n")

LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)

def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN on CIFAR-10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--nz", type=int, default=100, help="latent vector size")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resumeG", type=str, default="", help="path to generator checkpoint")
    parser.add_argument("--resumeD", type=str, default="", help="path to discriminator checkpoint"),
    parser.add_argument("--outputDir", type=str, default="gan", help="path to store weights and output")
    return parser.parse_args()

def set_seed(seed: int):
    manual_seed = seed if seed != 0 else random.randint(1, 10000)
    log("Random Seed:", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu: int, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            return nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu: int, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def build_dataloader(args):
    dataset = dset.CIFAR10(
        root=args.data_root,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    pin = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

def save_outputs(real, fake, epoch, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vutils.save_image(real, os.path.join(OUTPUT_DIR, 'real_samples.png'), normalize=True)
    vutils.save_image(
        fake.detach(),
        os.path.join(OUTPUT_DIR, f'fake_samples_epoch_{epoch:03d}.png'),
        normalize=True,
    )

def save_checkpoints(netG, netD, epoch, CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f'netG_epoch_{epoch}.pth'))
    torch.save(netD.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_epoch_{epoch}.pth'))

def main():
    args = parse_args()
    set_seed(args.seed)

    CHECKPOINT_DIR = os.path.join(args.outputDir, "weights")
    OUTPUT_DIR = os.path.join(args.outputDir, "output")

    dataloader = build_dataloader(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1

    netG = Generator(ngpu, args.nz, args.ngf).to(device)
    netG.apply(weights_init)
    if args.resumeG:
        netG.load_state_dict(torch.load(args.resumeG, map_location=device))
    log(netG)

    netD = Discriminator(ngpu, args.ndf).to(device)
    netD.apply(weights_init)
    if args.resumeD:
        netD.load_state_dict(torch.load(args.resumeD, map_location=device))
    log(netD)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(args.sample_batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    run_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(float(fake_label))
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(float(real_label))
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            log('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, i, len(dataloader),
                   errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                log('Saving output images')
                fake_samples = netG(fixed_noise)
                save_outputs(real_cpu, fake_samples, epoch, OUTPUT_DIR)

        save_checkpoints(netG, netD, epoch, CHECKPOINT_DIR)

        epoch_time = time.perf_counter() - epoch_start
        log(f"Epoch {epoch+1}/{args.epochs} finished in {epoch_time/60:.2f} min ({epoch_time:.1f} s)")

        total_time = time.perf_counter() - run_start
        log(f"Training time so far: {total_time/3600:.2f} h ({total_time/60:.2f} min)")

if __name__ == "__main__":
    main()
