import torch
import torchvision.utils as vutils
from gan_cifar import Generator  # make sure this matches your filename
import os

# Pick device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Initialize generator
    netG = Generator(ngpu=1).to(device)

    # Load trained weights
    netG.load_state_dict(torch.load("weights/netG_epoch_24.pth", map_location=device))
    netG.eval()
    print("Loaded trained generator weights")

    # Generate random noise (100-D latent vectors)
    noise = torch.randn(64, 100, 1, 1, device=device)

    # Generate fake images
    with torch.no_grad():
        fake_images = netG(noise).detach().cpu()

    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)

    # Count existing generated images to increment name
    existing = [f for f in os.listdir(output_dir) if f.startswith("generated_image(") and f.endswith(").png")]
    next_index = len(existing) + 1

    filename = os.path.join(output_dir, f"generated_image({next_index}).png")
    vutils.save_image(fake_images, filename, normalize=True, value_range=(-1, 1))

    print(f"Saved new image as: {filename}")


if __name__ == "__main__":
    main()
