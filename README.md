# DCGAN-CIFAR10-pytorch
A DCGAN built on the CIFAR10 dataset using pytorch

DCGAN is one of the popular and successful network designs for GAN. It mainly composes
of convolution layers without max pooling or fully connected layers. It uses convolutional
stride and transposed convolution for the downsampling and the upsampling. Architecture
guidelines for stable Deep Convolutional GANs as mentioned by Soumith Chintala

These are the guidelines for constructing a DCGAN as mentioned by Soumith Chintala (https://arxiv.org/abs/1511.06434)

Replace any pooling layers with strided convolutions(discriminator) and fractional-
strided convolutions (generator).

Use batchnorm in both the generator and the discriminator.

Remove fully connected hidden layers for deeper architectures.

Use ReLU activation in generator for all layers except for the output, which uses Tanh.

Use LeakyReLU activation in the discriminator for all layers.

The simplicity of DCGAN contributes to its success. We reach certain bottleneck that
increasing the complexity of the generator does not necessarily improve the image quality.
Until we identify the bottleneck and know how to train GANs more effective, DCGAN
remains a good start point for a new project.
I created a DCGAN model for mimicking the data distribution of CIFAR-10 dataset


Alright this is the stuff written by Jake + Isaiah.

## Download Dependencies

```bash
python -m pip install -r requirements.txt
```

## DCGAN experiment

### Train the GAN

```bash
python gan_cifar.py \
	--epochs 25 \
	--batch-size 128
```

Flags you can tweak:

- `--epochs`: number of passes over CIFAR-10 (default 25).
- `--batch-size`: training batch size (default 128, must fit in GPU memory).
- `--image-size`: resize for CIFAR-10 crops (default 64×64 to match diffusion runs).
- `--nz`: latent vector length fed into the generator (default 100).
- `--ngf` / `--ndf`: base channel widths for generator and discriminator (both default 64).
- `--beta1` / `--beta2`: Adam momentum terms; generally leave at (0.5, 0.999) unless experimenting.
- `--resumeG` / `--resumeD`: paths to checkpoints if you want to keep training.
- `--seed`: set for reproducible runs (0 picks a random seed).

The script now always saves checkpoints to `weights/`, sample grids to `output/`, and trains with the original DCGAN learning rate of 2e-4 so you don’t have to remember extra flags.

### Generate GAN images without retraining

```bash
python generate.py
```

This loads `weights/netG_epoch_24.pth` by default and writes `generated_samples/generated_image(N).png`. Point it to other checkpoints by editing the script or symlinking the weights you want.



## DDIM replication experiment

We swapped the DCGAN generator for a DDIM trained directly on CIFAR-10. The UNet backbone and sampler live in the `diffusion` package.

### Train the DDIM

```bash
python train_ddim.py \
	--epochs 50 \
	--batch-size 128 \
	--ckpt-dir weights/ddim \
	--sample-dir generated_samples_ddim
```

Flag guide:

- `--lr`: optimizer learning rate (default 2e-4; diffusion can be sensitive so keep close to this).
- `--timesteps`: length of the forward diffusion process (1000 matches DDPM/DDIM papers).
- `--channel-mults`: comma list that controls UNet width per resolution stage (e.g., `1,2,2,4`).
- `--base-channels`: starting channel count before multipliers (default 128).
- `--grad-clip`: gradient norm cap to keep training stable (1.0 by default).
- `--sample-every`: epochs between saving DDIM sample grids.
- `--sample-batch-size`: number of images in each saved grid.
- `--num-sample-steps`: reverse iterations for those preview grids (fewer = faster, more = cleaner).
- `--ckpt-dir` / `--sample-dir`: where checkpoints and preview grids land.
- `--resume`: continue from a saved checkpoint produced by this script.

During training we always use deterministic DDIM sampling (`η = 0`) when writing preview grids so the results are easy to compare over time.

### Generate images with a trained DDIM

```bash
python ddim_generate.py \
	--checkpoint weights/ddim/ddim_epoch_050.pth \
	--num-images 64 \
	--num-steps 50 \
	--output-dir generated_samples_ddim
```

Flag guide:

- `--checkpoint`: which training checkpoint to load.
- `--num-images`: total images to synthesize.
- `--batch-size`: sampler batch size (use a smaller value if VRAM is limited).
- `--num-steps`: number of reverse steps; 50 is a good quality/speed trade-off, and higher values produce sharper images.
- `--output-dir`: folder for the PNG grids (`ddim_batch_###.png`).

The generator script also fixes `η = 0`, so all samples are deterministic given the same random seed; edit `DDIM_ETA` inside `ddim_generate.py` if you ever want to explore stochastic sampling.