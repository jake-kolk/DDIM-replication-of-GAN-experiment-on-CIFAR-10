import subprocess

for i in range(1, 6):  # 1, 2, 3, 4, 5
    cmd = [
        "python3.10", "compare.py",
        "--ddim-checkpoint", f"weights/ddim_{i}/ddim_epoch_050.pth",
        "--gan-checkpoint", f"weights/gan_{i}/weights/netG_epoch_24.pth",
        "--num-samples", "10000",
        "--batch-size", "64",
        "--output-dir", f"comparison_results_{i}",
        "--real-data", "cifar10"
    ]

    print(f"\n=== Running comparison for i={i} ===")
    subprocess.run(cmd, check=True)
