import subprocess
import time

# Number of images in CIFAR-10 training set
TRAIN_DATASET_SIZE = 50000

BATCH_SIZE = 32
EPOCHS = 50

OUTPUT_FILE = "training_results.txt"


def ordinal(n):
    return ["once", "twice", "three times", "four times", "five times"][n-1]


def run_training():
    results = []

    # open the file fresh
    with open(OUTPUT_FILE, "w") as f:
        f.write("DDIM Training Performance Results\n")
        f.write("=================================\n\n")

    for i in range(1, 6):
        command = [
            "python3.10", "train_ddim.py",
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--ckpt-dir", f"weights/ddim_{i}",
            "--sample-dir", "generated_samples_ddim",
            "--accum-steps", "4"
        ]
        print(f"\n=== Starting training run {i} ===")

        # Time tracking (wall time only)
        start_wall = time.time()

        # Run training
        subprocess.run(command, check=True)

        end_wall = time.time()
        wall_time = end_wall - start_wall

        total_samples = TRAIN_DATASET_SIZE * EPOCHS
        throughput_sps = total_samples / wall_time

        results.append({
            "run": i,
            "wall_time_sec": wall_time,
            "throughput_sps": throughput_sps
        })

        print(f"Training has run {ordinal(i)}.")

        # Append run results to the text file
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"Run {i}:\n")
            f.write(f"  Wall time:  {wall_time:.2f} sec\n")
            f.write(f"  Throughput: {throughput_sps:.2f} samples/sec\n\n")

    # Summary printed to console
    print("\n=========== SUMMARY ===========")
    for r in results:
        print(f"Run {r['run']}:")
        print(f"  Wall time:  {r['wall_time_sec']:.2f} sec")
        print(f"  Throughput: {r['throughput_sps']:.2f} samples/sec\n")

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_training()
