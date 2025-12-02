import subprocess
import time
import psutil

# Number of images in CIFAR-10 training set
TRAIN_DATASET_SIZE = 50000

BATCH_SIZE = 128
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
        "python", "train_ddim.py",
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--ckpt-dir", f"weights/ddim_{i}",
        "--sample-dir", "generated_samples_ddim"
        ]
        print(f"\n=== Starting training run {i} ===")

        # Time tracking
        start_wall = time.time()
        start_cpu = time.process_time()

        # RAM tracking
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024**2)  # MB

        # Run training
        subprocess.run(command, check=True)

        end_wall = time.time()
        end_cpu = time.process_time()
        end_mem = process.memory_info().rss / (1024**2)

        wall_time = end_wall - start_wall
        cpu_time = end_cpu - start_cpu
        mem_change = end_mem - start_mem

        total_samples = TRAIN_DATASET_SIZE * EPOCHS
        throughput_sps = total_samples / wall_time

        results.append({
            "run": i,
            "wall_time_sec": wall_time,
            "cpu_time_sec": cpu_time,
            "memory_change_mb": mem_change,
            "throughput_sps": throughput_sps
        })

        print(f"Training has run {ordinal(i)}.")

        # Append run results to the text file
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"Run {i}:\n")
            f.write(f"  Wall time:  {wall_time:.2f} sec\n")
            f.write(f"  CPU time:   {cpu_time:.2f} sec\n")
            f.write(f"  Memory Δ:   {mem_change:.2f} MB\n")
            f.write(f"  Throughput: {throughput_sps:.2f} samples/sec\n\n")

    # Summary printed to console
    print("\n=========== SUMMARY ===========")
    for r in results:
        print(f"Run {r['run']}:")
        print(f"  Wall time:  {r['wall_time_sec']:.2f} sec")
        print(f"  CPU time:   {r['cpu_time_sec']:.2f} sec")
        print(f"  Memory Δ:   {r['memory_change_mb']:.2f} MB")
        print(f"  Throughput: {r['throughput_sps']:.2f} samples/sec\n")

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_training()
