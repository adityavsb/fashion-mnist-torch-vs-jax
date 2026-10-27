"""
Iteration 1 Benchmark Suite: Lightweight MLP Baseline (N=5 Isolated Runs).
Compares:
PyTorch (Apple M4 MPS GPU) vs JAX (XLA Compiled CPU) on a 3-layer MLP.
Hyperparameters: batch_size=64, epochs=5, lr=1e-3, Fashion-MNIST.
Outputs plot to assets/benchmark_results.png and serialized data to results/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics
from bench_utils import SCRIPT_DIR, repeat, save_results

def run_benchmark(epochs=5, batch_size=64, lr=1e-3, n=5, output_plot="assets/benchmark_results.png"):
    if not os.path.isabs(output_plot):
        output_plot = os.path.join(SCRIPT_DIR, output_plot)

    print("=" * 70)
    print("ITERATION 1: LIGHTWEIGHT MLP BASELINE (N=5 Isolated Subprocess Runs)")
    print(f"Dataset: FashionMNIST | Model: MLP | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)

    # 1. Run PyTorch (MPS)
    print("\n[1] Running PyTorch (MPS GPU)...")
    torch_summary = repeat("torch_train", "run_pytorch", n=n, model_type="mlp", dataset_name="FashionMNIST", epochs=epochs, batch_size=batch_size, lr=lr)

    # 2. Run JAX (XLA CPU)
    print("\n[2] Running JAX (XLA CPU)...")
    jax_summary = repeat("jax_train", "run_jax", n=n, model_type="mlp", dataset_name="FashionMNIST", epochs=epochs, batch_size=batch_size, lr=lr)

    # 3. Extract Statistics
    torch_runs = torch_summary["runs"]
    jax_runs = jax_summary["runs"]

    torch_loss_median = statistics.median([r["train_loss"][-1] for r in torch_runs])
    jax_loss_median = statistics.median([r["train_loss"][-1] for r in jax_runs])

    torch_acc_median = statistics.median([r["test_acc"][-1] for r in torch_runs])
    jax_acc_median = statistics.median([r["test_acc"][-1] for r in jax_runs])

    torch_avg_time_median = statistics.median([float(np.mean(r["epoch_times"])) for r in torch_runs])
    jax_avg_time_median = statistics.median([float(np.mean(r["epoch_times"])) for r in jax_runs])

    torch_tput_median = torch_summary["steady_median"]
    jax_tput_median = jax_summary["steady_median"]

    speedup = torch_avg_time_median / jax_avg_time_median if jax_avg_time_median > 0 else 1.0

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (Iteration 1: MLP Baseline, N=5)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'PyTorch (MPS GPU)':<22} | {'JAX (@jax.jit CPU)':<22}")
    print("-" * 80)
    print(f"{'Final Train Loss':<28} | {torch_loss_median:<22.4f} | {jax_loss_median:<22.4f}")
    print(f"{'Final Test Accuracy (%)':<28} | {torch_acc_median:<22.2f} | {jax_acc_median:<22.2f}")
    print(f"{'Avg Time / Epoch (s)':<28} | {torch_avg_time_median:<22.2f} | {jax_avg_time_median:<22.2f}")
    t_str = f"{torch_tput_median:.0f} [{torch_summary['steady_min']:.0f}–{torch_summary['steady_max']:.0f}]"
    j_str = f"{jax_tput_median:.0f} [{jax_summary['steady_min']:.0f}–{jax_summary['steady_max']:.0f}]"
    print(f"{'Steady-State (img/s)':<28} | {t_str:<22} | {j_str:<22}")
    print(f"{'Relative Speed (JAX / Torch)':<28} | {'1.00x (baseline)':<22} | {f'{speedup:.2g}x':<22}")
    print("=" * 80)

    # 4. Generate Visual Comparison Plot
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs_range = list(range(1, epochs + 1))

    # Representative curves from run 0
    t0 = torch_runs[0]
    j0 = jax_runs[0]

    # Loss plot
    axes[0].plot(epochs_range, t0["train_loss"], "o-", label="PyTorch (MPS)", color="#EE4C2C", lw=2)
    axes[0].plot(epochs_range, j0["train_loss"], "s-", label="JAX (JIT CPU)", color="#008080", lw=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs_range, t0["test_acc"], "o-", label="PyTorch (MPS)", color="#EE4C2C", lw=2)
    axes[1].plot(epochs_range, j0["test_acc"], "s-", label="JAX (JIT CPU)", color="#008080", lw=2)
    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Time per epoch bar chart (median epoch times across N runs)
    torch_epoch_medians = [statistics.median([r["epoch_times"][e] for r in torch_runs]) for e in range(epochs)]
    jax_epoch_medians = [statistics.median([r["epoch_times"][e] for r in jax_runs]) for e in range(epochs)]

    x_pos = np.arange(epochs)
    width = 0.35
    axes[2].bar(x_pos - width / 2, torch_epoch_medians, width, label="PyTorch (MPS)", color="#EE4C2C")
    axes[2].bar(x_pos + width / 2, jax_epoch_medians, width, label="JAX (JIT CPU)", color="#008080")
    axes[2].set_title("Median Time per Epoch (Seconds, N=5)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f"E{e}" for e in epochs_range])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\n[✓] Iteration 1 comparison plot saved to: {output_plot}")

    # Save JSON artifact
    payload = {
        "n_runs": n,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "torch_summary": torch_summary,
        "jax_summary": jax_summary,
        "torch_loss_median": torch_loss_median,
        "jax_loss_median": jax_loss_median,
        "torch_acc_median": torch_acc_median,
        "jax_acc_median": jax_acc_median,
        "torch_avg_time_median": torch_avg_time_median,
        "jax_avg_time_median": jax_avg_time_median,
        "speedup": speedup,
    }
    save_results("iteration1", payload)
    return payload

if __name__ == "__main__":
    run_benchmark(epochs=5, batch_size=64, n=5)
