"""
Comprehensive Benchmark Runner: PyTorch vs JAX.
Runs both implementations on identical datasets and hyperparameters,
compares runtime speed, memory, and accuracy, and produces comparison plots.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_train import run_pytorch
from jax_train import run_jax

def run_benchmark(epochs=5, batch_size=64, lr=1e-3, output_plot="assets/benchmark_results.png"):
    print("=" * 65)
    print(f"BENCHMARK: PyTorch vs JAX on Fashion-MNIST")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)

    # 1. Run PyTorch
    torch_history = run_pytorch(epochs=epochs, batch_size=batch_size, lr=lr)

    # 2. Run JAX
    jax_history = run_jax(epochs=epochs, batch_size=batch_size, lr=lr)

    # 3. Print Summary Table
    print("\n" + "=" * 65)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Metric':<25} | {'PyTorch':<15} | {'JAX':<15}")
    print("-" * 65)
    avg_torch_time = float(np.mean(torch_history["epoch_times"]))
    avg_jax_time = float(np.mean(jax_history["epoch_times"]))
    print(f"{'Final Train Loss':<25} | {torch_history['train_loss'][-1]:<15.4f} | {jax_history['train_loss'][-1]:<15.4f}")
    print(f"{'Final Test Accuracy (%)':<25} | {torch_history['test_acc'][-1]:<15.2f} | {jax_history['test_acc'][-1]:<15.2f}")
    print(f"{'Avg Time / Epoch (s)':<25} | {avg_torch_time:<15.2f} | {avg_jax_time:<15.2f}")
    speedup = avg_torch_time / avg_jax_time if avg_jax_time > 0 else 1.0
    print(f"{'Relative Speed':<25} | {'1.00x (baseline)':<15} | {f'{speedup:.2f}x':<15}")
    print("=" * 65)

    # 4. Generate Visual Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs_range = list(range(1, epochs + 1))

    # Loss plot
    axes[0].plot(epochs_range, torch_history["train_loss"], "o-", label="PyTorch (MPS/CPU)", color="#EE4C2C", lw=2)
    axes[0].plot(epochs_range, jax_history["train_loss"], "s-", label="JAX (JIT)", color="#008080", lw=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs_range, torch_history["test_acc"], "o-", label="PyTorch", color="#EE4C2C", lw=2)
    axes[1].plot(epochs_range, jax_history["test_acc"], "s-", label="JAX", color="#008080", lw=2)
    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Time per epoch bar chart
    x_pos = np.arange(epochs)
    width = 0.35
    axes[2].bar(x_pos - width / 2, torch_history["epoch_times"], width, label="PyTorch", color="#EE4C2C")
    axes[2].bar(x_pos + width / 2, jax_history["epoch_times"], width, label="JAX", color="#008080")
    axes[2].set_title("Time per Epoch (Seconds)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f"E{e}" for e in epochs_range])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\n[✓] Comparison plot saved to: {os.path.abspath(output_plot)}")

if __name__ == "__main__":
    run_benchmark(epochs=5)
