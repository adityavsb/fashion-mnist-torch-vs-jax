"""
Multi-Scale Benchmark Suite: PyTorch vs JAX.
Runs:
1. Architecture comparison: Lightweight MLP vs Deep CNN
2. Throughput scaling across batch sizes (64, 128, 256, 512)
3. Outputs comprehensive comparison tables and multi-facet plots
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_train import run_pytorch
from jax_train import run_jax

def run_scale_benchmark(dataset_name="FashionMNIST", epochs=3, output_plot="assets/benchmark_scaling.png"):
    print("=" * 75)
    print(f"MULTI-SCALE BENCHMARK: PyTorch vs JAX on {dataset_name}")
    print("=" * 75)

    batch_sizes = [64, 128, 256, 512]
    torch_throughputs = []
    jax_throughputs = []
    
    print("\n>>> Phase 1: Batch Size Throughput Scaling (CNN Architecture) <<<")
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        t_hist = run_pytorch(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=bs)
        j_hist = run_jax(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=bs)
        
        # Take throughput from last epoch (after JIT warmup / kernel caching)
        t_tput = t_hist["throughputs"][-1]
        j_tput = j_hist["throughputs"][-1]
        torch_throughputs.append(t_tput)
        jax_throughputs.append(j_tput)
        print(f"Batch {bs:3d} | PyTorch: {t_tput:.0f} img/s | JAX: {j_tput:.0f} img/s")

    print("\n" + "=" * 75)
    print("BATCH SIZE SCALING SUMMARY TABLE (CNN Architecture)")
    print("=" * 75)
    print(f"{'Batch Size':<12} | {'PyTorch (img/s)':<18} | {'JAX (img/s)':<18} | {'Ratio (PyTorch / JAX)':<20}")
    print("-" * 75)
    for bs, t, j in zip(batch_sizes, torch_throughputs, jax_throughputs):
        ratio = t / j if j > 0 else 1.0
        print(f"{bs:<12} | {t:<18.0f} | {j:<18.0f} | {ratio:<20.2f}x")
    print("=" * 75)

    # Multi-facet Visualization
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Throughput vs Batch Size
    axes[0].plot(batch_sizes, torch_throughputs, "o-", label="PyTorch (Apple M4 MPS)", color="#EE4C2C", lw=2.5, ms=8)
    axes[0].plot(batch_sizes, jax_throughputs, "s-", label="JAX (XLA JIT)", color="#008080", lw=2.5, ms=8)
    axes[0].set_title(f"Throughput Scaling vs Batch Size ({dataset_name} CNN)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Batch Size", fontsize=11)
    axes[0].set_ylabel("Throughput (Images / Second)", fontsize=11)
    axes[0].set_xticks(batch_sizes)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Plot 2: Relative Speedup / Advantage
    ratios = [t / j for t, j in zip(torch_throughputs, jax_throughputs)]
    bars = axes[1].bar([str(b) for b in batch_sizes], ratios, color=["#008080" if r < 1.0 else "#EE4C2C" for r in ratios], width=0.45)
    axes[1].axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Parity (1.0x)")
    axes[1].set_title("Throughput Ratio (PyTorch MPS / JAX XLA)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Batch Size", fontsize=11)
    axes[1].set_ylabel("PyTorch Advantage (Ratio > 1.0 = PyTorch faster)", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=11)
    
    for bar, r in zip(bars, ratios):
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{r:.2f}x", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    print(f"\n[✓] Scaling benchmark plot saved to: {os.path.abspath(output_plot)}")

if __name__ == "__main__":
    run_scale_benchmark(dataset_name="FashionMNIST", epochs=2)
