"""
Iteration 2 Benchmark Suite: Two-Layer CNN Batch Size Scaling (N=5 Isolated Runs).
Sweeps batch sizes: 64, 128, 256, 512 on Fashion-MNIST with SimpleCNN.
Outputs plot to assets/benchmark_scaling.png and serialized data to results/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from bench_utils import SCRIPT_DIR, repeat, save_results

def run_scale_benchmark(dataset_name="FashionMNIST", epochs=2, n=5, output_plot="assets/benchmark_scaling.png"):
    if not os.path.isabs(output_plot):
        output_plot = os.path.join(SCRIPT_DIR, output_plot)

    print("=" * 80)
    print(f"ITERATION 2: BATCH SIZE SCALING BENCHMARK (N={n} Isolated Subprocess Runs)")
    print(f"Dataset: {dataset_name} | Architecture: Two-layer CNN | Epochs: {epochs}")
    print("=" * 80)

    batch_sizes = [64, 128, 256, 512]
    torch_summaries = []
    jax_summaries = []
    
    print("\n>>> Phase 1: Batch Size Throughput Scaling (CNN Architecture) <<<")
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        print(f"  Running PyTorch (MPS) at bs={bs}...")
        t_summary = repeat("torch_train", "run_pytorch", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=bs)
        print(f"  Running JAX (XLA) at bs={bs}...")
        j_summary = repeat("jax_train", "run_jax", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=bs)
        
        torch_summaries.append(t_summary)
        jax_summaries.append(j_summary)
        print(f"Batch {bs:3d} | PyTorch median: {t_summary['steady_median']:.0f} img/s | JAX median: {j_summary['steady_median']:.0f} img/s")

    print("\n" + "=" * 90)
    print("BATCH SIZE SCALING SUMMARY TABLE (Two-layer CNN, N=5 Independent Runs)")
    print("=" * 90)
    print(f"{'Batch Size':<12} | {'PyTorch (img/s)':<25} | {'JAX (img/s)':<25} | {'Winner / Ratio':<20}")
    print("-" * 90)
    for bs, t_sum, j_sum in zip(batch_sizes, torch_summaries, jax_summaries):
        t_val = t_sum["steady_median"]
        j_val = j_sum["steady_median"]
        if t_val >= j_val:
            ratio = t_val / j_val if j_val > 0 else 1.0
            winner_str = f"PyTorch {ratio:.2g}x"
        else:
            ratio = j_val / t_val if t_val > 0 else 1.0
            winner_str = f"JAX {ratio:.2g}x"

        t_str = f"{t_val:.0f} [{t_sum['steady_min']:.0f}–{t_sum['steady_max']:.0f}]"
        j_str = f"{j_val:.0f} [{j_sum['steady_min']:.0f}–{j_sum['steady_max']:.0f}]"
        print(f"{bs:<12} | {t_str:<25} | {j_str:<25} | {winner_str:<20}")
    print("=" * 90)

    # Multi-facet Visualization
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t_medians = [s["steady_median"] for s in torch_summaries]
    j_medians = [s["steady_median"] for s in jax_summaries]

    t_err_low = [s["steady_median"] - s["steady_min"] for s in torch_summaries]
    t_err_high = [s["steady_max"] - s["steady_median"] for s in torch_summaries]
    j_err_low = [s["steady_median"] - s["steady_min"] for s in jax_summaries]
    j_err_high = [s["steady_max"] - s["steady_median"] for s in jax_summaries]

    # Plot 1: Throughput vs Batch Size
    axes[0].errorbar(batch_sizes, t_medians, yerr=[t_err_low, t_err_high], fmt="o-", label="PyTorch (Apple M4 MPS)", color="#EE4C2C", lw=2.5, ms=8, capsize=5)
    axes[0].errorbar(batch_sizes, j_medians, yerr=[j_err_low, j_err_high], fmt="s-", label="JAX (XLA JIT)", color="#008080", lw=2.5, ms=8, capsize=5)
    axes[0].set_title(f"Throughput Scaling vs Batch Size ({dataset_name} Two-layer CNN, N=5)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Batch Size", fontsize=11)
    axes[0].set_ylabel("Steady-State Throughput (Images / Second)", fontsize=11)
    axes[0].set_xticks(batch_sizes)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Plot 2: Relative Speedup / Advantage
    ratios = [t / j for t, j in zip(t_medians, j_medians)]
    bars = axes[1].bar([str(b) for b in batch_sizes], ratios, color=["#008080" if r < 1.0 else "#EE4C2C" for r in ratios], width=0.45)
    axes[1].axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Parity (1.0x)")
    axes[1].set_title("Throughput Ratio (PyTorch MPS / JAX XLA)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Batch Size", fontsize=11)
    axes[1].set_ylabel("PyTorch Advantage (Ratio > 1.0 = PyTorch faster)", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=11)
    
    for bar, r in zip(bars, ratios):
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{r:.2g}x", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    print(f"\n[✓] Scaling benchmark plot saved to: {output_plot}")

    # Save JSON artifact
    payload = {
        "n_runs": n,
        "epochs": epochs,
        "batch_sizes": batch_sizes,
        "torch_summaries": torch_summaries,
        "jax_summaries": jax_summaries,
        "ratios": ratios,
    }
    save_results("iteration2", payload)
    return payload

if __name__ == "__main__":
    run_scale_benchmark(dataset_name="FashionMNIST", epochs=2, n=5)
