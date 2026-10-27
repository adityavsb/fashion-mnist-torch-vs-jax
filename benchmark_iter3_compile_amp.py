"""
Iteration 3 Benchmark Suite: Compilers & Mixed Precision (N=5 Isolated Runs).
Evaluates:
1. Compiler Face-Off: PyTorch Eager (MPS) vs PyTorch torch.compile vs JAX @jax.jit
2. Mixed Precision: FP32 vs FP16 AMP & Memory Footprint
Outputs plot to assets/benchmark_iteration3.png and serialized data to results/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from bench_utils import SCRIPT_DIR, repeat, save_results

def run_iteration3_benchmark(dataset_name="FashionMNIST", batch_size=256, epochs=3, n=5, output_plot="assets/benchmark_iteration3.png"):
    if not os.path.isabs(output_plot):
        output_plot = os.path.join(SCRIPT_DIR, output_plot)

    print("=" * 80)
    print(f"ITERATION 3: COMPILERS & MIXED PRECISION (N={n} Isolated Subprocess Runs)")
    print(f"Dataset: {dataset_name} | Architecture: Two-layer CNN | Batch Size: {batch_size}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Experiment A: Compiler Face-Off
    # -------------------------------------------------------------
    print("\n>>> Experiment A: Compiler Face-Off (Warmup vs Steady-State) <<<")
    
    print("\n[A.1] Running PyTorch Eager (MPS)...")
    torch_eager = repeat("torch_train", "run_pytorch", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=False)

    print("\n[A.2] Running PyTorch torch.compile (MPS)...")
    torch_compiled = repeat("torch_train", "run_pytorch", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=True)

    print("\n[A.3] Running JAX @jax.jit (XLA)...")
    jax_compiled = repeat("jax_train", "run_jax", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_fp16=False)

    # -------------------------------------------------------------
    # Experiment B: Precision & Memory
    # -------------------------------------------------------------
    print("\n>>> Experiment B: Mixed Precision FP16 (AMP) & Memory Allocation <<<")
    
    print("\n[B.1] Running PyTorch FP16 (autocast on MPS)...")
    torch_fp16 = repeat("torch_train", "run_pytorch", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=False, use_amp=True)

    print("\n[B.2] Running JAX FP16 (jnp.float16)...")
    jax_fp16 = repeat("jax_train", "run_jax", n=n, model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_fp16=True)

    # -------------------------------------------------------------
    # Summary Table A: Compiler Comparison
    # -------------------------------------------------------------
    eager_tput = torch_eager["steady_median"]
    comp_tput = torch_compiled["steady_median"]
    jax_tput = jax_compiled["steady_median"]

    comp_speedup = comp_tput / eager_tput if eager_tput > 0 else 1.0
    jax_speedup = jax_tput / eager_tput if eager_tput > 0 else 1.0

    print("\n" + "=" * 115)
    print("EXPERIMENT A: COMPILER FACE-OFF SUMMARY (N=5 Independent Processes)")
    print("=" * 115)
    print(f"{'Engine':<25} | {'Warm-up cost (E1 − steady, s)':<30} | {'Steady-State (img/s)':<30} | {'Steady-State Speedup':<20}")
    print("-" * 115)
    eager_str = f"{eager_tput:.0f} [{torch_eager['steady_min']:.0f}–{torch_eager['steady_max']:.0f}]"
    comp_str = f"{comp_tput:.0f} [{torch_compiled['steady_min']:.0f}–{torch_compiled['steady_max']:.0f}]"
    jax_str = f"{jax_tput:.0f} [{jax_compiled['steady_min']:.0f}–{jax_compiled['steady_max']:.0f}]"

    print(f"{'PyTorch Eager (MPS)':<25} | {torch_eager['warmup_median_s']:<30.2f} | {eager_str:<30} | {'1.00x (baseline)':<20}")
    print(f"{'PyTorch torch.compile':<25} | {torch_compiled['warmup_median_s']:<30.2f} | {comp_str:<30} | {f'{comp_speedup:.2g}x':<20}")
    print(f"{'JAX @jax.jit (XLA)':<25} | {jax_compiled['warmup_median_s']:<30.2f} | {jax_str:<30} | {f'{jax_speedup:.2g}x':<20}")
    print("=" * 115)

    # -------------------------------------------------------------
    # Summary Table B: Precision & Memory Comparison
    # -------------------------------------------------------------
    t_fp32_tput = torch_eager["steady_median"]
    t_fp16_tput = torch_fp16["steady_median"]
    j_fp32_tput = jax_compiled["steady_median"]
    j_fp16_tput = jax_fp16["steady_median"]

    t_fp16_speedup = t_fp16_tput / t_fp32_tput if t_fp32_tput > 0 else 1.0
    j_fp16_speedup = j_fp16_tput / j_fp32_tput if j_fp32_tput > 0 else 1.0

    t_fp32_str = f"{t_fp32_tput:.0f} [{torch_eager['steady_min']:.0f}–{torch_eager['steady_max']:.0f}]"
    t_fp16_str = f"{t_fp16_tput:.0f} [{torch_fp16['steady_min']:.0f}–{torch_fp16['steady_max']:.0f}]"
    j_fp32_str = f"{j_fp32_tput:.0f} [{jax_compiled['steady_min']:.0f}–{jax_compiled['steady_max']:.0f}]"
    j_fp16_str = f"{j_fp16_tput:.0f} [{jax_fp16['steady_min']:.0f}–{jax_fp16['steady_max']:.0f}]"

    print("\n" + "=" * 125)
    print("EXPERIMENT B: MIXED PRECISION & MEMORY FOOTPRINT SUMMARY (N=5 Independent Processes)")
    print("=" * 125)
    print(f"{'Configuration':<25} | {'Throughput (img/s)':<30} | {'Speedup':<12} | {'Peak allocated in step (MB)':<30} | {'Resident state (MB)':<20}")
    print("-" * 125)
    print(f"{'PyTorch FP32 (MPS)':<25} | {t_fp32_str:<30} | {'1.00x':<12} | {torch_eager['peak_mem_mb']:<30.1f} | {torch_eager['resident_state_mb']:<20.1f}")
    print(f"{'PyTorch FP16 AMP (MPS)':<25} | {t_fp16_str:<30} | {f'{t_fp16_speedup:.2g}x':<12} | {torch_fp16['peak_mem_mb']:<30.1f} | {torch_fp16['resident_state_mb']:<20.1f}")
    print(f"{'JAX FP32 (XLA)':<25} | {j_fp32_str:<30} | {'1.00x':<12} | {'N/A (CPU RAM)':<30} | {'N/A (CPU RAM)':<20}")
    print(f"{'JAX FP16 (XLA)':<25} | {j_fp16_str:<30} | {f'{j_fp16_speedup:.2g}x':<12} | {'N/A (CPU RAM)':<30} | {'N/A (CPU RAM)':<20}")
    print("=" * 125)

    # -------------------------------------------------------------
    # Visualization: 2-Panel Diagnostic Chart with Error Bars (min/max)
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Compiler Face-Off
    comp_labels = ["PyTorch Eager\n(MPS)", "PyTorch Compiled\n(torch.compile)", "JAX Compiled\n(@jax.jit)"]
    comp_values = [eager_tput, comp_tput, jax_tput]
    comp_err_lower = [eager_tput - torch_eager["steady_min"], comp_tput - torch_compiled["steady_min"], jax_tput - jax_compiled["steady_min"]]
    comp_err_upper = [torch_eager["steady_max"] - eager_tput, torch_compiled["steady_max"] - comp_tput, jax_compiled["steady_max"] - jax_tput]
    colors_a = ["#EE4C2C", "#FF7A59", "#008080"]

    bars_a = axes[0].bar(comp_labels, comp_values, yerr=[comp_err_lower, comp_err_upper], capsize=5, color=colors_a, width=0.45)
    axes[0].set_title("Experiment A: Compiler Face-Off (Median Throughput, N=5)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Steady-State Throughput (Images / Sec)", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")
    for b, d in zip(bars_a, [torch_eager, torch_compiled, jax_compiled]):
        yval = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, d["steady_max"] + 150, f"{yval:.0f} img/s", ha="center", va="bottom", fontweight="bold")

    # Panel 2: FP32 vs FP16
    prec_labels = ["PyTorch FP32", "PyTorch FP16\n(AMP)", "JAX FP32", "JAX FP16"]
    prec_values = [t_fp32_tput, t_fp16_tput, j_fp32_tput, j_fp16_tput]
    prec_err_lower = [t_fp32_tput - torch_eager["steady_min"], t_fp16_tput - torch_fp16["steady_min"], j_fp32_tput - jax_compiled["steady_min"], j_fp16_tput - jax_fp16["steady_min"]]
    prec_err_upper = [torch_eager["steady_max"] - t_fp32_tput, torch_fp16["steady_max"] - t_fp16_tput, jax_compiled["steady_max"] - j_fp32_tput, jax_fp16["steady_max"] - j_fp16_tput]
    colors_b = ["#EE4C2C", "#C43314", "#008080", "#005959"]

    bars_b = axes[1].bar(prec_labels, prec_values, yerr=[prec_err_lower, prec_err_upper], capsize=5, color=colors_b, width=0.45)
    axes[1].set_title("Experiment B: Mixed Precision (Median Throughput, N=5)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Throughput (Images / Sec)", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")
    for b, d in zip(bars_b, [torch_eager, torch_fp16, jax_compiled, jax_fp16]):
        yval = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, d["steady_max"] + 150, f"{yval:.0f} img/s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    print(f"\n[✓] Iteration 3 plot saved to: {output_plot}")

    # Save JSON artifact
    payload = {
        "n_runs": n,
        "batch_size": batch_size,
        "epochs": epochs,
        "torch_eager": torch_eager,
        "torch_compiled": torch_compiled,
        "jax_compiled": jax_compiled,
        "torch_fp16": torch_fp16,
        "jax_fp16": jax_fp16,
    }
    save_results("iteration3", payload)
    return payload

if __name__ == "__main__":
    run_iteration3_benchmark(dataset_name="FashionMNIST", batch_size=256, epochs=3, n=5)
