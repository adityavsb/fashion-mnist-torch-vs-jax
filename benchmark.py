"""
Iteration 3 Benchmark Suite: Compilers & Mixed Precision on Apple M4 Mac Mini.
Experiments:
1. Compiler Face-Off: PyTorch Eager (MPS) vs. PyTorch torch.compile vs. JAX @jax.jit
2. Mixed Precision: Standard FP32 vs. Mixed Precision FP16 (AMP / float16) & Memory Footprint
Outputs charts to assets/benchmark_iteration3.png and prints analytical summary tables.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_train import run_pytorch
from jax_train import run_jax

def run_iteration3_benchmark(dataset_name="FashionMNIST", batch_size=256, epochs=3, output_plot="assets/benchmark_iteration3.png"):
    print("=" * 80)
    print(f"ITERATION 3: COMPILERS & MIXED PRECISION BENCHMARK (Apple M4 Mac Mini)")
    print(f"Dataset: {dataset_name} | Architecture: Deep CNN | Batch Size: {batch_size}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Experiment A: Compiler Face-Off (Eager vs torch.compile vs JAX JIT)
    # -------------------------------------------------------------
    print("\n>>> Experiment A: Compiler Face-Off (Warmup vs Steady-State) <<<")
    
    print("\n[A.1] Running PyTorch Eager (MPS)...")
    torch_eager = run_pytorch(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=False)

    print("\n[A.2] Running PyTorch torch.compile (MPS)...")
    torch_compiled = run_pytorch(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=True)

    print("\n[A.3] Running JAX @jax.jit (XLA)...")
    jax_compiled = run_jax(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_fp16=False)

    # -------------------------------------------------------------
    # Experiment B: Precision & Memory (FP32 vs FP16)
    # -------------------------------------------------------------
    print("\n>>> Experiment B: Mixed Precision FP16 (AMP) & Memory Allocation <<<")
    
    print("\n[B.1] Running PyTorch FP16 (autocast on MPS)...")
    torch_fp16 = run_pytorch(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_compile=False, use_amp=True)

    print("\n[B.2] Running JAX FP16 (jnp.float16)...")
    jax_fp16 = run_jax(model_type="cnn", dataset_name=dataset_name, epochs=epochs, batch_size=batch_size, use_fp16=True)

    # -------------------------------------------------------------
    # Summary Table A: Compiler Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT A: COMPILER FACE-OFF SUMMARY")
    print("=" * 80)
    print(f"{'Engine':<25} | {'Warmup Ep 1 (s)':<18} | {'Steady-State (img/s)':<22} | {'Steady-State Speedup':<20}")
    print("-" * 80)
    eager_tput = torch_eager["throughputs"][-1]
    comp_tput = torch_compiled["throughputs"][-1]
    jax_tput = jax_compiled["throughputs"][-1]

    print(f"{'PyTorch Eager (MPS)':<25} | {torch_eager['epoch_times'][0]:<18.2f} | {eager_tput:<22.0f} | {'1.00x (baseline)':<20}")
    print(f"{'PyTorch torch.compile':<25} | {torch_compiled['epoch_times'][0]:<18.2f} | {comp_tput:<22.0f} | {f'{comp_tput/eager_tput:.2f}x':<20}")
    print(f"{'JAX @jax.jit (XLA)':<25} | {jax_compiled['epoch_times'][0]:<18.2f} | {jax_tput:<22.0f} | {f'{jax_tput/eager_tput:.2f}x':<20}")

    # -------------------------------------------------------------
    # Summary Table B: Precision & Memory Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT B: MIXED PRECISION & MEMORY FOOTPRINT SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'Throughput (img/s)':<20} | {'Speedup':<12} | {'Peak GPU Memory (MB)':<20}")
    print("-" * 80)
    t_fp32_tput = torch_eager["throughputs"][-1]
    t_fp16_tput = torch_fp16["throughputs"][-1]
    j_fp32_tput = jax_compiled["throughputs"][-1]
    j_fp16_tput = jax_fp16["throughputs"][-1]

    print(f"{'PyTorch FP32 (MPS)':<25} | {t_fp32_tput:<20.0f} | {'1.00x':<12} | {torch_eager['peak_mem_mb']:<20.1f}")
    print(f"{'PyTorch FP16 AMP (MPS)':<25} | {t_fp16_tput:<20.0f} | {f'{t_fp16_tput/t_fp32_tput:.2f}x':<12} | {torch_fp16['peak_mem_mb']:<20.1f}")
    print(f"{'JAX FP32 (XLA)':<25} | {j_fp32_tput:<20.0f} | {'1.00x':<12} | {'N/A (CPU RAM)':<20}")
    print(f"{'JAX FP16 (XLA)':<25} | {j_fp16_tput:<20.0f} | {f'{j_fp16_tput/j_fp32_tput:.2f}x':<12} | {'N/A (CPU RAM)':<20}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Visualization: 2-Panel Diagnostic Chart
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Compiler Face-Off (Steady State Throughput)
    comp_labels = ["PyTorch Eager\n(MPS)", "PyTorch Compiled\n(torch.compile)", "JAX Compiled\n(@jax.jit)"]
    comp_values = [eager_tput, comp_tput, jax_tput]
    colors_a = ["#EE4C2C", "#FF7A59", "#008080"]
    bars_a = axes[0].bar(comp_labels, comp_values, color=colors_a, width=0.45)
    axes[0].set_title("Experiment A: Compiler Face-Off (Throughput)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Steady-State Throughput (Images / Sec)", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")
    for b in bars_a:
        yval = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, yval + 150, f"{yval:.0f} img/s", ha="center", va="bottom", fontweight="bold")

    # Panel 2: FP32 vs FP16 Speedup & Memory
    prec_labels = ["PyTorch FP32", "PyTorch FP16\n(AMP)", "JAX FP32", "JAX FP16"]
    prec_values = [t_fp32_tput, t_fp16_tput, j_fp32_tput, j_fp16_tput]
    colors_b = ["#EE4C2C", "#C43314", "#008080", "#005959"]
    bars_b = axes[1].bar(prec_labels, prec_values, color=colors_b, width=0.45)
    axes[1].set_title("Experiment B: Mixed Precision (FP32 vs FP16)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Throughput (Images / Sec)", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")
    for b in bars_b:
        yval = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, yval + 150, f"{yval:.0f} img/s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    print(f"\n[✓] Iteration 3 plot saved to: {os.path.abspath(output_plot)}")

if __name__ == "__main__":
    run_iteration3_benchmark(dataset_name="FashionMNIST", batch_size=256, epochs=3)
