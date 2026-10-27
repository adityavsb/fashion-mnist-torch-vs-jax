"""
Default benchmark runner for fashion-mnist-torch-vs-jax.
Executes Iteration 3 (Compilers & Mixed Precision) suite with N=5 isolated runs.
Individual iteration drivers:
  python benchmark_iter1_mlp.py
  python benchmark_iter2_scaling.py
  python benchmark_iter3_compile_amp.py
"""
from benchmark_iter3_compile_amp import run_iteration3_benchmark

if __name__ == "__main__":
    run_iteration3_benchmark(dataset_name="FashionMNIST", batch_size=256, epochs=3, n=5)
