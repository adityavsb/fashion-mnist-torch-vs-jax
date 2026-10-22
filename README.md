# M4 Mac Mini: Benchmarking Apple Silicon GPU Acceleration (PyTorch vs JAX)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(MPS%20Accelerated)-EE4C2C.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-0.12%20(XLA%20Compiled)-008080.svg)](https://jax.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4%20Mac%20mini-black.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical hobby project testing the real-world deep learning performance and GPU acceleration of the **Apple M4 Mac mini**, comparing **PyTorch (Metal Performance Shaders / MPS)** against **JAX (XLA JIT compiler)** across three incremental iterations:

1. **Iteration 1 (Baseline):** Lightweight MLP & kernel dispatch latency.
2. **Iteration 2 (Scaling):** Deep CNN & batch size throughput sweep (64 to 512).
3. **Iteration 3 (Advanced Optimization):** Graph compilation (`torch.compile` vs. `@jax.jit`) and mixed precision (FP32 vs. FP16 AMP) with unified memory profiling.

---

## The Motivation: Testing Apple Silicon M4 Hype vs. Reality

Apple's launch of the **M4 Mac mini** came with significant claims about massive GPU gains, unified memory bandwidth, and on-device machine learning acceleration. 

As a practitioner, I wanted to see past the marketing announcements and run an **honest, empirical benchmark** on my own desk:
* *Does Apple's Metal Performance Shaders (MPS) backend actually deliver tangible GPU speedups for everyday deep learning tasks?*
* *How does PyTorch's new `torch.compile` Inductor backend perform on Apple Silicon compared to eager PyTorch and JAX's mature XLA compiler?*
* *Does half-precision FP16 mixed precision yield throughput and memory benefits on unified memory?*
* *Where is the real crossover point where the GPU outpaces the CPU?*

For detailed technical trade-offs (including why JAX is evaluated on XLA CPU and the root cause of FP16 AMP overhead on unified memory), see [Architecture Decision Records (docs/decisions.md)](docs/decisions.md).

---

## Testbench & Hardware Specifications

```text
Machine:             Apple Mac mini (Late 2024)
SoC:                 Apple M4
Unified Memory:      16.00 GB
OS / Platform:       macOS Darwin (arm64)
Python Runtime:      Python 3.13.6
PyTorch Backend:     torch 2.14.1 (MPS - Metal Performance Shaders)
JAX Backend:         jax 0.12.0 / flax 0.12.10 (XLA CPU Vectorized)
```

---

## Benchmark Results Across 3 Iterations

### Iteration 3: Compilers (`torch.compile`) & Mixed Precision (FP16)

Tested on the Deep CNN architecture at batch size 256 for 3 epochs:

#### Experiment A: The Compiler Face-Off (Eager vs. `torch.compile` vs. JAX)

| Engine | Backend | Warmup Epoch 1 | Steady-State Throughput | Speedup vs Eager |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch Eager** | Apple M4 MPS GPU | `5.87s` | `11,761 img/s` | `1.00x` (baseline) |
| **PyTorch `torch.compile`** | Apple M4 MPS GPU | `5.86s` | **`13,167 img/s`** | **`1.12x` (+12% faster)** 🚀 |
| **JAX `@jax.jit`** | XLA Compiled CPU | `9.52s` | `6,333 img/s` | `0.54x` |

#### Experiment B: Mixed Precision (FP32 vs. FP16 AMP) & Memory Footprint

| Configuration | Throughput (img/s) | Speedup Factor | Peak GPU Memory Allocated |
| :--- | :---: | :---: | :---: |
| **PyTorch FP32 (MPS)** | `11,761 img/s` | `1.00x` | `6.4 MB` |
| **PyTorch FP16 AMP (MPS)** | `11,284 img/s` | `0.96x` | `6.4 MB` |
| **JAX FP32 (XLA)** | `6,333 img/s` | `1.00x` | *System RAM* |
| **JAX FP16 (XLA)** | `6,674 img/s` | `1.05x` (+5.4%) | *System RAM* |

![Iteration 3 Benchmark](assets/benchmark_iteration3.png)

> **Key Systems Takeaways from Iteration 3:**
> 1. **`torch.compile` works on Apple Silicon MPS:** Compiling with TorchDynamo and the MPS backend delivers a **+12% steady-state throughput boost**, pushing training past **13,100 images/second** with virtually zero initial warmup penalty ($5.86$s vs $5.87$s).
> 2. **FP16 on Small Models:** Because our CNN has a tiny footprint ($6.4$ MB of weights and activations), unified memory bandwidth is not the bottleneck. The overhead of software dynamic casting in `torch.autocast` slightly offsets ALU gains on smaller models. FP16 shines most when models exceed cache capacities (e.g. LLMs / ViTs).

---

### Iteration 2: Deep CNN & Batch Size Scaling (Heavy Compute)

In this iteration, the workload was scaled to a compute-intensive **Deep Convolutional Neural Network (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\times 2 \rightarrow$ Dense)** with a batch size sweep from $64$ to $512$ on Fashion-MNIST:

| Batch Size | PyTorch (M4 GPU via MPS) | JAX (Fused XLA `@jax.jit`) | Winner / Ratio |
| :---: | :---: | :---: | :---: |
| **64** | `5,392 img/s` | **`7,908 img/s`** | **JAX is 1.47x faster** ⚡ (low dispatch overhead) |
| **128** | **`10,418 img/s`** | `7,173 img/s` | **PyTorch GPU is 1.45x faster** |
| **256** | **`11,584 img/s`** | `6,706 img/s` | **PyTorch GPU is 1.73x faster** |
| **512** | **`12,606 img/s`** | `6,012 img/s` | **PyTorch GPU is 2.10x faster** 🚀 |

![Scaling Benchmark Comparison](assets/benchmark_scaling.png)

#### The GPU Crossover Point:
* **Small Batches ($\le 64$):** GPU kernel dispatch latency over Metal command buffers throttles the GPU. JAX's `@jax.jit` wins because it fuses operations into a single kernel executing within the Apple M4's low-latency CPU cache.
* **Large Batches ($\ge 128$):** The compute volume fully saturates the M4 GPU's parallel SIMD execution units. PyTorch on MPS scales from **$5,392$ to $12,606$ images/sec** ($2.34\times$ throughput scaling), outperforming compiled CPU execution by more than **$2.1\times$**.

---

### Iteration 1: Lightweight MLP Baseline

To establish a baseline, I first trained an identical 3-layer Multi-Layer Perceptron ($784 \rightarrow 256 \rightarrow 256 \rightarrow 10$) with the Adam optimizer ($lr=10^{-3}$) for 5 epochs:

| Metric | PyTorch (MPS GPU) | JAX (`@jax.jit`) | Advantage |
| :--- | :--- | :--- | :--- |
| **Final Train Loss** | `0.2828` | `0.2819` | Numerical Parity ($\Delta = 0.0009$) |
| **Final Test Accuracy** | `87.72%` | `88.02%` | Numerical Parity |
| **Average Time / Epoch** | `3.58s` | `1.69s` | **JAX is 2.11x faster** |

![MLP Benchmark Comparison](assets/benchmark_results.png)

---

## Architectural Parity: PyTorch vs JAX

Both models use mathematically equivalent initializations, loss formulations, and hyperparameters.

### PyTorch (Stateful Object-Oriented + Compiler & AMP)
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        return self.fc(torch.flatten(F.relu(self.bn(self.conv(x))), 1))

# Optional compilation & mixed precision
model = torch.compile(model.to("mps"))
with torch.autocast(device_type="mps", dtype=torch.float16):
    loss = criterion(model(images), labels)
loss.backward()
optimizer.step()
```

### JAX / Flax (Pure Functional Transforms + XLA Compilation)
```python
class SimpleCNN(nn.Module):
    dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
        return nn.Dense(features=10, dtype=self.dtype)(x.reshape((x.shape[0], -1)))

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x, train=True, mutable=['batch_stats'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, updates
    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state.replace(batch_stats=updates['batch_stats']), loss
```

---

## How to Reproduce

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/adityavsb/fashion-mnist-torch-vs-jax.git
cd fashion-mnist-torch-vs-jax

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Hardware Acceleration on Your Mac

```bash
python device_check.py
```

### 3. Run Benchmark Suites

```bash
# Run Iteration 3: Compilers (torch.compile vs JAX JIT) & FP16 Mixed Precision
python benchmark.py
```

### 4. Run Individual Implementations

```bash
# PyTorch with torch.compile on Apple Silicon MPS GPU
python -c "from torch_train import run_pytorch; run_pytorch(model_type='cnn', use_compile=True, batch_size=256)"

# PyTorch with FP16 Mixed Precision (AMP)
python -c "from torch_train import run_pytorch; run_pytorch(model_type='cnn', use_amp=True, batch_size=256)"

# JAX / Flax compiled with XLA (FP16)
python -c "from jax_train import run_jax; run_jax(model_type='cnn', use_fp16=True, batch_size=256)"
```

---

## Summary Verdict on the M4 Mac Mini

* **The M4 GPU is genuinely fast for compute-heavy workloads:** At batch size $128+$ on convolutional networks, Apple's Metal Performance Shaders deliver over **12,600 images/second**, running more than **$2\times$ faster** than CPU execution.
* **`torch.compile` is ready on Apple Silicon:** Enabling `torch.compile` gave an immediate **+12% speedup**, pushing throughput past **13,100 images/second** with negligible compilation overhead.
* **Watch out for dispatch latency on small models:** If your network is small (e.g. shallow MLPs) or batch size is tiny ($\le 64$), GPU dispatch overhead is noticeable, and compiled XLA execution on the M4 CPU is faster.
* **Verdict:** The M4 Mac mini is a highly capable, silent, and power-efficient local deep learning testbench for rapid prototyping before scaling to cloud clusters.

---

## License

This project is licensed under the [MIT License](LICENSE).
