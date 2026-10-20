# M4 Mac Mini: Benchmarking Apple Silicon GPU Acceleration (PyTorch vs JAX)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(MPS%20Accelerated)-EE4C2C.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-0.12%20(XLA%20Compiled)-008080.svg)](https://jax.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4%20Mac%20mini-black.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical hobby project testing the real-world deep learning performance and GPU acceleration of the **Apple M4 Mac mini**, comparing **PyTorch (Metal Performance Shaders / MPS)** against **JAX (XLA JIT compiler)**.

---

## The Motivation: Testing Apple Silicon M4 Hype vs. Reality

Apple's launch of the **M4 Mac mini** came with significant claims about massive GPU gains, unified memory bandwidth, and on-device machine learning acceleration. 

As a practitioner, I wanted to see past the marketing announcements and run an **honest, empirical benchmark** on my own desk:
* *Does Apple's Metal Performance Shaders (MPS) backend actually deliver tangible GPU speedups for everyday deep learning tasks?*
* *How does PyTorch's GPU acceleration compare against JAX's fused XLA machine code compiling directly to the Apple M4 ARM architecture?*
* *Where is the real crossover point where the GPU outpaces the CPU?*

To answer this, I built side-by-side implementations of identical neural network architectures and benchmarked them across two iterations: from a lightweight model to a compute-heavy scaled CNN with batch size sweeps.

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

## Benchmark Results

### Iteration 2: Deep CNN & Batch Size Scaling (Heavy Compute)

In this iteration, the workload was scaled to a compute-intensive **Deep Convolutional Neural Network (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\times 2 \rightarrow$ Dense)** with a batch size sweep from $64$ to $512$ on Fashion-MNIST:

| Batch Size | PyTorch (M4 GPU via MPS) | JAX (Fused XLA `@jax.jit`) | Winner / Ratio |
| :---: | :---: | :---: | :---: |
| **64** | `5,392 img/s` | **`7,908 img/s`** | **JAX is 1.47x faster** ⚡ (low dispatch overhead) |
| **128** | **`10,418 img/s`** | `7,173 img/s` | **PyTorch GPU is 1.45x faster** |
| **256** | **`11,584 img/s`** | `6,706 img/s` | **PyTorch GPU is 1.73x faster** |
| **512** | **`12,606 img/s`** | `6,012 img/s` | **PyTorch GPU is 2.10x faster** 🚀 |

![Scaling Benchmark Comparison](assets/benchmark_scaling.png)

#### Key Systems Insight & The Crossover Point:
* **The GPU Crossover Point happens between batch size 64 and 128.**
* **Small Batches ($\le 64$):** GPU kernel dispatch latency over Metal command buffers throttles the GPU. JAX's `@jax.jit` wins because it fuses forward propagation, autograd, and Adam momentum updates into a single machine code kernel that executes entirely within the Apple M4's low-latency CPU cache.
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

*Takeaway:* For small networks with lightweight matrix multiplications, moving tensors across Metal GPU buffers is slower than executing a fused JIT loop on the M4 CPU.

---

## Architectural Parity: PyTorch vs JAX

Both models use mathematically equivalent initializations, loss formulations, and hyperparameters.

### PyTorch (Stateful Object-Oriented)
```python
# Model owns mutable parameters and state
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        return self.fc(torch.flatten(F.relu(self.bn(self.conv(x))), 1))

# Training step mutates parameters in-place
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### JAX / Flax (Pure Functional Transforms)
```python
# Model is a pure stateless mathematical transformation
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.Dense(features=10)(x.reshape((x.shape[0], -1)))

# Training step is a compiled mathematical function returning new state
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, 
            x, train=True, mutable=['batch_stats']
        )
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

### 3. Run the Multi-Scale Benchmark Suite

```bash
# Sweeps batch sizes (64 to 512), benchmarks throughput, and generates scaling plots:
python benchmark.py
```

### 4. Run Individual Implementations

```bash
# PyTorch on Apple Silicon MPS GPU
python torch_train.py

# JAX / Flax compiled with XLA
python jax_train.py
```

---

## Summary Verdict on the M4 Mac Mini

* **The M4 GPU is genuinely fast for compute-heavy workloads:** Once batch size hits $128+$ on convolutional networks, Apple's Metal Performance Shaders deliver over **12,600 images/second**, running more than **$2\times$ faster** than CPU execution.
* **Watch out for dispatch latency on small models:** If your network is small (e.g. shallow MLPs) or your batch size is small ($\le 64$), Apple Silicon GPU dispatch overhead is noticeable, and compiled XLA execution on the M4 CPU is faster.
* **Verdict:** The M4 Mac mini is a highly capable, silent, and power-efficient local deep learning testbench for rapid prototyping before scaling to cloud clusters.

---

## License

This project is licensed under the [MIT License](LICENSE).
