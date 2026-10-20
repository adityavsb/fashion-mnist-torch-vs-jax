# Rosetta ML: PyTorch vs JAX Side-by-Side

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(MPS%20Ready)-EE4C2C.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-XLA%20Compiled-008080.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A side-by-side comparative Rosetta Stone contrasting **PyTorch** (imperative, object-oriented, stateful autograd) and **JAX** (pure functional, immutable pytrees, composable compiler transforms).

Each iteration implements identical architectures, datasets, and hyperparameters across both frameworks to provide concrete insights into **coding patterns**, **hardware execution**, and **runtime performance**.

---

## Benchmark Highlights

### Iteration 2: Scaled Deep CNN & Batch Size Throughput Sweep

In this iteration, we scaled from a simple MLP to a **Deep Convolutional Neural Network (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\times 2 \rightarrow$ Dense)** and swept batch sizes from $64$ to $512$ on an **Apple M4 Mac**:

| Batch Size | PyTorch (`mps` GPU) | JAX (`@jax.jit` CPU/XLA) | Advantage / Winner |
| :---: | :---: | :---: | :---: |
| **64** | `5,392 img/s` | **`7,908 img/s`** | **JAX is 1.47x faster** ⚡ (low dispatch overhead) |
| **128** | **`10,418 img/s`** | `7,173 img/s` | **PyTorch is 1.45x faster** (GPU parallel saturation) |
| **256** | **`11,584 img/s`** | `6,706 img/s` | **PyTorch is 1.73x faster** |
| **512** | **`12,606 img/s`** | `6,012 img/s` | **PyTorch is 2.10x faster** 🚀 |

![Scaling Benchmark Comparison](assets/benchmark_scaling.png)

> **The Systems Engineering Crossover Point:**
> * **At small batch sizes ($\le 64$):** Kernel dispatch latency over Metal command buffers limits GPU efficiency. JAX's `@jax.jit` wins because it fuses forward, backward, and optimizer updates into a single contiguous machine code kernel running inside the chip's ultra-low latency cache.
> * **At larger batch sizes ($\ge 128$):** The compute load saturates the GPU's matrix units. PyTorch's Metal Performance Shaders (`mps`) leverage the full parallel SIMD execution units of Apple Silicon, achieving **over 12,600 images/second** (a $2.34\times$ scaling gain from batch 64 to 512).

---

### Iteration 1: Lightweight MLP Baseline

Trained an identical 3-layer MLP ($784 \rightarrow 256 \rightarrow 256 \rightarrow 10$) with Adam optimizer ($lr=10^{-3}$) for 5 epochs:

| Metric | PyTorch (`mps` GPU) | JAX (`@jax.jit` CPU/XLA) | Advantage |
| :--- | :--- | :--- | :--- |
| **Final Train Loss** | `0.2828` | `0.2819` | Parity matched ($\Delta = 0.0009$) |
| **Final Test Accuracy** | `87.72%` | `88.02%` | Parity matched |
| **Average Time / Epoch** | `3.58s` | `1.69s` | **JAX is 2.11x faster** |

![MLP Benchmark Comparison](assets/benchmark_results.png)

---

## Core Paradigm Comparison

### PyTorch (Stateful Object-Oriented)
```python
# Model owns mutable state
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# Training step mutates parameters in-place
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### JAX / Flax (Pure Functional Transforms)
```python
# Model is a pure stateless transformation
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)

# Training step is a compiled mathematical function returning new state
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

## Quickstart

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

### 2. Verify Hardware Acceleration

```bash
python device_check.py
```

### 3. Run Multi-Scale Benchmark Suite

```bash
# Sweeps batch sizes (64 to 512), benchmarks throughput, and plots scaling curves:
python benchmark.py
```

### 4. Train Individual Models Directly

```bash
# Run PyTorch with CNN on Apple M4 GPU (MPS)
python torch_train.py

# Run JAX with Flax CNN on XLA
python jax_train.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
