# Rosetta ML: PyTorch vs JAX Side-by-Side

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(MPS%20Ready)-EE4C2C.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-XLA%20Compiled-008080.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A side-by-side comparative Rosetta Stone contrasting **PyTorch** (imperative, object-oriented, stateful autograd) and **JAX** (pure functional, immutable pytrees, composable compiler transforms).

Each project implements identical architectures, datasets, and hyperparameters across both frameworks to provide concrete insights into **coding patterns**, **hardware execution**, and **runtime performance**.

---

## Benchmark Highlights: Project 1 (Fashion-MNIST)

Trained an identical 3-layer MLP ($784 \rightarrow 256 \rightarrow 256 \rightarrow 10$) with Adam optimizer ($lr=10^{-3}$) for 5 epochs on an **Apple M4 Mac**:

| Metric | PyTorch (`mps` GPU) | JAX (`@jax.jit` CPU/XLA) | Advantage |
| :--- | :--- | :--- | :--- |
| **Final Train Loss** | `0.2828` | `0.2819` | Parity matched ($\Delta = 0.0009$) |
| **Final Test Accuracy** | `87.72%` | `88.02%` | Parity matched |
| **Average Time / Epoch** | `3.58s` | `1.69s` | **JAX is 2.11x faster** ⚡ |

![Benchmark Comparison](assets/benchmark_results.png)

> **Key Systems Insight:** For small-to-medium networks, GPU kernel dispatch overhead over Metal command buffers can exceed computation time. JAX's `@jax.jit` fuses forward, backward, and optimizer updates into a single machine-code kernel that runs entirely inside the chip's cache and vector registers.

---

## Core Paradigm Comparison

### PyTorch (Stateful Object-Oriented)
```python
# Model owns mutable state
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Training step mutates parameters in-place
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### JAX (Pure Functional Transforms)
```python
# Model is a pure function over an immutable parameter pytree
def forward(params, x):
    h = x.reshape((x.shape[0], -1))
    return jnp.dot(h, params["w"]) + params["b"]

# Training step is a compiled mathematical function returning new state
@jax.jit
def train_step(params, opt_state, x, y, lr=1e-3):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    new_params, new_opt_state = adam_step(params, opt_state, grads, lr=lr)
    return new_params, new_opt_state, loss
```

---

## Projects Roadmap

- [x] **Project 1: Rosetta Stone Classifier** (Fashion-MNIST / MNIST)
  - OOP `nn.Module` vs. Functional PyTrees
  - Stateful `loss.backward()` vs. `jax.value_and_grad`
  - Global `torch.manual_seed` vs. explicit PRNG keys (`jax.random.split`)
  - Runtime speed & compilation benchmarks
- [ ] **Project 2: Physics-Informed Neural Network (PINN)**
  - Solving a damped harmonic oscillator ($m\ddot{x} + c\dot{x} + kx = 0$)
  - Higher-order derivatives: `jax.grad(jax.grad(f))` vs. `torch.autograd.grad(create_graph=True)`
- [ ] **Project 3: Attention Mechanism & Mini-Transformer from Scratch**
  - Scaled dot-product attention
  - Vectorization across heads and sequences via `jax.vmap` vs. `torch.vmap`
- [ ] **Project 4: Vectorized Reinforcement Learning**
  - Multi-armed bandits and tabular Q-learning
  - Simulating 10,000 parallel environments on hardware using `jax.lax.scan`
- [ ] **Project 5: Expectation-Maximization for GMM**
  - Pure tensor mathematics and Log-Sum-Exp numerical stability

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

### 3. Run Benchmark (PyTorch vs JAX)

```bash
# Run both implementations side-by-side and generate comparison plots
python benchmark.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
