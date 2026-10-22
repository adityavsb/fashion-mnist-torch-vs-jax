# Architecture Decision Records (ADRs)

Engineering design rationale, architectural trade-offs, and empirical decisions made in the `fashion-mnist-torch-vs-jax` benchmark.

---

## ADR-001: Backend Topology — PyTorch Metal (MPS) vs. JAX CPU (XLA)

### Status
Accepted

### Context
When benchmarking deep learning acceleration on Apple Silicon (macOS Darwin arm64), the hardware backend options for PyTorch and JAX are fundamentally asymmetric:
* **PyTorch** has first-class native support for Apple Silicon GPU acceleration through the `torch.backends.mps` (Metal Performance Shaders) pipeline, deeply maintained by Apple and PyTorch core.
* **JAX** historically provided an experimental `jax-metal` plugin developed by Apple. However, `jax-metal` is pinned to legacy JAX releases, lacks maintenance for modern JAX versions ($\ge 0.12$), and suffers from Metal command buffer allocation panics on newer Apple Silicon chips (M3/M4).

### Decision
We benchmark PyTorch on the **Apple M4 GPU via MPS** against JAX on the **Apple M4 CPU using fused XLA compilation** (`@jax.jit` with NEON vectorization).

### Consequences
* This reflects the real-world operational choice for Apple Silicon engineers: Apple's specialized Metal GPU pipeline vs. Google's mature XLA whole-graph compiler running within low-latency CPU cache.
* It directly exposes the kernel dispatch latency penalty: at small batch sizes ($\le 64$), JAX XLA CPU outperforms PyTorch GPU by $1.47\times$ because CPU dispatch avoids Metal command buffer submission overhead.

---

## ADR-002: Workload & Dataset Selection — Fashion-MNIST over ImageNet/CIFAR

### Status
Accepted

### Context
Deep learning performance benchmarks on local workstations are frequently confounded by external bottlenecks that have nothing to do with compute:
1. Disk read latency and filesystem caching.
2. Image decompression overhead (libjpeg-turbo / Pillow CPU bottlenecks).
3. Python multiprocessing dataloader worker IPC overhead.

If the dataloader cannot feed the GPU fast enough, GPU utilization drops and the benchmark measures Python I/O rather than raw hardware execution.

### Decision
We chose **Fashion-MNIST** ($28 \times 28$ grayscale images, 60,000 samples, 10 classes) loaded entirely into unified memory.

### Consequences
* Eliminates disk throughput and image decoding jitter completely.
* Ensures $100\%$ of measured epoch time reflects raw tensor arithmetic, kernel dispatch latency, memory bandwidth, and compiler execution efficiency.
* Allows seamless, reproducible benchmarking without multi-gigabyte dataset downloads.

---

## ADR-003: FP16 Mixed Precision on Sub-10MB Unified Memory Models

### Status
Accepted

### Context
On discrete data-center GPUs (e.g., NVIDIA A100/H100), Automatic Mixed Precision (AMP / FP16) almost universally yields $1.5\times - 2.5\times$ throughput speedups because VRAM bandwidth is the primary bottleneck and Tensor Cores feature dedicated FP16/BF16 matrix units.

### Decision
We implemented and tested FP16 mixed precision on the M4 Mac mini using:
* PyTorch: `torch.autocast(device_type="mps", dtype=torch.float16)`
* JAX: Explicit casting via `jnp.float16` and half-precision layer definitions.

### Empirical Findings & Trade-Off
* **Result:** PyTorch FP16 on MPS achieved **$11,284\text{ img/s}$** compared to **$11,761\text{ img/s}$** on FP32—a **$4\%$ throughput degradation** ($0.96\times$).
* **Root Cause:** The Deep CNN has a tiny footprint ($6.4\text{ MB}$ total parameter and activation memory). On the Apple M4's $120+\text{ GB/s}$ unified memory architecture, memory bandwidth is not saturated. The runtime overhead of dynamic type casting and loss scaling in `torch.autocast` slightly exceeds raw ALU savings on small architectures.
* **Engineering Rationale:** We candidly document this negative result rather than suppressing it. FP16 on Apple Silicon provides massive leverage for memory-bound workloads (LLMs, large Vision Transformers), but should not be blindly enabled for lightweight convolutional models.

---

## ADR-004: Compiler Optimization Strategy — `torch.compile` vs. `@jax.jit`

### Status
Accepted

### Context
Historically, PyTorch on macOS was strictly interpreted and eager. PyTorch 2.x introduced `torch.compile` powered by TorchDynamo and Inductor. Support for compiling graphs targeting the Apple Silicon MPS backend has only recently stabilized in PyTorch 2.14+.

Meanwhile, JAX was designed from inception as a compiled functional framework where `@jax.jit` performs end-to-end XLA graph compilation and operator fusion.

### Decision
We evaluated:
1. First-epoch warmup penalty (compilation latency).
2. Steady-state throughput across epochs 2–3.

### Consequences
* `torch.compile` on Apple Silicon MPS delivers a genuine **$+12\%$ steady-state throughput gain** ($13,167\text{ img/s}$ vs. $11,761\text{ img/s}$) with negligible initial compilation penalty ($5.86\text{s}$ vs. $5.87\text{s}$ for the first epoch).
* Shows that graph capture and kernel fusion on Apple Silicon are now viable for production PyTorch codebases.
