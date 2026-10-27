# Architecture Decision Records (ADRs)

Engineering design rationale, architectural trade-offs, and empirical decisions made in the `fashion-mnist-torch-vs-jax` benchmark.

---

## ADR-001: Backend Topology — PyTorch Metal (MPS) vs. JAX CPU (XLA)

### Status
Accepted — 2026-10-20, Aditya Bhadoria

### Revisit
Revisit if an official Apple Silicon Metal plugin for JAX emerges or when MLX reaches mature parity with PyTorch MPS.

### Context
When benchmarking deep learning acceleration on Apple Silicon (macOS Darwin arm64), the hardware backend options for PyTorch and JAX are fundamentally asymmetric:
* **PyTorch** has first-class native support for Apple Silicon GPU acceleration through the `torch.backends.mps` (Metal Performance Shaders) pipeline, deeply maintained by Apple and PyTorch core.
* **JAX** historically provided an experimental `jax-metal` plugin developed by Apple. However, `jax-metal` (latest 0.1.1, released 2024-10-08 on PyPI) requires `jax>=0.4.34` and has not been updated for current JAX (0.12.x at the time of this benchmark). It was therefore not used.

### Decision
Benchmark PyTorch on the **Apple M4 GPU via MPS** against JAX on the **Apple M4 CPU using fused XLA compilation** (`@jax.jit` with NEON vectorization).

### Consequences
* This reflects the real-world operational choice for Apple Silicon engineers: Apple's specialized Metal GPU pipeline vs. Google's mature XLA whole-graph compiler running within low-latency CPU cache.
* It directly exposes the kernel dispatch latency penalty: at small batch sizes (≤ 64), JAX XLA CPU outperforms PyTorch GPU by 1.5x because CPU dispatch avoids Metal command buffer submission overhead.

---

## ADR-002: Workload & Dataset Selection — Fashion-MNIST over ImageNet/CIFAR

### Status
Accepted — 2026-10-20, Aditya Bhadoria

### Revisit
Revisit if evaluating modern high-resolution vision backbones (such as Vision Transformers or larger spatial inputs) where memory transfer bottlenecks reappear.

### Context
Deep learning performance benchmarks on local workstations are frequently confounded by external bottlenecks that have nothing to do with compute:
1. Disk read latency and filesystem caching.
2. Image decompression overhead (libjpeg-turbo / Pillow CPU bottlenecks).
3. Python multiprocessing dataloader worker IPC overhead.

If the dataloader cannot feed the GPU fast enough, GPU utilization drops and the benchmark measures Python I/O rather than raw hardware execution.

### Decision
Choose **Fashion-MNIST** (28×28 grayscale images, 60,000 samples, 10 classes) loaded entirely into unified memory.

### Consequences
* Eliminates disk throughput and image decoding jitter completely.
* The timed region contains only device-side training steps on preloaded tensors; the input pipeline is timed separately and reported.
* Allows seamless, reproducible benchmarking without multi-gigabyte dataset downloads.

---

## ADR-003: FP16 Mixed Precision on Sub-10MB Unified Memory Models

### Status
Accepted — 2026-10-20, Aditya Bhadoria

### Revisit
Revisit on models exceeding 50M parameters or when evaluating BF16 autocast support on next-generation M-series hardware.

### Context
On discrete data-centre GPUs with tensor cores (e.g. NVIDIA A100/H100), mixed precision commonly gives large speed-ups on matmul-heavy models because it halves memory traffic and uses dedicated FP16/BF16 units (vendor guidance; not measured here).

### Decision
Evaluate FP16 mixed precision on the M4 Mac mini using:
* PyTorch: `torch.autocast(device_type="mps", dtype=torch.float16)`
* JAX/Flax: compute dtype `jnp.float16` on every layer (`dtype=`); parameters remain float32 (Flax `param_dtype` default). This is mixed precision, not pure FP16.

### Empirical Findings & Trade-Off
* **Result:** PyTorch FP16 on MPS achieved **11,284 img/s** compared to **11,761 img/s** on FP32 (a 4% throughput degradation, 0.96x).
* **Root Cause:** our CNN's resident state (weights, gradients, Adam moments) is 6.4 MB and its peak step allocation is 299.4 MB at batch 256. On the Apple M4's 120 GB/s unified memory architecture, memory bandwidth is not saturated. `torch.autocast` inserts per-op dtype casts around each kernel; no `GradScaler` is used in this benchmark (loss scaling is not performed). On a 422k-parameter model the cast overhead is comparable to the arithmetic saved, so no net gain is expected; this is an observation, not a profiled root cause.
* **Engineering Rationale:** The FP16 result is reported as measured. FP16 on Apple Silicon provides massive leverage for memory-bound workloads (LLMs, large Vision Transformers), but should not be blindly enabled for lightweight convolutional models.

---

## ADR-004: Compiler Optimization Strategy — `torch.compile` vs. `@jax.jit`

### Status
Accepted — 2026-10-20, Aditya Bhadoria

### Revisit
Revisit when MPSInductor graduates from prototype status in PyTorch releases (such as PyTorch 2.8+) and stabilizes compiler warmup variance.

### Context
Historically, PyTorch on macOS was strictly interpreted and eager. PyTorch 2.x introduced `torch.compile` powered by TorchDynamo and Inductor. An Inductor backend that emits Metal shaders for MPS (`MPSInductor`) shipped as a prototype in PyTorch 2.7.0 (April 2025, [#143893](https://github.com/pytorch/pytorch/pull/143893); tracker [#150121](https://github.com/pytorch/pytorch/issues/150121)) and was proposed as a release feature for 2.8.0 ([#157957](https://github.com/pytorch/pytorch/issues/157957)). This benchmark uses PyTorch 2.14.1.

Meanwhile, JAX was designed from inception as a compiled functional framework where `@jax.jit` performs end-to-end XLA graph compilation and operator fusion.

### Decision
Evaluate:
1. First-epoch warmup penalty (compilation latency).
2. Steady-state throughput across epochs 2–3.

### Consequences
* In isolated multi-process benchmarks ($N=5$), `torch.compile` on Apple Silicon MPS exhibits significant launch variance (steady-state median 6,814 img/s, range 6,600–11,663 img/s) with an initial compilation warmup penalty of 1.05 s.
* Shows that `torch.compile` captures and accelerates this small CNN on MPS without graph breaks; larger models were not tested.
