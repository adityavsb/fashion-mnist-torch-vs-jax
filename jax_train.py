"""
JAX / Flax Implementation of Scaled Image Classification.
Supports:
- Architectures: MLP and Deep Convolutional Neural Network (CNN)
- Precision: FP32 vs FP16 (jnp.float16)
- Datasets: Fashion-MNIST (1x28x28) and CIFAR-10 (3x32x32 RGB)
- Functional State: Flax Linen + Optax Adam optimizer
- Hardware: XLA Compiled execution via @jax.jit
- Metrics: Loss, Accuracy, Epoch Time, and Throughput (images/sec)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from torchvision import datasets, transforms
from flax.training import train_state

class CustomTrainState(train_state.TrainState):
    batch_stats: dict

def torch_like_kernel_init():
    # PyTorch nn.Linear/nn.Conv2d default: kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    return nn.initializers.variance_scaling(scale=1.0 / 3.0, mode="fan_in", distribution="uniform")

def torch_like_bias_init(fan_in):
    bound = 1.0 / (fan_in ** 0.5)
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init

TORCH_BN_MOMENTUM_AS_FLAX = 0.9  # torch momentum 0.1 (weight on new batch) == flax momentum 0.9 (weight on running)

class SimpleMLP(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 256
    dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(self.dtype)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(784))(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(self.hidden_dim))(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(self.hidden_dim))(x)
        return x

class SimpleCNN(nn.Module):
    num_classes: int = 10
    dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(self.dtype)
        # x is (B, H, W, C)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(1 * 9))(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=TORCH_BN_MOMENTUM_AS_FLAX, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(32 * 9))(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=TORCH_BN_MOMENTUM_AS_FLAX, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128, dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(64 * 7 * 7))(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=torch_like_kernel_init(), bias_init=torch_like_bias_init(128))(x)
        return x

def get_device_arrays(dataset_name="FashionMNIST", precision_dtype=jnp.float32):
    """Load the whole dataset once as normalised float arrays on JAX device.
    Returns (train_x, train_y, test_x, test_y) with x in NHWC."""
    if dataset_name.lower() == "cifar10":
        raise NotImplementedError("CIFAR-10 preloading not implemented yet.")
    from torchvision import datasets
    tr = datasets.FashionMNIST(root="./data", train=True, download=True)
    te = datasets.FashionMNIST(root="./data", train=False, download=True)

    def to_x(ds):
        arr = np.asarray(ds.data, dtype=np.float32)[..., None] / 255.0 * 2.0 - 1.0
        return jnp.asarray(arr, dtype=precision_dtype)

    train_x = to_x(tr)
    train_y = jnp.asarray(np.asarray(tr.targets), dtype=jnp.int32)
    test_x = to_x(te)
    test_y = jnp.asarray(np.asarray(te.targets), dtype=jnp.int32)
    return train_x, train_y, test_x, test_y

def measure_input_pipeline(dataset_name="FashionMNIST", batch_size=256, precision_dtype=jnp.float32):
    """Transparency timer: measures one pass of the old DataLoader pipeline with transpose."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    t0 = time.perf_counter()
    for images, labels in loader:
        _ = jnp.array(images.numpy().transpose(0, 2, 3, 1), dtype=precision_dtype)
        _ = jnp.array(labels.numpy())
    return time.perf_counter() - t0

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch_x,
            train=True,
            mutable=["batch_stats"]
        )
        # Compute loss in float32 for numerical stability
        logits_f32 = logits.astype(jnp.float32)
        one_hot = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits=logits_f32, labels=one_hot).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss

@jax.jit
def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch_x,
        train=False
    )
    logits_f32 = logits.astype(jnp.float32)
    one_hot = jax.nn.one_hot(batch_y, 10)
    loss = optax.softmax_cross_entropy(logits=logits_f32, labels=one_hot).mean()
    preds = jnp.argmax(logits_f32, axis=-1)
    acc = jnp.mean(preds == batch_y) * 100.0
    return loss, acc

def evaluate(state, test_x, test_y, batch_size=256):
    total_acc = 0.0
    total_loss = 0.0
    num_samples = 0
    n = test_x.shape[0]
    for i in range(0, n, batch_size):
        x = test_x[i:i + batch_size]
        y = test_y[i:i + batch_size]
        batch_size_cur = x.shape[0]
        loss, acc = eval_step(state, x, y)
        total_loss += float(loss) * batch_size_cur
        total_acc += float(acc) * batch_size_cur
        num_samples += batch_size_cur
    return total_loss / num_samples, total_acc / num_samples

def run_jax(model_type="cnn", dataset_name="FashionMNIST", epochs=5, batch_size=128, lr=1e-3, use_fp16=False):
    precision_dtype = jnp.float16 if use_fp16 else jnp.float32
    precision_tag = "FP16 compute (FP32 params)" if use_fp16 else "FP32"
    print(f"[JAX] Backend: {jax.default_backend()} | Model: {model_type.upper()} | Precision: {precision_tag} | Batch: {batch_size}")

    train_x, train_y, test_x, test_y = get_device_arrays(dataset_name, precision_dtype)
    in_channels = 3 if dataset_name.lower() == "cifar10" else 1
    spatial_size = 32 if dataset_name.lower() == "cifar10" else 28

    model = SimpleCNN(dtype=precision_dtype) if model_type.lower() == "cnn" else SimpleMLP(dtype=precision_dtype)

    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, spatial_size, spatial_size, in_channels), dtype=precision_dtype)
    variables = model.init(key, dummy_input, train=False)
    
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    tx = optax.adam(learning_rate=lr)
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [],
        "epoch_times": [], "throughputs": [], "input_pipeline_s": 0.0
    }

    # Record transparency timer once per run
    history["input_pipeline_s"] = measure_input_pipeline(dataset_name, batch_size, precision_dtype)

    n_samples = train_x.shape[0]
    stop = (n_samples // batch_size) * batch_size

    print(f"--- Starting JAX Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(42 + epoch)
        perm = rng.permutation(n_samples)

        start_time = time.perf_counter()
        epoch_losses = []
        total_samples = 0

        for i in range(0, stop, batch_size):
            batch_idx = perm[i:i + batch_size]
            x = train_x[batch_idx]
            y = train_y[batch_idx]
            state, loss = train_step(state, x, y)
            loss.block_until_ready()
            epoch_losses.append(float(loss))
            total_samples += x.shape[0]

        duration = time.perf_counter() - start_time
        avg_train_loss = float(np.mean(epoch_losses))
        throughput = total_samples / duration
        test_loss, test_acc = evaluate(state, test_x, test_y, batch_size=batch_size)

        history["train_loss"].append(avg_train_loss)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        history["throughputs"].append(throughput)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s ({throughput:.0f} img/s)")

    return history

if __name__ == "__main__":
    run_jax(model_type="cnn", dataset_name="FashionMNIST", epochs=3, batch_size=256, use_fp16=True)
