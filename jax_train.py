"""
JAX / Flax Implementation of Scaled Image Classification.
Supports:
- Architectures: MLP and Deep Convolutional Neural Network (CNN)
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
from torch.utils.data import DataLoader

# 1. Architectures in Flax
class SimpleMLP(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class SimpleCNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x is (B, H, W, C)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

# 2. Data Loading
def get_dataloaders(dataset_name="FashionMNIST", batch_size=64):
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset_cls = datasets.CIFAR10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_cls = datasets.FashionMNIST

    train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 3. Compiled Training Step
@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch_x,
            train=True,
            mutable=["batch_stats"]
        )
        one_hot = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
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
    one_hot = jax.nn.one_hot(batch_y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y) * 100.0
    return loss, acc

def evaluate(state, dataloader):
    total_acc = 0.0
    total_loss = 0.0
    num_samples = 0
    for images, labels in dataloader:
        # Convert torch (B, C, H, W) -> JAX (B, H, W, C)
        x = jnp.array(images.numpy().transpose(0, 2, 3, 1))
        y = jnp.array(labels.numpy())
        batch_size = x.shape[0]
        loss, acc = eval_step(state, x, y)
        total_loss += float(loss) * batch_size
        total_acc += float(acc) * batch_size
        num_samples += batch_size
    return total_loss / num_samples, total_acc / num_samples

from flax.training import train_state

class CustomTrainState(train_state.TrainState):
    batch_stats: dict

def run_jax(model_type="cnn", dataset_name="FashionMNIST", epochs=5, batch_size=128, lr=1e-3):
    print(f"[JAX] Backend: {jax.default_backend()} | Model: {model_type.upper()} | Dataset: {dataset_name} | Batch: {batch_size}")

    train_loader, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=batch_size)
    in_channels = 3 if dataset_name.lower() == "cifar10" else 1
    spatial_size = 32 if dataset_name.lower() == "cifar10" else 28

    model = SimpleCNN() if model_type.lower() == "cnn" else SimpleMLP()

    # Initialize model variables with sample shape (1, H, W, C)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, spatial_size, spatial_size, in_channels))
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

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_times": [], "throughputs": []}

    print(f"--- Starting JAX Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        epoch_losses = []
        total_samples = 0
        
        for images, labels in train_loader:
            x = jnp.array(images.numpy().transpose(0, 2, 3, 1))
            y = jnp.array(labels.numpy())
            state, loss = train_step(state, x, y)
            loss.block_until_ready()
            epoch_losses.append(float(loss))
            total_samples += x.shape[0]
            
        duration = time.perf_counter() - start_time
        avg_train_loss = float(np.mean(epoch_losses))
        throughput = total_samples / duration
        test_loss, test_acc = evaluate(state, test_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        history["throughputs"].append(throughput)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s ({throughput:.0f} img/s)")

    return history

if __name__ == "__main__":
    run_jax(model_type="cnn", dataset_name="FashionMNIST", epochs=5, batch_size=128)
