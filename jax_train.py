"""
JAX pure-functional implementation of an MLP classifier on MNIST/Fashion-MNIST.
Demonstrates:
- Pure parameter pytrees (dictionaries of immutable arrays)
- Functional autograd via jax.value_and_grad
- JIT compilation via @jax.jit
- Explicit PRNG keys with jax.random.split
- Pure functional Adam optimizer
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Parameter Initialization (Pure Functional)
def init_linear(in_features, out_features, key):
    k_w, k_b = jax.random.split(key)
    # He/Kaiming normal initialization
    std = jnp.sqrt(2.0 / in_features)
    w = jax.random.normal(k_w, (in_features, out_features)) * std
    b = jnp.zeros((out_features,))
    return {"w": w, "b": b}

def init_mlp(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for in_f, out_f, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        params.append(init_linear(in_f, out_f, k))
    return params

# 2. Forward Propagation
def forward(params, x):
    # Flatten input (B, 1, 28, 28) -> (B, 784)
    h = x.reshape((x.shape[0], -1))
    for layer in params[:-1]:
        h = jnp.maximum(0.0, jnp.dot(h, layer["w"]) + layer["b"])
    # Final linear layer (logits)
    logits = jnp.dot(h, params[-1]["w"]) + params[-1]["b"]
    return logits

# 3. Loss & Metric Functions
def cross_entropy_loss(params, x, y):
    logits = forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # One-hot cross entropy
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
    return loss

def compute_accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y) * 100.0

# 4. Pure Functional Adam Optimizer
def init_adam(params):
    # First and second moment accumulators
    m = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    v = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    return {"m": m, "v": v, "step": 0}

@jax.jit
def adam_step(params, opt_state, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    step = opt_state["step"] + 1
    
    # Update biased first moment: m = beta1 * m + (1 - beta1) * g
    new_m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    # Update biased second raw moment: v = beta2 * v + (1 - beta2) * (g^2)
    new_v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * (g ** 2), opt_state["v"], grads)
    
    # Compute bias-corrected moments and update parameters
    def update_param(p, m, v):
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        return p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        
    new_params = jax.tree.map(update_param, params, new_m, new_v)
    new_state = {"m": new_m, "v": new_v, "step": step}
    return new_params, new_state

# 5. Compiled Training Step
@jax.jit
def train_step(params, opt_state, x, y, lr=1e-3):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    new_params, new_opt_state = adam_step(params, opt_state, grads, lr=lr)
    return new_params, new_opt_state, loss

# 6. Data Loading
def get_dataloaders(batch_size=64, dataset_cls=datasets.FashionMNIST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate(params, dataloader):
    total_acc = 0.0
    total_loss = 0.0
    num_samples = 0
    for images, labels in dataloader:
        x = jnp.array(images.numpy())
        y = jnp.array(labels.numpy())
        batch_size = x.shape[0]
        loss = cross_entropy_loss(params, x, y)
        acc = compute_accuracy(params, x, y)
        total_loss += float(loss) * batch_size
        total_acc += float(acc) * batch_size
        num_samples += batch_size
    return total_loss / num_samples, total_acc / num_samples

def run_jax(epochs=5, batch_size=64, lr=1e-3):
    print(f"JAX Default Backend: {jax.default_backend()}")
    print(f"JAX Available Devices: {jax.devices()}")

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    # 784 -> 256 -> 256 -> 10
    layer_sizes = [784, 256, 256, 10]
    params = init_mlp(layer_sizes, init_key)
    opt_state = init_adam(params)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_times": []}

    print(f"\n--- Starting JAX Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        epoch_losses = []
        
        for images, labels in train_loader:
            x = jnp.array(images.numpy())
            y = jnp.array(labels.numpy())
            params, opt_state, loss = train_step(params, opt_state, x, y, lr=lr)
            # block_until_ready() ensures asynchronous GPU/device execution completes
            loss.block_until_ready()
            epoch_losses.append(float(loss))
            
        duration = time.perf_counter() - start_time
        avg_train_loss = float(np.mean(epoch_losses))
        test_loss, test_acc = evaluate(params, test_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s")

    return history

if __name__ == "__main__":
    run_jax(epochs=5)
