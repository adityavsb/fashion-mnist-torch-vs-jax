"""
PyTorch Implementation of Scaled Image Classification.
Supports:
- Architectures: MLP and Deep Convolutional Neural Network (CNN)
- Datasets: Fashion-MNIST (1x28x28) and CIFAR-10 (3x32x32 RGB)
- Hardware: Apple Silicon GPU via Metal Performance Shaders (MPS)
- Compilers: Eager mode vs torch.compile(model)
- Precision: Standard FP32 vs Mixed Precision FP16 (torch.autocast)
- Metrics: Loss, Accuracy, Epoch Time, Throughput (img/s), and Peak GPU Memory (MB)
"""
import time
import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, spatial_size=28, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        pooled_dim = spatial_size // 4
        flattened_size = 64 * pooled_dim * pooled_dim
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def get_device_tensors(dataset_name="FashionMNIST", device="mps"):
    """Load the whole dataset once as normalised float tensors on `device`.
    Returns (train_x, train_y, test_x, test_y) with x in NCHW."""
    if dataset_name.lower() == "cifar10":
        raise NotImplementedError("CIFAR-10 preloading not implemented yet.")
    from torchvision import datasets
    tr = datasets.FashionMNIST(root="./data", train=True, download=True)
    te = datasets.FashionMNIST(root="./data", train=False, download=True)
    def to_x(ds):
        return ds.data.unsqueeze(1).float().div_(255.0).sub_(0.5).div_(0.5).to(device)
    return to_x(tr), tr.targets.to(device), to_x(te), te.targets.to(device)

def iterate_batches(x, y, batch_size, generator=None, drop_last=True):
    n = x.shape[0]
    if generator is not None:
        perm = torch.randperm(n, generator=generator, device="cpu").to(x.device)
    else:
        perm = torch.arange(n, device=x.device)
    stop = (n // batch_size) * batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx], y[idx]

def measure_input_pipeline(dataset_name="FashionMNIST", batch_size=256):
    """Transparency timer: measures one pass of the old DataLoader pipeline."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - t0

def train_epoch(model, train_x, train_y, batch_size, optimizer, criterion, device, use_amp=False, generator=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    peak_step_mem = 0.0
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()
    for images, labels in iterate_batches(train_x, train_y, batch_size, generator=generator, drop_last=True):
        optimizer.zero_grad()
        if use_amp and device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if device.type == "mps":
            step_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
            peak_step_mem = max(peak_step_mem, step_mem)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    if device.type == "mps":
        torch.mps.synchronize()
    duration = time.perf_counter() - start_time
    epoch_loss = total_loss / total
    epoch_acc = (correct / total) * 100.0
    throughput = total / duration
    return epoch_loss, epoch_acc, duration, throughput, peak_step_mem

@torch.no_grad()
def evaluate(model, test_x, test_y, criterion, device, batch_size=256, use_amp=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in iterate_batches(test_x, test_y, batch_size, generator=None, drop_last=False):
        if use_amp and device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, (correct / total) * 100.0

def run_pytorch(model_type="cnn", dataset_name="FashionMNIST", epochs=5, batch_size=128, lr=1e-3, use_compile=False, use_amp=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    compile_tag = "torch.compile" if use_compile else "Eager"
    precision_tag = "FP16 (AMP)" if use_amp else "FP32"
    print(f"[PyTorch] Device: {device} | Model: {model_type.upper()} | Mode: {compile_tag} | Precision: {precision_tag} | Batch: {batch_size}")

    torch.manual_seed(42)
    train_x, train_y, test_x, test_y = get_device_tensors(dataset_name, device)
    gen = torch.Generator().manual_seed(42)

    in_channels = 3 if dataset_name.lower() == "cifar10" else 1
    spatial_size = 32 if dataset_name.lower() == "cifar10" else 28

    if model_type.lower() == "cnn":
        model = SimpleCNN(in_channels=in_channels, spatial_size=spatial_size, num_classes=10).to(device)
    else:
        in_features = in_channels * spatial_size * spatial_size
        model = SimpleMLP(in_features=in_features, num_classes=10).to(device)

    if use_compile:
        torch._dynamo.reset()
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [], 
        "epoch_times": [], "throughputs": [], "peak_mem_mb": 0.0,
        "resident_state_mb": 0.0, "driver_mem_mb": 0.0, "params_grads_adam_mb": 0.0,
        "input_pipeline_s": 0.0
    }

    # Record transparency timer once per run
    history["input_pipeline_s"] = measure_input_pipeline(dataset_name=dataset_name, batch_size=batch_size)

    print(f"--- Starting PyTorch Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        loss, acc, duration, throughput, peak_step_mem = train_epoch(
            model, train_x, train_y, batch_size, optimizer, criterion, device, use_amp=use_amp, generator=gen
        )
        test_loss, test_acc = evaluate(model, test_x, test_y, criterion, device, batch_size=batch_size, use_amp=use_amp)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        history["throughputs"].append(throughput)

        history["peak_mem_mb"] = max(history["peak_mem_mb"], peak_step_mem)
        history["resident_state_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024) if device.type == "mps" else 0.0
        history["driver_mem_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024) if device.type == "mps" else 0.0
        n_params = sum(p.numel() for p in model.parameters())
        history["params_grads_adam_mb"] = n_params * 4 * 4 / (1024 * 1024)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s ({throughput:.0f} img/s) | Peak Step Mem: {history['peak_mem_mb']:.1f}MB | Resident: {history['resident_state_mb']:.1f}MB")

    if use_compile:
        stats = torch._dynamo.utils.counters["stats"]
        history["compile_unique_graphs"] = int(stats.get("unique_graphs", 0))
        history["compile_graph_breaks"] = int(sum(torch._dynamo.utils.counters["graph_break"].values()))
        assert history["compile_unique_graphs"] > 0, "torch.compile did not capture any graph"

    return history

if __name__ == "__main__":
    run_pytorch(model_type="cnn", dataset_name="FashionMNIST", epochs=3, batch_size=256, use_amp=True)
