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
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.perf_counter()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if use_amp and device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    duration = time.perf_counter() - start_time
    epoch_loss = total_loss / total
    epoch_acc = (correct / total) * 100.0
    throughput = total / duration
    return epoch_loss, epoch_acc, duration, throughput

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
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
    train_loader, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=batch_size)
    
    in_channels = 3 if dataset_name.lower() == "cifar10" else 1
    spatial_size = 32 if dataset_name.lower() == "cifar10" else 28
    
    if model_type.lower() == "cnn":
        model = SimpleCNN(in_channels=in_channels, spatial_size=spatial_size, num_classes=10).to(device)
    else:
        in_features = in_channels * spatial_size * spatial_size
        model = SimpleMLP(in_features=in_features, num_classes=10).to(device)

    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile fallback to eager: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [], 
        "epoch_times": [], "throughputs": [], "peak_mem_mb": 0.0
    }

    print(f"--- Starting PyTorch Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        loss, acc, duration, throughput = train_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        history["throughputs"].append(throughput)
        
        mem_mb = 0.0
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            history["peak_mem_mb"] = max(history["peak_mem_mb"], mem_mb)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s ({throughput:.0f} img/s) | Mem: {mem_mb:.1f}MB")
    
    return history

if __name__ == "__main__":
    run_pytorch(model_type="cnn", dataset_name="FashionMNIST", epochs=3, batch_size=256, use_amp=True)
