"""
PyTorch implementation of an MLP classifier on MNIST/Fashion-MNIST.
Demonstrates:
- Object-oriented architecture (torch.nn.Module)
- Stateful training loop (loss.backward(), optimizer.step())
- Hardware acceleration via Apple Silicon MPS (Metal Performance Shaders)
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
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Flatten image (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        return self.net(x)

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

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.perf_counter()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
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
    return epoch_loss, epoch_acc, duration

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, (correct / total) * 100.0

def run_pytorch(epochs=5, batch_size=64, lr=1e-3):
    # Device selection: Apple Silicon GPU (MPS) or CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU via MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    torch.manual_seed(42)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_times": []}

    print(f"\n--- Starting PyTorch Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        loss, acc, duration = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(duration)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {duration:.2f}s")
    
    return history

if __name__ == "__main__":
    run_pytorch(epochs=5)
