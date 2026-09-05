"""
Week 2 Task: Hyperparameter Tuning Experiments
Testing different epochs and learning rates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

print("=" * 60)
print("WEEK 2 TASK - HYPERPARAMETER TUNING EXPERIMENTS")
print("=" * 60)

# ============================================
# SETUP (Same as main code)
# ============================================

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("\n📥 Loading MNIST dataset...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# MODEL DEFINITION
# ============================================

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

def evaluate_model(model):
    """Helper function to evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def train_model(epochs, learning_rate):
    """Train model and return results"""
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
    
    accuracy = evaluate_model(model)
    training_time = time.time() - start_time
    
    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'accuracy': accuracy,
        'final_loss': losses[-1],
        'time': training_time,
        'losses': losses
    }

# ============================================
# EXPERIMENT 1: Test Different Epochs
# ============================================

print("\n" + "=" * 60)
print("📊 EXPERIMENT 1: Testing Different Epochs")
print("=" * 60)

epoch_counts = [3, 5, 10]
epoch_results = []

for epochs in epoch_counts:
    print(f"\n🔬 Training with {epochs} epochs...")
    result = train_model(epochs, 0.001)
    epoch_results.append(result)
    print(f"   ✅ Test Accuracy: {result['accuracy']:.2f}%")
    print(f"   ⏱️  Time: {result['time']:.2f} seconds")

# ============================================
# EXPERIMENT 2: Test Different Learning Rates
# ============================================

print("\n" + "=" * 60)
print("📊 EXPERIMENT 2: Testing Different Learning Rates")
print("=" * 60)

learning_rates = [0.01, 0.001, 0.0001]
lr_results = []

for lr in learning_rates:
    print(f"\n🔬 Training with learning rate: {lr}")
    result = train_model(5, lr)
    lr_results.append(result)
    print(f"   ✅ Test Accuracy: {result['accuracy']:.2f}%")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("📈 SUMMARY OF RESULTS")
print("=" * 60)

print("\n🏆 Epoch Comparison:")
print("-" * 50)
print(f"{'Epochs':<10} {'Accuracy':<12} {'Final Loss':<12} {'Time (s)':<10}")
print("-" * 50)
for r in epoch_results:
    print(f"{r['epochs']:<10} {r['accuracy']:<12.2f} {r['final_loss']:<12.4f} {r['time']:<10.2f}")

print("\n🏆 Learning Rate Comparison:")
print("-" * 40)
print(f"{'Learning Rate':<15} {'Accuracy':<12}")
print("-" * 40)
for r in lr_results:
    print(f"{r['lr']:<15} {r['accuracy']:<12.2f}")

# Best results
print("\n" + "=" * 60)
print("⭐ BEST RESULTS")
print("=" * 60)

best_epoch = max(epoch_results, key=lambda x: x['accuracy'])
best_lr = max(lr_results, key=lambda x: x['accuracy'])

print(f"\nBest Epoch Count: {best_epoch['epochs']} → Accuracy: {best_epoch['accuracy']:.2f}%")
print(f"Best Learning Rate: {best_lr['learning_rate']} → Accuracy: {best_lr['accuracy']:.2f}%")

print("\n" + "=" * 60)
print("✅ EXPERIMENTS COMPLETED!")
print("=" * 60)