"""
Week 2 Task: Deep Learning Model Implementation and Training

Objective: Implement and train a deep learning model for image classification.
Framework: PyTorch
Dataset: MNIST
Model: Multilayer Perceptron (MLP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. DEVICE CONFIGURATION
# ============================================
print("=" * 50)
print("1. DEVICE CONFIGURATION")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================
# 2. DATA PREPROCESSING AND LOADING
# ============================================
print("=" * 50)
print("2. DATA PREPROCESSING AND LOADING")
print("=" * 50)

# Define transforms: Convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to tensor (values 0-1)
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5, std 0.5
])

# Download and load MNIST dataset
print("Downloading MNIST dataset...")
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

# Create data loaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}\n")

# ============================================
# 3. MODEL ARCHITECTURE DEFINITION
# ============================================
print("=" * 50)
print("3. MODEL ARCHITECTURE DEFINITION")
print("=" * 50)

class NeuralNetwork(nn.Module):
    """Simple Multilayer Perceptron (MLP) for image classification."""
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # Flatten 28x28 images to 784 vector
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),  # Input layer: 784 -> 128 neurons
            nn.ReLU(),              # ReLU activation
            nn.Linear(128, 10)      # Output layer: 128 -> 10 classes (digits 0-9)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Instantiate the model and move to device (CPU/GPU)
model = NeuralNetwork().to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")

# ============================================
# 4. TRAINING SETUP
# ============================================
print("=" * 50)
print("4. TRAINING SETUP")
print("=" * 50)

# Loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss Function: {loss_fn}")
print(f"Optimizer: {optimizer}")
print(f"Learning Rate: 0.001\n")

# ============================================
# 5. TRAINING LOOP
# ============================================
print("=" * 50)
print("5. TRAINING LOOP")
print("=" * 50)

epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device (CPU/GPU)
        data, target = data.to(device), target.to(device)
        
        # Forward pass: Compute predictions
        output = model(data)
        
        # Compute loss
        loss = loss_fn(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        running_loss += loss.item()
    
    # Calculate average loss for this epoch
    avg_epoch_loss = running_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}')

print("\nTraining complete!\n")

# ============================================
# 6. VISUALIZE TRAINING PROGRESS
# ============================================
print("=" * 50)
print("6. VISUALIZE TRAINING PROGRESS")
print("=" * 50)

# Plot training loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(train_losses, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(train_losses)), [f'Epoch {i+1}' for i in range(len(train_losses))])
plt.show()

print(f"Final Training Loss: {train_losses[-1]:.4f}\n")

# ============================================
# 7. MODEL EVALUATION
# ============================================
print("=" * 50)
print("7. MODEL EVALUATION")
print("=" * 50)

model.eval()  # Set model to evaluation mode
correct = 0
total = 0

# Disable gradient calculation for evaluation (faster, less memory)
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Correct predictions: {correct}/{total}\n')

# ============================================
# 8. SAVE THE TRAINED MODEL
# ============================================
print("=" * 50)
print("8. SAVE THE TRAINED MODEL")
print("=" * 50)

torch.save(model.state_dict(), 'mnist_mlp.pth')
print("Model saved as 'mnist_mlp.pth'")

# ============================================
# 9. PERFORMANCE SUMMARY
# ============================================
print("\n" + "=" * 50)
print("9. PERFORMANCE SUMMARY")
print("=" * 50)

print("MODEL ARCHITECTURE:")
print("  - Input: 784 (28x28 pixels, flattened)")
print("  - Hidden Layer: 128 neurons (ReLU activation)")
print("  - Output Layer: 10 neurons (digits 0-9)")
print("\nTRAINING PARAMETERS:")
print("  - Optimizer: Adam (learning rate = 0.001)")
print("  - Loss Function: CrossEntropyLoss")
print("  - Batch Size: 64")
print("  - Epochs: 5")
print("\nPERFORMANCE:")
print(f"  - Final Training Loss: {train_losses[-1]:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.2f}%")
