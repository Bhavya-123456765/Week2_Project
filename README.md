# Week 2 Task: Deep Learning Model for Handwritten Digit Recognition

## Overview
This project implements a deep learning model for recognizing handwritten digits (0-9) using the MNIST dataset and PyTorch.

## Model Architecture
- **Framework:** PyTorch
- **Model:** Multilayer Perceptron (MLP)
- **Input:** 784 neurons (28x28 pixels)
- **Hidden Layer:** 128 neurons (ReLU activation)
- **Output:** 10 neurons (digits 0-9)

## Performance
- **Test Accuracy:** 97.01%
- **Final Training Loss:** 0.0817
- **Total Parameters:** 101,770

## Hyperparameter Tuning
Tested different:
- Learning rates: 0.01, 0.001, 0.0001
- Epochs: 3, 5, 10
- Batch sizes: 32, 64, 128

**Best Configuration:**
- Learning Rate: 0.001
- Epochs: 5
- Batch Size: 64

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python Task2.py

# Run hyperparameter experiments
python experiments.py
