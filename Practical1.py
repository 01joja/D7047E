import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


network = nn.Sequential(
    
    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3),

    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),

    nn.Linear(10500, 100),
    nn.Linear(100, 2)
)