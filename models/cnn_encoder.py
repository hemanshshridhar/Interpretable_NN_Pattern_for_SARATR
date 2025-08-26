import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

# Assuming `PolarimetricSARDataset` class is already defined as in the previous code
# Make sure to import complex_correlation and rotate functions here if needed
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleBranchCNN(nn.Module):
    def __init__(self):
        super(SingleBranchCNN, self).__init__()
        self.conv1 = nn.Conv2d(72, 1, kernel_size=1, padding=0)   # Input: (B, 72, 54, 54
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.squeeze()
        x = self.relu(self.conv1(x))     # → (B, 8, 36, 27, 27)
        
        x = x.view(x.size(0), -1)                # Flatten

        return x

class ParallelChannelCNN(nn.Module):
    def __init__(self, num_classes):
        super(ParallelChannelCNN, self).__init__()
        self.branch1 = SingleBranchCNN()
        self.branch2 = SingleBranchCNN()
        self.branch3 = SingleBranchCNN()
        self.branch4 = SingleBranchCNN()

        # After flatten: 16 * 2916 = 48672
        self.fc1 = nn.Linear(11664, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):

        x1 = self.branch1(x[:, 0:1, :, :, :])  # Shape: (B, 1, 72, 54, 54)
        x2 = self.branch2(x[:, 1:2, :, :, :])
        x3 = self.branch3(x[:, 2:3, :, :, :])
        x4 = self.branch4(x[:, 3:4, :, :, :])

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # Concatenate features
        x = self.fc1(x_cat)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x




# Set up the dataset and dataloader
root_dir = "/kaggle/input/split-vehicles/VEHICLES_REORG/train"
channel_dirs = ["HH_NPY", "HV_NPY", "VH_NPY", "VV_NPY"]

dataset = PolarimetricSARDataset(root_dir, channel_dirs, rotate=rotate, correlation=complex_correlation)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model initialization and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParallelChannelCNN(num_classes=7).to(device).float()


train_model(model, dataloader, num_epochs=5, lr=0.001)