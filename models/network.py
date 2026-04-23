# Yuyao Xu
# Apr 2026
# Defines CNN baseline, ResNet18 transfer learning model and DenseNet121 for art style classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Custom CNN baseline for art style classification
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNBaseline, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Fully connected layers (input: 128 * 28 * 28 after 3x pooling of 224x224)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    # Forward pass through the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32 x 112 x 112
        x = self.pool(F.relu(self.conv2(x)))   # -> 64 x 56 x 56
        x = self.pool(F.relu(self.conv3(x)))   # -> 128 x 28 x 28
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Build a ResNet18 model with a custom classification head
def build_resnet18(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        # Strategy a: freeze all layers except the final head
        for param in model.parameters():
            param.requires_grad = False
    # Replace the final fully connected layer with a new classification head
    # matching the number of target style classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# Print a summary of trainable parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")


# Build a DenseNet121 model with a custom classification head
def build_densenet121(num_classes, freeze_backbone=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Replace the classifier layer
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model