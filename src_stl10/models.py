import os

import torch
import torch.nn as nn


class ConvNetNew(nn.Module):
    def __init__(self, num_classes=100, p_drop=0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: [3,32,32] -> [6,28,28] -> pool -> [6,14,14]
            nn.Conv2d(3, 6, 5),  # no padding (original)
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            # Block 2: [6,14,14] -> [16,10,10] -> pool -> [16,5,5]
            nn.Conv2d(6, 16, 5),  # no padding (original)
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            # 2 new layers (convolutional with padding, + BN/Dropout)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # (no pool here; stay at 5x5)
            # Final projection to 1x1 (expect 32 now)
            nn.Conv2d(32, 120, 5),  # [32,5,5] -> [120,1,1]
            nn.Tanh(),
        )

        # Classifier block remains unchanged
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(freeze_features=False):

    weight_path = "./cifar_pretrained_path.pth"

    model = ConvNetNew(p_drop=0.3)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))

    # Freeze Conv Layers
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Create two bottleneck layers
    bottleneck_1 = 34680 // 4
    bottleneck_2 = bottleneck_1 // 4

    # Modify classifier to match 5 classes
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(34680, bottleneck_1),
        nn.ReLU(),
        nn.Linear(bottleneck_1, bottleneck_2),
        nn.ReLU(),
        nn.Linear(bottleneck_2, 5),
    )

    return model
