import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

BOTTLENECK_LIST = [
    "avg_8x8",
    "max_8x8",
    "global_avg",
    "global_max",
    "2x2_conv",
    "3x3_conv",
    "5x5_conv",
    "linear",
]

BOTTLENECK_TYPE = Literal[
    "avg_8x8",
    "max_8x8",
    "global_avg",
    "global_max",
    "2x2_conv",
    "3x3_conv",
    "5x5_conv",
    "linear",
]


class ConvNet(nn.Module):

    def __init__(self, num_classes=100):
        """
        Initializes the convolutional neural network model.

        Args:
            num_classes : the number of classes
        """

        # YOUR CODE HERE
        # Initialize the PyTorch module
        super().__init__()

        # (Sequential) Feature extraction block (convolution + pooling)
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 6, 5
            ),  # Conv1: input RGB (3 channels), applies 6 filters → [3, 32, 32] reduced to [6, 28, 28]
            nn.Tanh(),  # Non-linear activation
            nn.AvgPool2d(
                2, 2
            ),  # Average Pooling 1: reduce size by half (stride S=2) → [6, 14, 14]
            nn.Conv2d(
                6, 16, 5
            ),  # Conv2: 6 input channels → applies 16 filters: [6, 14, 14] → [16, 10, 10]
            nn.Tanh(),  # Non-linear activation function
            nn.AvgPool2d(2, 2),  # Average Pooling 2: reduce size by half → [16, 5, 5]
            nn.Conv2d(
                16, 120, 5
            ),  # Conv3: 16 input channels → applies 120 filters: [16, 5, 5] → [120, 1, 1]
            nn.Tanh(),  # Non-linear activation function
        )

        # Classification block (fully connected (FC) layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten feature map, convert 4D tensors to 2D -> [batch, features=120]
            nn.Linear(120, 84),  # FC1: hidden layer with 84 neurons
            nn.Tanh(),  # Non-linear activation function
            nn.Linear(84, num_classes),  # FC2: output logits for each class (100)
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        # YOUR CODE HERE
        # Pass input through convolutional layers
        x = self.features(x)

        # Pass result through classifier to get class scores
        return self.classifier(x)


class WrapperNetwork(nn.Module):
    """Our ConvNet output dimensions increase as the dimensions of the input image do.
    Here we will add additional modules to make model training scalable and more efficient.

    ConvNet was initially trained on images with shape 32x32. We are now working with a
    subset 96x96.

    Let's take a deeper look at our initial architecture. We will be computing output sizes
    using the following formula:

    output_size = floor(input_size - kernel_size / stride) + 1

    We first apply a convolutional layer with a 5x5 kernel.
        - Then with 32x32 images the output size would be:
            output_size = floor(32 - 5 / 1) + 1 = 28
        - And for 96x96 images we have:
            output_size = floor(96 - 5 / 1) + 1 = 92

    Then we apply a 2d average pool with a 2x2 kernel and stride of 2, reducing output size
    to by half.
        - Then 32x32, output_size = 28 / 2 = 14
        - And for 96x96, output_size = 92 / 2 = 46

    We repeat this two times, let's summarize remaining computations below:
        - For 32x32:
            2nd conv, output_size = floor(14 - 5 / 1) + 1 = 10
            2nd avg pool, output_size = 10 / 2 = 5
            3rd conv, output_size = floor(5 - 5 / 1) + 1 = 1
        - For 96x96:
            2nd conv, output_size = floor(46 - 5 / 1) + 1 = 42
            2nd avg pool, output_size = 42 / 2 = 21
            3rd conv, output_size = floor(21 - 5 / 1) + 1 = 17

    We have 120 channels at the end. This means that for an input of 32x32 we have an output
    volume of 1x1x120 = 120, and for 17x17x120=34680.

    We need to find a way to reduce the output size for our new images. If we used fully
    connected layers as before, the number of parameters for the weights would be:
    30720 x output_size, which inevitably would increase the parameters to learn, and would
    make training more difficult.

    We will consider 5 cases here:
        1. Perform max pooling and average pooling to reduce the output volume
            1.1. Max Pool 4x4 with 4 stride 2x2x120=480
            1.2. Avg Pool 4x4 with 4 stride 2x2x120=480
            1.3. Max Pool over the whole image (global max pool) 1x1x120=120
            1.4. Avg Pool over the whole image (global avg pool) 1x1x120=120
        2. Perform 2x2 convolution with stride 2 to reduce the output channels to 4 channels
            This would reduce the output volume from
            16x16x120 to 8x8x4=256.
        3. Add a new convolutional layer with:
            3.1. kernel size 3x3 and stride of 5 and 40 output channels. Then the
                output_size = floor(17 - 3 / 5) + 1 = 3, resulting in an
                output volume of 3x3x40=360
            3.2. kernel_size 5x5 and stride of 5 and 12 output channels. Then the
                output_size = floor(17 - 5 / 5) + 1 = 3, resulting in an
                output volume of 3x3x12=108
        4. Naively placing a classifier layer on top of the output volume.

    For each case we will use a 2-layer MLP which will grab the final output volume,
    and reduce it to a 120 feature vector, followed by ReLU as non-linearity, and following
    with a final classification layer to number of classes 5. We are not doing this for
    our naive approach since the number of parameters would inflate too much, making training
    unsustainably slow.

    To summarize the number of additional parameters that we will add a 120x5 + 5 =605 additional
    parameters to every architecture. Let's compute the additional weights required for each
    case:
        1.1. 480x120=57600 + 120 + 605 = 58325
        1.2. 480x120=57600 + 120 + 605 = 58325
        1.3. 120x120=14400 + 120 + 605 = 15125
        1.4. 120x120=14400 + 120 + 605 = 15125
        2. (Additional Conv Parameters) 2x2x120x4=1920 + 4 + (256x120=30720 + 120) + 605 = 33369
        3.1. (Additional Conv Parameters) 3x3x120x40=43200 + 40 + (360x120=43200 + 120) + 605 = 87165
        3.2. (Additional Conv Parameters) 5x5x120x12=36000 + 12 + (108x120=12960+ 120) + 605 = 49607
        4. 17x17x120 = 34680x5 + 5 = 173405
    """

    def __init__(
        self,
        convnet: ConvNet,
        bottleneck_type: BOTTLENECK_TYPE,
        freeze_features: bool = True,
    ):
        super().__init__()
        self.convnet = convnet
        self.freeze_features = freeze_features

        if self.freeze_features:
            # Freeze features if specified
            for param in self.convnet.features.parameters():
                param.requires_grad = False

        # Trash previous classifier
        self.convnet.classifier = nn.Identity()

        # Instantiate bottleneck and classifier
        if bottleneck_type == "avg_8x8":
            self.intermediate_features = nn.AvgPool2d(8, 8)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(480, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "max_8x8":
            self.intermediate_features = nn.MaxPool2d(8, 8)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(480, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "global_avg":
            self.intermediate_features = nn.AvgPool2d(17, 1)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(120, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "global_max":
            self.intermediate_features = nn.MaxPool2d(17, 1)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(120, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "2x2_conv":
            self.intermediate_features = nn.Conv2d(120, 4, 2, 2)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "3x3_conv":
            self.intermediate_features = nn.Conv2d(120, 40, 3, 5)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(360, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "5x5_conv":
            self.intermediate_features = nn.Conv2d(120, 12, 5, 5)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(108, 120), nn.ReLU(), nn.Linear(120, 5)
            )
        elif bottleneck_type == "linear":
            self.intermediate_features = nn.Flatten()
            self.classifier = nn.Linear(34680, 5)
        else:
            raise ValueError(
                f"No implementation for {bottleneck_type}, valid options are {BOTTLENECK_TYPE}"
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)
        x = self.intermediate_features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnet(x)
        x = self.intermediate_features(x)
        return self.classifier(x)


def _checkpoints_dir() -> Path:
    """
    Return the checkpoints directory located at the repository package root.
    Uses Path(__file__).parents[1] / "checkpoints" as requested.
    """
    return Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint_for(model_name: str) -> Path | None:
    """
    Find the latest checkpoint file for a given model_name.
    Searches for patterns *_ckpt_*.pt and *_weights_*.pth and returns the newest Path or None.
    If look_in_long is True and a 'long' subdirectory exists it will prefer that directory.
    """
    base = _checkpoints_dir()

    patt_ckpt = f"*{model_name.lower()}*_ckpt_*.pt"
    patt_wts = f"*{model_name.lower()}*_weights_*.pth"

    candidates = []
    if base.exists():
        candidates.extend(sorted(base.glob(patt_ckpt)))
        candidates.extend(sorted(base.glob(patt_wts)))

    return candidates[-1] if candidates else None


def load_best_checkpoint():
    """
    Load the best/latest checkpoint for `model_name` into the provided `model` instance.

    Returns a tuple (model, path) where path is the Path that was loaded (or None if not found).
    """

    path = _latest_checkpoint_for("convnet")

    if path is None:
        raise RuntimeError(f"Failed to load best checkpoint from `{path}`")

    model = ConvNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded best checkpoint from `{path}`")

    return model, path


def get_model(
    load_best: bool = True,
):
    """
    Create a model instance by name and optionally load the best checkpoint.

    Args:
        model_name: 'convnet' or 'twolayer' (case-insensitive).
        freeze_features: if True, freeze feature extractor params when available.
        num_classes: number of output classes for the final linear layer.
        input_size: flattened input size for TwoLayerNet; if None, will default to 3*96*96.
        load_best: if True, attempt to load latest checkpoint for the chosen model.
        device: device to map loaded weights to (defaults to cuda if available else cpu).

    Returns:
        torch.nn.Module: instantiated model (weights loaded if found).
    """

    if load_best:
        model, _ = load_best_checkpoint()
        return model

    # default to ConvNet
    return ConvNet(num_classes=5)
