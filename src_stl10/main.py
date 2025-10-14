import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch import nn
from torchsummary import summary

from src_stl10.data import get_dataloaders
from src_stl10.models import BOTTLENECK_LIST, WrapperNetwork, get_model
from src_stl10.train import train


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--freeze_features",
        action="store_true",
        help="Whether to freeze the feature extraction layers",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="Whether to load the best model or not.",
    )
    parser.add_argument(
        "--bottleneck_type",
        type=str,
        choices=BOTTLENECK_LIST,
        help="Type of bottleneck to use.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Whether to use Automatic Mixed Precision (AMP) for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for the optimizer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataloader, test_dataloader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    convnet = get_model(load_best=args.load_best)

    model = WrapperNetwork(convnet, args.bottleneck_type, args.freeze_features)

    summary(model.cpu(), (3, 96, 96), device="cpu")

    out = train(
        model,
        train_dataloader,
        test_dataloader,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        ),
        epochs=args.epochs,
        use_amp=args.use_amp,
    )


if __name__ == "__main__":
    main()
