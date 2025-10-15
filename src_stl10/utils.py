import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def save_summary(
    args: argparse.Namespace, train_losses: list, train_accs: list, test_accs: list
):

    # Perform sanity checks
    assert len(train_losses) == len(train_accs) == len(test_accs) == args.epochs

    # Create pandas df with hp summary and final metrics
    df = pd.DataFrame(
        {
            "bottleneck": [args.bottleneck_type],
            "load_best": [args.load_best],
            "freeze_features": [args.freeze_features],
            "lr": [args.lr],
            "weight_decay": [args.weight_decay],
            "epochs": [args.epochs],
            "final_train_loss": [train_losses[-1]],
            "final_train_acc": [train_accs[-1]],
            "final_test_acc": [test_accs[-1]],
        }
    )

    summary_dir = Path(__file__).parents[1] / "runs" / "STL10" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(summary_dir / "summary.csv"):
        prev_data = pd.read_csv(summary_dir / "summary.csv")
        df = pd.concat([prev_data, df], ignore_index=True)

    df.to_csv(summary_dir / "summary.csv", index=False)


def save_stats(log_dir: Path, train_losses: list, train_accs: list, test_accs: list):
    stats_path = log_dir / "artifacts" / "stats"
    stats_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats_path / "train_stats.npz",
        train_losses=np.array(train_losses),
        train_accs=np.array(train_accs),
        test_accs=np.array(test_accs),
    )


def load_stats(path: Path):
    data = np.load(path)
    train_losses = data["train_losses"].tolist()
    train_accs = data["train_accs"].tolist()
    test_accs = data["test_accs"].tolist()
    return train_losses, train_accs, test_accs


def save_embeddings(
    model: torch.nn.Module,
    log_dir: Path,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
):
    model.eval()
    train_embeddings_list = []
    test_embeddings_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, _ in train_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            embeddings = model.forward_features(inputs)
            train_embeddings_list.append(embeddings.cpu())

        for inputs, _ in test_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            embeddings = model.forward_features(inputs)
            test_embeddings_list.append(embeddings.cpu())

    # Concate embeddings
    train_embeddings = (
        torch.cat(train_embeddings_list, dim=0).numpy().astype(np.float32)
    )
    test_embeddings = torch.cat(test_embeddings_list, dim=0).numpy().astype(np.float32)
    y_train = train_dataloader.dataset.labels.numpy()
    y_test = test_dataloader.dataset.labels.numpy()

    embeddings_path = log_dir / "artifacts" / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        embeddings_path / "embeddings.npz",
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        y_train=y_train,
        y_test=y_test,
    )


def load_embeddings(path: Path):
    data = np.load(path)
    train_embeddings = data["train_embeddings"]
    test_embeddings = data["test_embeddings"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    return train_embeddings, test_embeddings, y_train, y_test
