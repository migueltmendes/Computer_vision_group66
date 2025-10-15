import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src_stl10.utils import save_stats


def validate(net: torch.nn.Module, testloader: DataLoader):
    """
    Validates the model on the test dataset.

    Args:
        net (torch.nn.Module): The neural network model.
        testloader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        float: The accuracy of the model on the test dataset.
    """

    # Set the model to evaluation mode
    net.eval()

    # Determine the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct, total = 0, 0

    # Disable gradient computation
    with torch.no_grad():

        # Iterate over the test dataset
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {accuracy:.2f} %")

    return accuracy


def train(
    net: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    hp_conf: str,
    device=None,
    use_amp: bool=False,
):
    """
    Trains the neural network model.

    Args:
        net (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        epochs (int): The number of epochs to train the model.

    Returns:
        None
    """

    # YOUR CODE HERE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    log_dir = Path(__file__).parents[1] / "runs" / "STL10" / hp_conf
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TB] Logging to: {log_dir.resolve()}")

    train_losses = []  # define a list to store losses -> for plotting effect
    train_accs = []  # list for accuracy
    # test_losses = []
    test_accs = []

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Cosine annealing scheduler (no restarts): decay LR smoothly over the full training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    for ep in range(1, epochs + 1):
        net.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)

        test_acc = validate(net, test_loader)

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, ep)
        writer.add_scalar("Acc/Train", epoch_acc, ep)
        writer.add_scalar("Acc/Test", test_acc, ep)
        # log learning rate
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], ep)

        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"train_loss={epoch_loss:.4f} "
            f"train_acc={epoch_acc:.2f}%  "
            f"test_acc={test_acc:.2f}%  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        # test_losses.append(test_loss)
        test_accs.append(test_acc)

    writer.flush()

    # step the cosine scheduler once per epoch
    scheduler.step()

    # close the SummaryWriter after training
    writer.close()

    # Save weights in writer folder
    checkpoint_path = log_dir / "artifacts" / "weights"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), checkpoint_path / "model.pth")

    # Save losses and accuracies
    save_stats(
        log_dir, train_losses, train_accs, test_accs
    )

    return net, train_losses, train_accs, test_accs, log_dir
