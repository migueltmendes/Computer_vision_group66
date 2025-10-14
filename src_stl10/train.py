from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def validate(net, testloader):
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
    net,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    device=None,
    use_amp=False,
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

    model_name = net.__class__.__name__
    ts = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(__file__).parents[1] / "runs" / "STL10" / model_name / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TB] Logging to: {log_dir.resolve()}")

    train_losses = []  # define a list to store losses -> for plotting effect
    train_accs = []  # list for accuracy
    # test_losses = []
    test_accs = []

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

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

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)

        test_acc = validate(net, test_loader)

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, ep)
        writer.add_scalar("Acc/Train", epoch_acc, ep)
        writer.add_scalar("Acc/Test", test_acc, ep)

        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"train_loss={epoch_loss:.4f} "
            f"train_acc={epoch_acc:.2f}%  "
            f"test_acc={test_acc:.2f}%  "
        )

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        # test_losses.append(test_loss)
        test_accs.append(test_acc)

        writer.flush()
        writer.close()

    return net, train_losses, train_accs, test_accs
