# Yuyao Xu
# Apr 2026
# Trains a custom CNN baseline from scratch on the WikiArt style subset.

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.network import CNNBaseline, count_parameters
from utils.transforms import get_train_transform, get_val_transform
from utils.dataset import get_dataloaders


# Train for one epoch, return average loss and accuracy
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# Evaluate model on a dataloader, return average loss and accuracy
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# Save training curves to a plot
def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", color="steelblue")
    ax1.plot(val_losses,   label="Val Loss",   color="coral")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accs, label="Train Acc", color="steelblue")
    ax2.plot(val_accs,   label="Val Acc",   color="coral")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves: {save_path}")


# Main training function
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/processed_debug")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--debug",      action="store_true",
                        help="Use small dataset for local debugging")
    args = parser.parse_args(argv[1:])

    # Device selection: MPS for Apple Silicon, CUDA for NVIDIA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data_dir,
        get_train_transform(),
        get_val_transform(),
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    # Build model
    model = CNNBaseline(num_classes=num_classes).to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": vl_acc,
            }, "checkpoints/cnn_baseline.pth")
            print(f"  -> Saved best model (val_acc={vl_acc:.4f})")

    # Save training curves
    os.makedirs("outputs", exist_ok=True)
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                "outputs/cnn_baseline_curves.png")
    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main(sys.argv)
