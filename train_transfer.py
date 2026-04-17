# Yuyao Xu
# Apr 2026
# Fine-tunes a pretrained backbone on the WikiArt style subset.
# Compares two strategies:
#   (a) freeze_backbone=True  — only the final head is trained
#   (b) freeze_backbone=False — all layers are fine-tuned end-to-end
#
# Usage:
#   python train_transfer.py --backbone resnet18   --strategy frozen --data_dir data/processed_debug
#   python train_transfer.py --backbone resnet18   --strategy full   --data_dir data/processed_debug
#   python train_transfer.py --backbone densenet121 --strategy frozen --data_dir data/processed_debug
#   python train_transfer.py --backbone densenet121 --strategy full   --data_dir data/processed_debug

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.network import build_resnet18, build_densenet121, count_parameters
from utils.transforms import get_train_transform, get_val_transform
from utils.dataset import get_dataloaders
from train_baseline import train_one_epoch, evaluate, plot_curves


# Main training function for transfer learning
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/processed_debug")
    parser.add_argument("--strategy",   type=str, choices=["frozen", "full"], default="frozen",
                        help="frozen: freeze backbone; full: fine-tune all layers")
    parser.add_argument("--backbone",   type=str, choices=["resnet18", "densenet121"], default="resnet18",
                        help="Backbone architecture to use")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args(argv[1:])

    freeze     = (args.strategy == "frozen")
    ckpt_name  = f"{args.backbone}_{args.strategy}.pth"
    curve_name = f"{args.backbone}_{args.strategy}_curves.png"

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Backbone: {args.backbone} | Strategy: {args.strategy} | Device: {device}")

    # Load data
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data_dir,
        get_train_transform(),
        get_val_transform(),
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    # Build model based on selected backbone
    if args.backbone == "densenet121":
        model = build_densenet121(num_classes=num_classes, freeze_backbone=freeze).to(device)
    else:
        model = build_resnet18(num_classes=num_classes, freeze_backbone=freeze).to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    # Use a lower lr for full fine-tuning to avoid destroying pretrained weights
    lr = args.lr if freeze else args.lr * 0.1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names":      class_names,
                "backbone":         args.backbone,
                "strategy":         args.strategy,
                "val_acc":          vl_acc,
            }, f"checkpoints/{ckpt_name}")
            print(f"  -> Saved best model (val_acc={vl_acc:.4f})")

    # Save training curves
    os.makedirs("outputs", exist_ok=True)
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                f"outputs/{curve_name}")
    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main(sys.argv)