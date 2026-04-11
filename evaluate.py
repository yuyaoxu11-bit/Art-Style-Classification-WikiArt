# Yuyao Xu
# Apr 2026
# Evaluates all trained models on the test set.
# Generates confusion matrices and per-class F1 scores.

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from models.network import CNNBaseline, build_resnet18
from utils.transforms import get_val_transform
from utils.dataset import get_dataloaders


# Load a saved checkpoint and return model + class_names
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    # Determine model type from checkpoint filename
    name = os.path.basename(ckpt_path)
    if "cnn_baseline" in name:
        model = CNNBaseline(num_classes=num_classes)
    else:
        # frozen or full fine-tuned ResNet18
        model = build_resnet18(num_classes=num_classes, freeze_backbone=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


# Run inference on test set and collect predictions and ground truth labels
def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# Plot and save a confusion matrix
def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


# Main evaluation function
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed_debug")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args(argv[1:])

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # List checkpoints to evaluate
    checkpoints = {
        "CNN Baseline":       "cnn_baseline.pth",
        "ResNet18 (frozen)":  "resnet18_frozen.pth",
        "ResNet18 (full)":    "resnet18_full.pth",
    }

    for model_name, ckpt_file in checkpoints.items():
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")

        model, class_names = load_model(ckpt_path, device)

        _, _, test_loader, _ = get_dataloaders(
            args.data_dir,
            get_val_transform(),
            get_val_transform(),
            batch_size=32,
        )

        labels, preds = get_predictions(model, test_loader, device)

        # Print classification report (includes per-class F1)
        print(classification_report(labels, preds, target_names=class_names))

        # Plot confusion matrix
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            labels, preds, class_names,
            title=f"Confusion Matrix — {model_name}",
            save_path=f"outputs/confusion_{safe_name}.png",
        )


if __name__ == "__main__":
    main(sys.argv)
