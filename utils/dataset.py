# Yuyao Xu
# Apr 2026
# Dataset loading, sampling, and splitting utilities for WikiArt

import os
import random
import shutil
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Target style categories and sample limit per category
STYLE_CLASSES = [
    "Impressionism",
    "Baroque",
    "Cubism",
    "Romanticism",
    "Realism",
    "Abstract_Expressionism",
]

# Use a small subset size for local debugging
DEBUG_SAMPLES_PER_CLASS = 50
FULL_SAMPLES_PER_CLASS  = 1500


# Sample a balanced subset from a flat source directory and organize into
# train/val/test splits following ImageFolder format:
#   root/train/ClassName/img.jpg
#   root/val/ClassName/img.jpg
#   root/test/ClassName/img.jpg
def prepare_local_dataset(source_dir, output_dir, samples_per_class,
                           train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        for cls in STYLE_CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in STYLE_CLASSES:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing class directory: {cls_dir}")
            continue

        all_images = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sampled = random.sample(all_images, min(samples_per_class, len(all_images)))

        n_train = int(len(sampled) * train_ratio)
        n_val   = int(len(sampled) * val_ratio)

        splits = {
            "train": sampled[:n_train],
            "val":   sampled[n_train:n_train + n_val],
            "test":  sampled[n_train + n_val:],
        }

        for split, files in splits.items():
            for fname in files:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(output_dir, split, cls, fname)
                shutil.copy2(src, dst)

        print(f"[{cls}] train={len(splits['train'])}  "
              f"val={len(splits['val'])}  test={len(splits['test'])}")

    print(f"\nDataset written to: {output_dir}")


# Build DataLoaders for train, val, and test splits
def get_dataloaders(data_dir, train_transform, val_transform, batch_size=32):
    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset   = ImageFolder(os.path.join(data_dir, "val"),   transform=val_transform)
    test_dataset  = ImageFolder(os.path.join(data_dir, "test"),  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
