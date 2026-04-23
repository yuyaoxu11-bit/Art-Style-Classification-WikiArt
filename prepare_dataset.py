# Yuyao Xu
# Apr 2026
# Downloads WikiArt from HuggingFace and prepares a balanced local subset
# for art style classification.
#
#   python prepare_dataset.py       full mode 1500/class
#   python prepare_dataset.py --debug       debug mode 50/class

import sys
import os
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from utils.dataset import STYLE_CLASSES, DEBUG_SAMPLES_PER_CLASS, FULL_SAMPLES_PER_CLASS
from utils.dataset import prepare_local_dataset


# Download WikiArt from HuggingFace and save images by style into source_dir
def download_wikiart(source_dir, samples_per_class, seed=42):
    print("Loading WikiArt dataset from HuggingFace...")
    ds = load_dataset("Artificio/WikiArt", split="train")
    # Normalize style label strings
    style_set = set(STYLE_CLASSES)

    # Collect image paths per class
    from collections import defaultdict
    import random
    random.seed(seed)
    buckets = defaultdict(list)
    for item in ds:
        style = item.get("style", "")
        # Match ignoring spaces/underscores
        normalized = style.replace(" ", "_")
        if normalized in style_set or style in style_set:
            key = normalized if normalized in style_set else style
            buckets[key].append(item)

    for cls, items in buckets.items():
        cls_dir = os.path.join(source_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        sampled = random.sample(items, min(samples_per_class * 2, len(items)))
        saved = 0
        for i, item in enumerate(sampled):
            if saved >= samples_per_class * 2:
                break
            try:
                img = item["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(os.path.join(cls_dir, f"{cls}_{i:05d}.jpg"))
                saved += 1
            except Exception as e:
                print(f"[WARN] Skipping image {i}: {e}")
        print(f"[{cls}] Saved {saved} images")

    print(f"\nRaw images written to: {source_dir}")


# Main function
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Use small subset (50 images/class) for local debugging")
    parser.add_argument("--source_dir", type=str, default="data/raw",
                        help="Directory to save raw downloaded images")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory for train/val/test split")
    args = parser.parse_args(argv[1:])
    samples = DEBUG_SAMPLES_PER_CLASS if args.debug else FULL_SAMPLES_PER_CLASS
    print(f"Mode: {'DEBUG' if args.debug else 'FULL'} — {samples} images/class")
    # Download raw images
    if not os.path.exists(args.source_dir):
        download_wikiart(args.source_dir, samples_per_class=samples)
    else:
        print(f"Source directory already exists, skipping download: {args.source_dir}")
    # Organize into train/val/test splits
    output = args.output_dir + ("_debug" if args.debug else "")
    prepare_local_dataset(args.source_dir, output, samples_per_class=samples)


if __name__ == "__main__":
    main(sys.argv)
