# Yuyao Xu
# Apr 2026
# Gradio demo for art style classification.
# Supports image upload and live webcam capture.
# Loads the best available checkpoint automatically.

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from models.network import CNNBaseline, build_resnet18
from utils.transforms import get_inference_transform

# Priority order for auto checkpoint selection
CKPT_PRIORITY = [
    "checkpoints/resnet18_full.pth",
    "checkpoints/resnet18_frozen.pth",
    "checkpoints/cnn_baseline.pth",
]


# Load model from checkpoint, return model and class names
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    name = os.path.basename(ckpt_path)
    if "cnn_baseline" in name:
        model = CNNBaseline(num_classes=num_classes)
    else:
        model = build_resnet18(num_classes=num_classes, freeze_backbone=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from: {ckpt_path}")
    print(f"Classes: {class_names}")
    return model, class_names


# Run inference on a single PIL image, return top-k predictions as a dict
def predict(image: Image.Image, model, class_names, device, top_k=3):
    transform = get_inference_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))
    results = {
        class_names[idx.item()]: round(prob.item(), 4)
        for prob, idx in zip(top_probs, top_indices)
    }
    return results


# Build and launch the Gradio interface
def launch_demo(model, class_names, device):
    def classify(image):
        if image is None:
            return {}
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        return predict(pil_image, model, class_names, device)
    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(
            sources=["upload", "webcam"],
            label="Upload a painting or take a photo",
        ),
        outputs=gr.Label(
            num_top_classes=3,
            label="Predicted Art Style",
        ),
        title="🎨 Art Style Classifier",
        description=(
            "Upload a painting or use your webcam to identify its artistic style. "
            "Trained on WikiArt using ResNet18 transfer learning."
        ),
        examples=[],
    )

    demo.launch(share=False)


# Main function
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (auto-selects best if not specified)")
    args = parser.parse_args(argv[1:])
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Auto-select checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        for path in CKPT_PRIORITY:
            if os.path.exists(path):
                ckpt_path = path
                break

    if ckpt_path is None:
        print("No checkpoint found. Please train a model first.")
        print("Run: python 2_train_baseline.py or python 3_train_transfer.py")
        sys.exit(1)
    model, class_names = load_model(ckpt_path, device)
    launch_demo(model, class_names, device)


if __name__ == "__main__":
    main(sys.argv)
