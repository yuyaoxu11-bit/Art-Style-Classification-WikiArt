# Art Style Classification on WikiArt

CS5330 Computer Vision — Final Project  
Yuyao Xu · Northeastern University · Spring 2026

## Project Overview

Classifies paintings into artistic styles (Impressionism, Baroque, Cubism, Romanticism, Realism,
Abstract Expressionism) using WikiArt dataset. Compares a custom CNN baseline against ResNet18
transfer learning (frozen vs. full fine-tuning). Includes a Gradio demo on Hugging Face Spaces: https://huggingface.co/spaces/yyx11/wikiart-style-classifier.

<img width="1760" height="893" alt="Screenshot 2026-04-12 at 16 53 17" src="https://github.com/user-attachments/assets/192a4e52-66ee-451b-8a5c-37663076c942" />


## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### Local (debugging with 50 images/class)

```bash
# Step 1: prepare small dataset
python prepare_dataset.py --debug

# Step 2: train CNN baseline
python train_baseline.py --data_dir data/processed_debug --epochs 5

# Step 3: transfer learning (frozen backbone)
python train_transfer.py --data_dir data/processed_debug --strategy frozen --epochs 5

# Step 3: transfer learning (full fine-tuning)
python train_transfer.py --data_dir data/processed_debug --strategy full --epochs 5

# Step 4: evaluate all models
python evaluate.py --data_dir data/processed_debug

# Step 5: run demo
python demo.py
```

### Colab (full training with 1500 images/class)

```bash
python prepare_dataset.py
python train_baseline.py --data_dir data/processed --epochs 20
python train_transfer.py --data_dir data/processed --strategy frozen --epochs 20
python train_transfer.py --data_dir data/processed --strategy full   --epochs 20
python evaluate.py --data_dir data/processed
```

## File Structure

```
wikiart-style-classification/
├── models/
│   └── network.py          # CNN baseline + ResNet18 definitions
├── utils/
│   ├── transforms.py       # data preprocessing and augmentation
│   └── dataset.py          # dataset loading, sampling, splitting
├── checkpoints/            # saved model weights
├── outputs/                # training curves, confusion matrices
├── prepare_dataset.py
├── train_baseline.py
├── train_transfer.py
├── evaluate.py
├── demo.py
├── requirements.txt
└── README.md
```

## Dataset

WikiArt via HuggingFace: https://huggingface.co/datasets/Artificio/WikiArt

## References

- Cetinic et al. (2018). Fine-tuning CNNs for Fine Art Classification. Expert Systems with Applications.
- Zhao et al. (2021). Compare the Performance of the Models in Art Classification. PLOS ONE.
