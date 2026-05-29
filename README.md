# Final Year Project - Emotion Classification

This repository contains a complete facial emotion classification study built around the augmented CK+ dataset. The project compares four model families trained from scratch in PyTorch: VGG16, ResNet50, DenseNet121, and Vision Transformer (ViT).

## Project Overview

The dataset is organized into train, validation, and test splits, each split containing the same eight emotion classes:

- Anger
- Contempt
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

The images are stored as 48x48 JPG files with three channels. The repository includes the training notebooks, final checkpoints, class-index mappings, and a saved training-history file used to compare the models.

## Repository Layout

- `dataset/` - augmented CK+ emotion dataset and dataset notes
- `vgg/` - VGG16 training notebook, checkpoint files, and class mapping
- `resnet/` - ResNet50 training notebook, checkpoint files, and class mapping
- `densenet/` - DenseNet121 training notebook, checkpoint files, and class mapping
- `vit/` - Vision Transformer training notebook, checkpoint files, and class mapping
- `training_history.json` - consolidated train/validation history for all models
- `comparative_analysis.ipynb` - root-level notebook that compares all four models

## Dataset Architecture

The dataset is split into three balanced folders:

- `dataset/train/<class>/`
- `dataset/validation/<class>/`
- `dataset/test/<class>/`

Each class has the same number of samples in every split, which makes the evaluation fair and keeps the metrics directly comparable across models.

## Models

### VGG16

The VGG notebook implements a scratch-built VGG16-style CNN adapted to 48x48 inputs.

Architecture summary:

- Five convolutional blocks
- 3x3 convolutions with batch normalization and ReLU
- Max-pooling after the first three blocks
- Adaptive average pooling before the classifier
- Fully connected head: 512 -> 256 -> 8

Why it works well here:

- The architecture is strong at local texture extraction
- Batch normalization stabilizes training
- The updated inference cell now uses a small test-time augmentation pass so the single-image confidence is more robust

### ResNet50

The ResNet notebook builds a ResNet50-style bottleneck network from scratch.

Architecture summary:

- 3x3 convolutional stem for small inputs
- Bottleneck residual blocks with 1x1, 3x3, 1x1 convolutions
- Stage layout: 3, 4, 6, 3 bottleneck blocks
- Channel expansion factor of 4 inside each bottleneck
- Adaptive average pooling and a linear classifier to 8 classes

Why it works well here:

- Residual shortcuts preserve gradient flow
- Deeper convolutional feature extraction helps distinguish subtle facial cues
- It typically gives a strong balance between validation accuracy and test stability

### DenseNet121

The DenseNet notebook implements a DenseNet121-style model from scratch.

Architecture summary:

- 3x3 convolutional stem
- Dense blocks with concatenated feature reuse
- Block configuration: 6, 12, 24, 16 layers
- Growth rate of 32
- Transition layers between dense blocks for compression and downsampling
- Adaptive average pooling and a final linear classifier

Why it works well here:

- Dense feature reuse improves gradient propagation
- The model is parameter-efficient compared with some other deep CNNs
- It often shows strong validation accuracy on compact image datasets

### Vision Transformer (ViT)

The ViT notebook implements a scratch transformer for 48x48 images.

Architecture summary:

- Patch size: 6x6
- Number of patches: 64
- Embedding dimension: 256
- Transformer depth: 8 encoder blocks
- Attention heads: 8
- MLP hidden dimension: 512
- Learnable class token and positional embeddings
- LayerNorm + linear classifier head

Recent training improvements:

- Fixed augmentation order so `RandomErasing` runs after tensor conversion and normalization
- Added normalization to both training and validation pipelines
- Reduced over-regularization and increased patience so the model can train longer and validate properly
- Added gradient clipping for more stable optimization

## Training And Evaluation

Each notebook follows the same high-level workflow:

1. Inspect the dataset split layout
2. Build transforms and dataloaders
3. Define the model architecture from scratch
4. Train the model and track train/validation loss and accuracy
5. Save the best checkpoint
6. Run test-set evaluation
7. Generate a single-image inference example

The saved metrics in the notebooks include:

- Accuracy
- Precision
- Recall
- F1-score
- Weighted average metrics
- Macro average metrics
- Confusion matrix and per-class breakdowns

## Comparative Analysis Notebook

The root notebook, `comparative_analysis.ipynb`, reads:

- `training_history.json` for the training and validation curves
- The saved notebook outputs for each model to build a side-by-side metric comparison

It produces:

- Training vs validation accuracy and loss curves for all four models
- A summary table for accuracy, weighted precision/recall/F1, and macro precision/recall/F1
- A confidence comparison for the single-image inference examples

## How To Run

Open the notebook you want to inspect and run the cells top to bottom.

Suggested order:

1. Run the model notebook you want to study or retrain
2. Run `comparative_analysis.ipynb` to compare all models side by side
3. Use the saved checkpoints in each model folder for inference

### Real-Time Webcam App

The repository includes a fully-featured real-time webcam application with a rich HUD overlay. It loads trained PyTorch checkpoints and performs live face detection plus emotion classification at speed.

#### Quick Start

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run with any supported architecture
python realtime_emotion_app.py --architecture resnet
python realtime_emotion_app.py --architecture vgg
python realtime_emotion_app.py --architecture densenet
python realtime_emotion_app.py --architecture vit
```

#### All Options

| Flag | Default | Description |
|---|---|---|
| `--architecture` | `resnet` | Model to load: `resnet`, `vgg`, `densenet`, `vit` |
| `--checkpoint` | auto | Override the `.pt` checkpoint path |
| `--class-indices` | auto | Override the `class_indices.json` path |
| `--camera-index` | `0` | Webcam device index |
| `--confidence-threshold` | `0.0` | Min confidence to show label (0–1) |
| `--width` | `1280` | Capture frame width |
| `--height` | `720` | Capture frame height |

#### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `M` | Cycle to next model architecture (lazy-loads on demand) |
| `S` | Save screenshot to `screenshots/` folder |
| `H` | Toggle HUD overlay |
| `D` | Toggle debug mode (mini probability bars under each face) |

#### Features

- **Rich HUD**: Top status bar shows active model, live FPS, face count, and compute device
- **Emotion probability panel**: Live bar chart of all 8 emotion scores for the primary face
- **Per-emotion colours**: Each emotion has a unique colour used for bounding boxes and bars
- **EMA smoothing**: Exponential moving-average on raw probabilities reduces flickering
- **Lazy model switching**: Switch architectures at runtime with `M`; models load once and are cached
- **Screenshot capture**: One-keystroke save of annotated frames
- **Rounded UI**: Polished rounded-rectangle overlays with transparency

## Notes

- The dataset is balanced by design, so weighted and macro metrics are both meaningful.
- The project uses 48x48 images, so the architectures are adapted for smaller spatial resolution rather than standard ImageNet-sized inputs.
- If you retrain the notebooks after the ViT fixes, the comparative notebook will update automatically from the saved history and outputs.
- For the webcam app, install the runtime dependencies from `requirements.txt` first if they are not already available in your environment.

## Files Of Interest

- `dataset/about_dataset.txt`
- `training_history.json`
- `vgg/notebook.ipynb`
- `resnet/notebook.ipynb`
- `densenet/notebook.ipynb`
- `vit/notebook.ipynb`
- `comparative_analysis.ipynb`

## Expected Outcome

The current codebase is set up to compare the four architectures on the same balanced dataset and to document the architecture choices, evaluation metrics, and inference behavior in a single place.