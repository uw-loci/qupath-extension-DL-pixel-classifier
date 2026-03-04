# Training Guide

Step-by-step guide to training a deep learning pixel classifier.

## Overview

Training a classifier involves:
1. Preparing annotated training data in QuPath
2. Configuring the model and training parameters
3. Running training with progress monitoring
4. Reviewing training areas to identify annotation issues (optional)
5. Verifying the trained model

## Step 1: Prepare Annotations

### Create annotation classes

1. Open an image in QuPath
2. Right-click in the annotation class list (left panel) to add classes
3. Create at least two classes (e.g., "Tumor", "Stroma", "Background")

### Draw annotations

1. Use the brush, polygon, or polyline tools to annotate regions
2. Assign each annotation to a class (right-click > Set class)
3. Annotate representative examples of each class throughout the image

### Annotation best practices

- **Quality over quantity**: A few well-placed annotations are better than many sloppy ones
- **Cover variability**: Include examples from different regions, staining intensities, and tissue morphologies
- **Balance classes**: Try to annotate roughly similar areas for each class
- **Include boundaries**: Annotate along class boundaries where the model needs to make decisions
- **Use brush annotations**: The brush tool is efficient for painting regions. Line annotations also work but require setting the line stroke width parameter

## Step 2: Open the Training Dialog

Go to **Extensions > DL Pixel Classifier > Train Classifier...**

The dialog has collapsible sections. Sections marked with a collapse arrow can be expanded for advanced options.

## Loading Settings from a Previous Model

When retraining or iterating on a model, you can pre-populate all dialog settings from a previously trained classifier:

1. Click **"Retrain or refine a previously created model..."** at the top of the training dialog
2. Select a model from the table (sorted by date, newest first)
3. Click **OK**

This populates:
- **Architecture, backbone, tile size, downsample, context scale, epochs** from the model metadata
- **Learning rate, batch size, augmentation, scheduler, loss function, early stopping, and all other hyperparameters** from the model's saved training settings
- **Classifier name** auto-generated as `Retrain_OriginalName_YYYYMMDD`
- **Class auto-matching** after you load classes from images -- classes matching the source model are auto-selected

All fields can be adjusted before training. Older models (trained before this feature) will only populate the architecture-level settings; hyperparameters will keep their preference defaults.

## Step 3: Configure Basic Settings

### Classifier Info

| Setting | Description |
|---------|-------------|
| **Classifier Name** | Unique identifier (letters, numbers, underscore, hyphen). Used as filename. |
| **Description** | Optional free-text description for documentation. |

### Training Data Source

Check the project images to include in training. Only images with classified annotations are shown.

| Step | Description |
|------|-------------|
| **1. Select images** | Check images in the list. Use "Select All" / "Select None" for bulk selection. |
| **2. Load Classes** | Click **"Load Classes from Selected Images"** to populate the class list and initialize channels from the first image. |

Multi-image training combines patches from all selected images into one training set, improving generalization. If you previously loaded settings from a model, classes matching the source model are auto-selected after loading.

## Step 4: Configure Model Architecture

### Architecture

| Architecture | Best for | Reference |
|-------------|----------|-----------|
| **UNet** | General-purpose segmentation. Good default. | [Paper](https://arxiv.org/abs/1505.04597) |
| **Custom ONNX** | Importing externally trained models. Advanced users. | - |

### Backbone (Encoder)

The UNet architecture supports the following backbones:

**Standard backbones (ImageNet-pretrained):**

| Backbone | Speed | Accuracy | VRAM | Notes |
|----------|-------|----------|------|-------|
| resnet18 | Very fast | Good | Very low | Lightweight option |
| resnet34 | Fast | Good | Low | Best default |
| resnet50 | Medium | Better | Medium | For complex tasks |
| efficientnet-b0 | Very fast | Good | Very low | Lightweight |
| efficientnet-b1 | Fast | Good | Low | Slightly larger than b0 |
| efficientnet-b2 | Fast | Better | Low | Good accuracy/speed balance |
| mobilenet_v2 | Very fast | Good | Very low | Smallest model |

**Histology-pretrained backbones (ResNet-50 based, ~100 MB download on first use):**

| Backbone | Pretraining | Notes |
|----------|-------------|-------|
| resnet50_lunit-swav (Histology) | Lunit SwAV self-supervised on 19M TCGA patches | Best for general tissue classification |
| resnet50_lunit-bt (Histology) | Lunit Barlow Twins self-supervised on 19M TCGA patches | Alternative self-supervised approach |
| resnet50_kather100k (Histology) | Supervised on 100K colorectal tissue patches | Trained on colorectal tissue at 20x |
| resnet50_tcga-brca (Histology) | SimCLR self-supervised on TCGA breast cancer | Trained on breast cancer tissue at 20x |

Histology-pretrained backbones (marked "Histology" in the dropdown) use weights learned from millions of **H&E-stained brightfield patches at approximately 20x magnification (3-channel RGB)**. They typically produce better results for H&E histopathology with less training data.

> **Important:** Histology backbones are designed for H&E brightfield images. For **fluorescence, multiplex IF, or multi-channel (>3 channel) images**, use a standard ImageNet backbone (resnet34 or resnet50) instead. The histology-pretrained first conv layer encodes H&E color responses that do not transfer to fluorescence intensity patterns. See [Backbone Selection](BEST_PRACTICES.md#backbone-selection) for detailed guidance.

## Step 5: Configure Training Parameters

### Core parameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Epochs** | 50 | 50-200 for small datasets, 20-100 for large. Early stopping prevents overfitting. |
| **Batch Size** | 8 | 4-8 for 8GB VRAM with 512px tiles. Reduce if out-of-memory. |
| **Learning Rate** | 0.001 | Safe default for AdamW. Reduce to 1e-4 if loss oscillates. When using OneCycleLR, an LR finder auto-runs to suggest the optimal max learning rate. |
| **Validation Split** | 20% | 15-25% typical. 10% for very small datasets. |
| **Tile Size** | 512 | Must be divisible by 32. 256 for cell-level, 512 for tissue-level. |
| **Resolution** | 1x | 1x for cell-level, 2-4x for tissue-level classification. |
| **Tile Overlap** | 0% | 10-25% generates more patches from limited annotations. |
| **Line Stroke Width** | 0 | Width for polyline annotation masks (0 = use QuPath's stroke thickness). Increase for sparse lines. |

### Training Strategy (advanced, collapsed by default)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| **LR Scheduler** | One Cycle | Best default. "Reduce on Plateau" is a good alternative when training is noisy. [PyTorch docs](https://pytorch.org/docs/stable/optim.html) |
| **Loss Function** | CE + Dice | Recommended. Dice optimizes IoU directly. [smp losses](https://smp.readthedocs.io/en/latest/losses.html) |
| **Early Stop Metric** | Mean IoU | More reliable than validation loss. |
| **Early Stop Patience** | 15 | Epochs without improvement before stopping. |
| **Mixed Precision** | Enabled | Auto-detects BF16 (Ampere+ GPUs) or FP16. ~2x speedup. [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) |
| **Gradient Accumulation** | 1 | Accumulate gradients over N batches. Set 2-4 to simulate larger batches on limited VRAM. |
| **Progressive Resizing** | Off | Train at half resolution first (40% of epochs), then full resolution. Speeds up early training. |

### Transfer Learning (advanced, collapsed by default)

- **Use pretrained weights**: Almost always recommended. [Guide](https://cs231n.github.io/transfer-learning/)
- **Layer freezing**: Freeze early encoder layers to prevent overfitting on small datasets
  - Small datasets (<500 tiles): Freeze most encoder layers
  - Medium datasets (500-5000 tiles): Freeze early layers only
  - Large datasets (>5000 tiles): Unfreeze nearly all layers

### Automatic Optimizations

The following optimizations are applied automatically -- no configuration needed:

- **AdamW optimizer** with fast.ai-tuned hyperparameters (betas=0.9/0.99, eps=1e-5, weight_decay=0.01). AdamW decouples weight decay from the gradient update, producing better generalization than Adam.
- **Discriminative learning rates**: When using pretrained weights, the encoder automatically receives 1/10th of the base learning rate while the decoder and segmentation head train at the full rate. This prevents catastrophic forgetting of pretrained features.
- **LR Finder**: When using the One Cycle scheduler on a new training run (not resuming from checkpoint), an automatic learning rate range test runs before training to find the optimal max learning rate. The suggested LR is logged and used as the OneCycleLR max_lr.
- **BF16 auto-detection**: On Ampere+ GPUs (RTX 3000 series and newer), training and inference automatically use BF16 mixed precision instead of FP16. BF16 has a larger dynamic range and does not require gradient scaling, improving stability.

### Data Augmentation (collapsed by default)

| Augmentation | Default | Notes |
|-------------|---------|-------|
| Horizontal flip | On | Almost always beneficial |
| Vertical flip | On | Safe for most histopathology |
| Rotation (90 deg) | On | Combines with flips for 8x augmentation |
| Color jitter | Off | Enable for H&E, disable for fluorescence |
| Elastic deformation | Off | Effective but ~30% slower. [Albumentations](https://albumentations.ai/docs/) |

## Step 6: Select Channels and Classes

### Channels

- For RGB brightfield images, channels are auto-configured
- For fluorescence/spectral images, select and order channels manually
- Channel order must match at inference time

### Classes

- Select at least 2 annotation classes
- Adjust weight multipliers for imbalanced classes (>1.0 boosts rare classes)

## Step 7: Start Training

Click **Start Training**. A progress window shows:
- Patch extraction progress
- Epoch-by-epoch training and validation loss
- Live loss chart
- Early stopping status

### What to watch for

- **Training loss should decrease** over epochs
- **Validation loss should follow** training loss down
- **Diverging losses** (val goes up, train goes down) = overfitting
- **Both losses plateau** = model has converged

## Step 8: Review Training Areas (Optional)

When training completes successfully, a **"Review Training Areas..."** button appears in the progress dialog. This runs the trained model over all training tiles and ranks them by loss to help you identify annotation errors, hard cases, and model failures.

> **Important:** Training tiles are cleaned up when you close the progress dialog. Review your training areas *before* closing.

### How it works

1. Click **"Review Training Areas..."** in the completed training progress dialog
2. The model evaluates every training tile (train + val splits) and computes per-tile metrics
3. A results dialog opens showing tiles sorted by loss (highest first)

### The Training Area Issues dialog

| Column | Description |
|--------|-------------|
| **Image** | Source image name (for multi-image training) |
| **Split** | Whether the tile was in the train or val split |
| **Loss** | Cross-entropy loss -- higher = model disagrees more with annotation |
| **Disagree%** | Percentage of pixels where prediction differs from annotation |
| **mIoU** | Mean Intersection-over-Union across classes present in the tile |
| **Classes** | Which classes are present in the tile |

### Filtering and navigation

- **Filter by split**: Use the dropdown to show only train or val tiles
- **Loss threshold**: Use the slider to show only tiles above a minimum loss
- **Double-click a row** to navigate the QuPath viewer to that tile's location
- For multi-image projects, double-clicking automatically switches to the correct image

### What to look for

| Pattern | Likely cause | Action |
|---------|-------------|--------|
| Very high loss on a few tiles | Annotation error (wrong class) | Fix the annotation and retrain |
| High loss cluster in one region | Inconsistent annotation criteria | Re-annotate the region consistently |
| High loss on val but not train | Model memorizing, not generalizing | Add more diverse annotations |
| High disagreement on boundaries | Normal -- boundaries are hardest | Consider annotating boundaries more carefully |
| Entire class has high loss | Class may be poorly defined | Check if the class has consistent visual features |

See [BEST_PRACTICES.md](BEST_PRACTICES.md#interpreting-tile-evaluation-results) for detailed guidance on interpreting results and improving annotations.

## Step 9: Verify the Result

When training completes, the classifier is saved to your QuPath project under `classifiers/`. View it via **Extensions > DL Pixel Classifier > Manage Models...**

### Quick verification

1. Apply the classifier to a test annotation (see [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md))
2. Check results visually using the overlay output type
3. If results are poor, see [BEST_PRACTICES.md](BEST_PRACTICES.md) for improvement strategies

## Copy as Script

Click the **"Copy as Script"** button in the training dialog to generate a Groovy script matching your current settings. Paste into QuPath's Script Editor for reproducible and batch training workflows. See [SCRIPTING.md](SCRIPTING.md) for details.
