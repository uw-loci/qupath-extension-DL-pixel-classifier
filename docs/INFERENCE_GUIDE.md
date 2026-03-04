# Inference Guide

Step-by-step guide to applying a trained classifier to images.

## Overview

Inference applies a trained pixel classifier to new images, producing results as measurements, detection objects, or a live color overlay.

## Step 1: Open an Image

Open the image you want to classify in QuPath. The image type (brightfield, fluorescence) should match what the classifier was trained on.

## Step 2: Open the Inference Dialog

Go to **Extensions > DL Pixel Classifier > Apply Classifier...**

## Step 3: Select a Classifier

The classifier table shows all available trained models with their:
- Name and type (architecture)
- Number of input channels and classes
- Training date

Click a row to select it. The info panel below shows architecture details, input requirements, and class names.

> **Channel validation**: The channel panel will indicate if the current image has the wrong number of channels for the selected classifier.

## Step 4: Configure Output Type

| Output Type | Description | Best for |
|-------------|-------------|----------|
| **MEASUREMENTS** | Adds per-class probability values as annotation measurements | Quantification (% area per class) |
| **OBJECTS** | Creates detection or annotation objects from the classification map | Spatial analysis, counting structures |
| **OVERLAY** | Renders a live color overlay on the viewer | Visual inspection, quality checking |

### Object output options (OBJECTS only)

| Option | Description | Typical value |
|--------|-------------|---------------|
| **Object Type** | DETECTION (lightweight) or ANNOTATION (editable) | DETECTION for quantification |
| **Min Object Size** | Discard objects smaller than this area (um^2) | 10-100 um^2 |
| **Hole Filling** | Fill holes smaller than this area (um^2) | 5-50 um^2 |
| **Boundary Smoothing** | Simplify jagged boundaries (microns tolerance) | 0.5-2.0 um |

## Step 5: Configure Processing Options

These options are collapsed by default. Expand **PROCESSING OPTIONS** to adjust.

| Option | Default | Description |
|--------|---------|-------------|
| **Tile Size** | Auto-set from classifier | Should match training tile size |
| **Tile Overlap (%)** | 12.5% | Higher = better blending but slower (max 50%). See below. |
| **Blend Mode** | LINEAR | How overlapping tiles merge. LINEAR or GAUSSIAN recommended. |
| **Use GPU** | Yes | 10-50x faster than CPU |
| **Test-Time Augmentation (TTA)** | No | Apply D4 transforms (flips + 90-degree rotations) and average predictions. ~8x slower but typically 1-3% better quality. Best for final production runs. |

### Tile overlap and blending

Overlap determines how much adjacent tiles share:

| Overlap | Quality | Speed | Notes |
|---------|---------|-------|-------|
| 0% | Seams visible | Fastest | Objects may split at tile boundaries |
| 5-10% | Moderate | Fast | Some seam reduction |
| 10-15% | Good | Moderate | Recommended for seamless results |
| 15-25% | Best | Slower | ~2x processing time vs 0% |
| 25-50% | Diminishing returns | Much slower | Only needed for very large receptive fields |

For **overlay mode**, the overlap is automatically computed from a physical distance (default 25 um) using the image's pixel calibration. This ensures consistent overlap regardless of objective magnification. The preference **Overlay Overlap (um)** in Edit > Preferences controls this distance.

The **blend mode** controls how overlapping predictions merge:

- **LINEAR**: Weighted average favoring tile centers. Good default.
- **GAUSSIAN**: Gaussian-weighted for smoother transitions. Best for overlays.
- **NONE**: Last tile wins. Fastest but may show seams.

### Image-level normalization

The extension automatically computes normalization statistics across the entire image before starting inference. This ensures all tiles receive identical input normalization, eliminating the "blocky" tile boundary artifacts that occur when each tile independently computes its own statistics.

**Priority order:**
1. **Training dataset statistics** (best) -- stored in model metadata for newly trained models
2. **Image-level sampling** -- samples ~16 tiles in a 4x4 grid across the image (~1-3s one-time cost)
3. **Per-tile normalization** -- fallback if sampling fails

This is fully automatic and requires no configuration.

### BatchRenorm (model-internal normalization)

Newly trained models use **BatchRenorm** instead of standard BatchNorm for the network's internal normalization layers. Standard BatchNorm causes a train/eval disparity: running statistics accumulated during training diverge from actual tile statistics during inference, creating artifacts that no amount of overlap or blending can fix. BatchRenorm uses consistent global statistics in both modes, producing seamless tiled predictions. Older models trained with standard BatchNorm will continue to work but may exhibit more visible tile boundaries.

## Step 6: Set Application Scope

| Scope | Description |
|-------|-------------|
| **Whole image** | Classify the entire image (no annotations needed) |
| **All annotations** | Classify within all annotations |
| **Selected annotations** | Classify only within selected annotations |

> **Tip**: Use "Selected annotations" to test on a small region before processing the entire image.

### Backup option

Check **Create backup of annotation measurements** to save existing measurements before overwriting. Recommended when re-running inference on previously classified images.

## Step 7: Apply

Click **Apply** to start inference. Progress is shown in the QuPath log.

## Live Overlay Mode

For quick visual inspection without the full inference dialog:

1. **Extensions > DL Pixel Classifier > Toggle Prediction Overlay** (check/uncheck)
2. Select a classifier from the popup
3. The overlay renders as you pan and zoom
4. Uncheck to remove the overlay

Use **Extensions > DL Pixel Classifier > Remove Classification Overlay** to permanently remove and free resources.

## Copy as Script

Click the **"Copy as Script"** button to generate a Groovy script matching your current settings. See [SCRIPTING.md](SCRIPTING.md) for batch processing.
