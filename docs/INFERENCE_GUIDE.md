# Inference Guide

Step-by-step guide to applying a trained classifier to images.

## Overview

Inference applies a trained pixel classifier to new images, producing results as measurements, detection objects, or a color overlay.

## Step 1: Open an Image

Open the image you want to classify in QuPath. The image type (brightfield, fluorescence) should match what the classifier was trained on.

## Step 2: Open the Inference Dialog

Go to **Extensions > DL Pixel Classifier > Apply Classifier...**

## Step 3: Select a Classifier

The classifier table shows all available trained models with columns for:
- **Name** -- classifier identifier
- **Type** -- architecture (e.g., unet, muvit)
- **Channels** -- input channel count (with context scale info, e.g., "3 +2x ctx")
- **Classes** -- number of output classes
- **Trained** -- training date

Click a row to select it. The info panel below shows the architecture + backbone, input channels + context scale + tile dimensions, downsample level, and class names.

### Channel Mapping

The **CHANNEL MAPPING** section shows how the classifier's expected channels map to the current image. Each row displays:
- **Expected** channel name from the classifier
- **Mapped To** the image channel it will use
- **Status**: [OK] (exact match), [?] (fuzzy/substring match), or [X] (unmapped)

For unmatched channels, use the dropdown to manually remap to the correct image channel. For brightfield images, channels are auto-configured and this section collapses automatically.

## Step 4: Configure Output Type

| Output Type | Description | Best for |
|-------------|-------------|----------|
| **RENDERED_OVERLAY** | Batch inference with tile blending, producing a seamless overlay. **Default and recommended.** Best for validating classifier quality -- accurately represents what OBJECTS output would look like. | Quality validation, visual comparison |
| **MEASUREMENTS** | Adds per-class probability values as annotation measurements | Quantification (% area per class) |
| **OBJECTS** | Creates detection or annotation objects from the classification map | Spatial analysis, counting structures |
| **OVERLAY** | Renders a live on-demand color overlay as you pan and zoom | Quick visual inspection |

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
| **Tile Size** | Auto-set from classifier | Should match training tile size. Range: 64-8192. |
| **Tile Overlap (%)** | 12.5% | Higher = better blending but slower (max 50%). See below. |
| **Blend Mode** | LINEAR | How overlapping tiles merge. See blend mode details below. |
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

For **overlay mode**, the overlap is automatically computed from a physical distance (default 25 um) using the image's pixel calibration. This ensures consistent overlap regardless of objective magnification. The preference **Overlay Overlap (um)** in **Edit > Preferences > DL Pixel Classifier** controls this distance.

The **blend mode** controls how overlapping predictions merge:

| Blend Mode | Description | Recommended for |
|------------|-------------|-----------------|
| **LINEAR** | Weighted average favoring tile centers. Good balance of quality and speed. | CNN models (UNet) |
| **GAUSSIAN** | Cosine-bell blending for smoother transitions. Handles smooth prediction gradients from global attention better than linear. Set automatically for MuViT overlays. | ViT/MuViT models |
| **CENTER_CROP** | Keep only center predictions, discard overlap margins. Zero boundary artifacts but ~4x slower (more tiles needed). | When artifact-free results are critical |
| **NONE** | No blending; last tile wins. Fastest but may show visible tile seams. | Debugging, or with 0% overlap |

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
| **Apply to whole image** | Classify the entire image (no annotations needed) |
| **Apply to all annotations** | Classify within all annotations (default) |
| **Apply to selected annotations only** | Classify only within selected annotations |

> **Tip**: Use "Apply to selected annotations only" to test on a small region before processing the entire image.

### Backup option

Check **Create backup of annotation measurements** to save existing measurements before overwriting. Recommended when re-running inference on previously classified images.

## Step 7: Apply

Click **Apply** to start inference. Progress is shown in the QuPath log.

> **Note:** All inference dialog settings (output type, blend mode, smoothing, application scope, backup) are remembered across sessions.

## Live Overlay Mode

For quick visual inspection without the full inference dialog:

1. **Extensions > DL Pixel Classifier > Toggle Prediction Overlay** (check/uncheck)
2. Select a classifier from the popup
3. The overlay renders as you pan and zoom
4. Uncheck to remove the overlay (the overlay is destroyed, not just hidden -- you will need to re-select a classifier to restore it)

Use **Extensions > DL Pixel Classifier > Remove Classification Overlay** to explicitly remove the overlay and free resources.

## Copy as Script

Click the **"Copy as Script"** button (left side of the button bar) to generate a Groovy script matching your current settings. See [SCRIPTING.md](SCRIPTING.md) for batch processing.
