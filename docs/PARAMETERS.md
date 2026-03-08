# Parameter Reference

Complete reference for every parameter in the training and inference dialogs. This mirrors the tooltip content in the extension UI.

## Training Parameters

### Retrain or Refine a Previously Created Model

| Parameter | Description |
|-----------|-------------|
| **Retrain or refine a previously created model...** | Button at the top of the dialog. Opens a model picker with all trained classifiers (name, architecture, classes, date). Populates all dialog fields from the selected model's metadata and saved training settings. Older models without saved training settings will only populate architecture-level fields. |
| **Loaded model label** | Shows the name of the model whose settings were loaded (or empty if none). |

### Classifier Info

| Parameter | Type | Description |
|-----------|------|-------------|
| **Classifier Name** | Text | Unique identifier for the classifier. Used as filename. Only letters, numbers, underscore, and hyphen allowed. |
| **Description** | Text | Optional free-text description stored in classifier metadata. |

### Training Data Source

| Parameter | Type | Description |
|-----------|------|-------------|
| **Image selection list** | Checkbox list | Check project images to include in training. Only images with classified annotations are shown. |
| **Load Classes from Selected Images** | Button | Reads annotations from selected images, populates the class list, and initializes channel configuration. If a model was loaded, auto-matches classes. |

### Model Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Architecture** | UNet, Custom ONNX | Segmentation architecture. UNet is the best general-purpose choice. Custom ONNX allows importing externally trained models. See [UNet paper](https://arxiv.org/abs/1505.04597). |
| **Backbone** | resnet18, resnet34, resnet50, efficientnet-b0/b1/b2, mobilenet_v2, plus 4 histology-pretrained variants (resnet50_lunit-swav, resnet50_lunit-bt, resnet50_kather100k, resnet50_tcga-brca) | Pretrained encoder network. Histology backbones use H&E tissue-pretrained weights (20x, 3-channel RGB) instead of ImageNet -- best for H&E brightfield. For fluorescence or multi-channel images, use ImageNet backbones. See [Backbone Selection](BEST_PRACTICES.md#backbone-selection). |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 50 | 1-1000 | Complete passes through training data. Early stopping prevents overfitting. 50-200 for small datasets, 20-100 for large. |
| **Batch Size** | 8 | 1-128 | Tiles per training step. Larger = more stable gradients, more VRAM. 4-8 for 8GB VRAM with 512px tiles. |
| **Learning Rate** | 0.001 | 0.00001-1.0 | Step size for gradient descent. 1e-3 for AdamW default, 1e-4 if oscillating, 1e-5 for full fine-tuning. When using OneCycleLR, an automatic LR finder runs before training to suggest an optimal max learning rate. |
| **Validation Split** | 20% | 5-50% | Percentage held out for validation. 15-25% typical. |
| **Tile Size** | 512 | 64-1024 | Patch size in pixels. Must be divisible by 32. 256 for cell-level, 512 for tissue-level. |
| **Resolution** | 1x | 1x, 2x, 4x, 8x | Image downsample level. Higher = more context per tile, less detail. |
| **Tile Overlap** | 0% | 0-50% | Overlap between training tiles. 10-25% generates more patches from limited annotations. |
| **Line Stroke Width** | 0 | 0-50 | Pixel width for polyline annotation masks. Default 0 means "use QuPath's annotation stroke thickness". |

### Training Strategy (advanced)

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **LR Scheduler** | One Cycle | One Cycle, Cosine Annealing, Step Decay, Reduce on Plateau, None | Learning rate schedule. One Cycle is recommended for most cases. Reduce on Plateau automatically lowers the LR when the monitored metric stops improving (factor=0.5, patience=10). See [PyTorch schedulers](https://pytorch.org/docs/stable/optim.html). |
| **Loss Function** | CE + Dice | CE + Dice, Cross Entropy | CE + Dice is recommended. Dice directly optimizes IoU. See [smp losses](https://smp.readthedocs.io/en/latest/losses.html). |
| **Early Stop Metric** | Mean IoU | Mean IoU, Validation Loss | Mean IoU is more reliable than loss for stopping. |
| **Early Stop Patience** | 15 | 3-50 | Epochs without improvement before stopping. 10-15 default, 20-30 for noisy curves. |
| **Mixed Precision** | Enabled | On/Off | Automatic mixed precision. Auto-detects BF16 on Ampere+ GPUs (RTX 3000+), falls back to FP16 with gradient scaling on older GPUs. ~2x speedup. See [PyTorch AMP](https://pytorch.org/docs/stable/amp.html). |
| **Gradient Accumulation** | 1 | 1-8 | Number of batches to accumulate before updating weights. Effectively multiplies batch size without increasing VRAM. Set to 2-4 to simulate larger batches on limited GPU memory. |
| **Progressive Resizing** | Off | On/Off | Train at half resolution for the first 40% of epochs, then switch to full resolution. Speeds up early training and acts as implicit regularization. Inspired by fast.ai. |

### Transfer Learning (advanced)

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Use Pretrained Weights** | On | Initialize encoder with pretrained weights. Almost always recommended. See [transfer learning guide](https://cs231n.github.io/transfer-learning/). |
| **Training Data preset** | Medium | Freeze strategy based on dataset size: Small (<500), Medium (500-5000), Large (>5000), Custom. |
| **Freeze All Encoder** | Button | Freeze all encoder layers. Most conservative -- only decoder trains. |
| **Unfreeze All** | Button | Unfreeze all layers for full fine-tuning. Risk of overfitting on small datasets. |
| **Use Recommended** | Button | Apply server's recommended freeze configuration for the selected backbone. |

### Channel Configuration

| Parameter | Description |
|-----------|-------------|
| **Available Channels** | Image channels available for selection. Multi-select with Ctrl+click. |
| **Selected Channels** | Channels used as model input. Order must match at inference time. |
| **Normalization** | Per-channel intensity normalization: PERCENTILE_99 (recommended), MIN_MAX, Z_SCORE, FIXED_RANGE. |

### Annotation Classes

| Parameter | Description |
|-----------|-------------|
| **Class list** | Annotation classes found in the image. At least 2 must be selected. |
| **Weight multiplier** | Per-class weight multiplier. >1.0 boosts underrepresented classes. |

### Data Augmentation (advanced)

| Augmentation | Default | Description |
|-------------|---------|-------------|
| **Horizontal flip** | On | Mirror tiles left-right. Almost always beneficial. |
| **Vertical flip** | On | Mirror tiles top-bottom. Safe for most histopathology. |
| **Rotation (90 deg)** | On | Rotate by 0/90/180/270. Combines with flips for 8x augmentation. |
| **Color jitter** | Off | Perturb brightness/contrast/saturation. Good for H&E, not for fluorescence. |
| **Elastic deformation** | Off | Smooth spatial deformations. Effective but ~30% slower. See [Albumentations](https://albumentations.ai/docs/). |

## MAE Pretraining Parameters

These parameters are available in the **MAE Pretrain Encoder** dialog (**Extensions > DL Pixel Classifier > Utilities > MAE Pretrain Encoder...**). MAE pretraining is a standalone workflow for self-supervised pretraining of MuViT encoder weights on unlabeled image tiles.

### Model Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model Configuration** | muvit-small, muvit-base, muvit-large | MuViT model size. Must match the model you plan to use for supervised training. Larger models learn richer features but need more data and training time. |
| **Patch Size** | 8, 16 | Vision transformer patch size. Smaller patches capture finer detail but increase compute. 16 is recommended for most cases. |
| **Level Scales** | Text (e.g., "1,4") | Comma-separated multi-resolution scale factors for multi-scale feature fusion. |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 100 | 10-2000 | Number of pretraining epochs. Auto-suggested based on dataset size: <50 tiles -> 500, <200 -> 300, <1000 -> 100, 1000+ -> 50. |
| **Mask Ratio** | 0.75 | 0.5-0.9 | Fraction of image patches masked during pretraining. Higher = harder reconstruction task. 0.75 is the standard MAE default. |
| **Batch Size** | 8 | 1-64 | Tiles per training step. Reduce if out-of-memory. |
| **Learning Rate** | 0.00015 | 0.00001-0.01 | AdamW learning rate. The default (1.5e-4) follows the original MAE paper recommendation. Displayed with 5 decimal places. |
| **Warmup Epochs** | 5 | 0-50 | Number of epochs for linear learning rate warmup from 0 to the target learning rate. |

### Data

| Parameter | Description |
|-----------|-------------|
| **Data Directory** | Directory containing unlabeled image tiles (.png, .tif, .tiff, .jpg, .jpeg, .raw). The dialog scans the directory and reports the number of images found with an auto-suggested epoch count. |
| **Dataset Info** | Auto-populated label showing image count and epoch recommendation after selecting a directory. |

### Output

| Parameter | Description |
|-----------|-------------|
| **Output Directory** | Where to save the pretrained encoder weights. Defaults to `{project}/mae_pretrained/`. The directory is created automatically if it does not exist. |

## Inference Parameters

### Classifier Selection

| Parameter | Description |
|-----------|-------------|
| **Classifier table** | Select a trained classifier. Channel count must match the image. |

### Output Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **Output Type** | MEASUREMENTS | MEASUREMENTS, OBJECTS, OVERLAY | How results are represented. |
| **Object Type** | DETECTION | DETECTION, ANNOTATION | QuPath object type (OBJECTS output only). |
| **Min Object Size** | 10 um^2 | 0-10000 | Discard objects below this area. |
| **Hole Filling** | 5 um^2 | 0-1000 | Fill interior holes below this area. |
| **Boundary Smoothing** | 1.0 um | 0-10 | Simplification tolerance in microns. |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Tile Size** | Auto | Auto-set from classifier. Must be divisible by 32. |
| **Tile Overlap (%)** | 12.5% | 0-50%. Higher = better blending, slower processing. |
| **Blend Mode** | LINEAR | LINEAR (recommended), GAUSSIAN (smoothest), NONE (fastest). |
| **Use GPU** | On | 10-50x faster than CPU. Falls back automatically. |
| **Test-Time Augmentation (TTA)** | Off | Applies D4 transforms (flips + 90-degree rotations) during inference and averages the predictions. Typically improves segmentation quality by 1-3% at the cost of ~8x slower inference. Best for final production runs where quality matters most. |

### Normalization

| Parameter | Description |
|-----------|-------------|
| **Image-level normalization** | Automatically enabled. Computes per-channel normalization statistics once across the entire image (sampling ~16 tiles in a 4x4 grid), then applies the same statistics to every tile. Eliminates tile boundary artifacts caused by per-tile normalization. |
| **Training dataset stats** | When available in the model metadata (models trained after this update), normalization uses statistics from the training dataset for the best consistency. Falls back to image-level sampling for older models. |

### Application Scope

| Parameter | Description |
|-----------|-------------|
| **Whole image** | Classify entire image without annotations. |
| **All annotations** | Classify within all annotations. |
| **Selected annotations** | Classify only within selected annotations. |
| **Create backup** | Save existing measurements before overwriting. |
