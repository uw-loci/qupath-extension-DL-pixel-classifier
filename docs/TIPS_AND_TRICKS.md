# Tips and Tricks

Practical workflow tips for getting the best results from the DL Pixel Classifier.

## Iterative Training: Quick Runs to Fix Annotations

**Don't commit to a long training run before checking your annotations.**

1. **Run a short 5-10 epoch training** with your initial annotations
2. **Toggle the overlay** to see where the model makes mistakes
3. **Look for the largest errors** -- these usually indicate annotation problems:
   - A region classified as the wrong class may have been annotated incorrectly
   - Boundaries between classes may be drawn too loosely or tightly
   - An entire class may be underrepresented (too few annotations)
4. **Fix the annotations** in QuPath based on what you see
5. **Re-train** with the corrected annotations for a longer run (50-200 epochs)

This iterative approach saves hours compared to training for 200 epochs only to discover that an annotation was wrong. Even a few epochs is enough for the model to reveal gross labeling errors.

**Pro tip:** After the short run, use **Review Training Areas** to see which tiles the model classified correctly and which it struggled with. The worst-performing tiles often point directly to annotation issues.

## Hard Pixel % (OHEM): Large Homogeneous Regions

If your training images have **large uniform regions** (e.g., background glass, tissue interiors, empty space), most pixels in each batch are "easy" -- the model classifies them correctly very early in training. Without intervention, the model spends most of its training time on these easy pixels, gaining almost nothing.

**When to use Hard Pixel %:**
- Your annotations include large area annotations where most of the interior is homogeneous
- The model quickly reaches high accuracy but stops improving on difficult boundary regions
- You see the model correctly classifying tissue interiors but struggling at class boundaries

**Recommended workflow:**
1. Start training WITHOUT Hard Pixel % (100%, the default) for the first run
2. If the model plateaus on easy regions but boundary accuracy is poor, reduce to 25-50%
3. For very large homogeneous regions, try 5-10% to focus almost entirely on boundaries

**Why not always use it?** For small annotations or images with few easy pixels, OHEM can be too aggressive -- it may discard pixels the model genuinely needs to learn. If your annotations are primarily lines or narrow polygons along class boundaries, OHEM provides little benefit because most pixels are already "hard."

**Alternative: Focal Loss** is a softer approach that down-weights easy pixels rather than completely ignoring them. Try Focal Loss first if you're unsure; switch to OHEM if Focal Loss doesn't focus enough on the hard cases.

## Line Annotations vs. Area Annotations

**Line annotations along class boundaries are often more effective than large area fills:**

- Lines focus training on the pixels that matter most -- the boundaries between classes
- Area annotations overrepresent the easy interior pixels (see Hard Pixel % above)
- Lines are faster to draw and easier to correct
- The model learns "this is where class A meets class B" rather than "here are 50,000 pixels of class A interior"

**When to use area annotations:**
- The class has no clear boundary (e.g., "tissue vs. background")
- The interior texture varies and needs to be learned (e.g., heterogeneous tumor)
- You have very few annotations and need more training pixels

**When to use line annotations:**
- Classes are defined by their boundaries (e.g., "vein margin", "epithelial border")
- The interior of each class is relatively uniform
- You want to train faster with less annotation effort

You can mix both: use area annotations for background/tissue and line annotations for fine structures.

## Using the Overlay for Quality Control

The **Toggle Prediction Overlay** is your primary tool for evaluating classifier quality:

1. Train a model (even a short run)
2. Toggle the overlay on
3. Pan across the entire image, looking for:
   - **Systematic errors**: entire regions consistently misclassified = annotation problem
   - **Noisy boundaries**: ragged, speckled edges = model needs more epochs or smoother annotations
   - **Missing classes**: a class never appears in the overlay = check that it's in the training data
4. The overlay and Apply Classifier (OBJECTS) use the exact same inference pipeline, so what the overlay shows is what you'll get as objects

## Choosing Tile Size

| Tile Size | Best for | Trade-offs |
|-----------|----------|------------|
| 256 | Cell-level features, small structures | Fast training, more tiles needed for coverage |
| 512 | General purpose (recommended default) | Good balance of context and memory |
| 1024 | Large tissue structures, architecture patterns | Slower training, requires more VRAM |

**Rule of thumb:** the tile should be large enough that a human could classify the center pixel by looking at the tile. If you need to see more surrounding context to make the classification, increase the tile size.

## When Training Stalls

If validation loss or mIoU plateaus early:

1. **Check class balance**: use "Rebalance Classes" to auto-weight underrepresented classes
2. **Add more annotations**: especially for the worst-performing class
3. **Try a different backbone**: resnet50 has more capacity than resnet34
4. **Increase tile size**: the model may need more spatial context
5. **Enable Hard Pixel %**: if easy pixels are dominating the loss (see above)
6. **Lower learning rate**: if loss oscillates, try 1e-4 instead of 1e-3

## Sharing Models

To share a trained classifier with someone else:

1. Navigate to `{project}/classifiers/dl/{model_name}/`
2. Share **only** `model.pt` and `metadata.json`
3. The recipient places both files in a folder under their project's `classifiers/dl/` directory
4. Files named `best_in_progress_*.pt` and `checkpoint_*.pt` are training artifacts (5x larger) and are NOT needed for inference -- safe to delete

## Multi-Scale Context

Enable **Context Scale** (2x-16x) when classification depends on what surrounds a region, not just the region itself:

- **Tumor vs. stroma**: the tissue architecture around a cell cluster helps distinguish tumor from benign
- **Tissue type classification**: knowing you're in liver vs. kidney helps classify individual structures
- **Anatomical regions**: large-scale spatial patterns that a single tile can't capture

Context scale adds minimal memory overhead (~5-10%) by interleaving a downsampled wide-view with the detail tile. Both the overlay and Apply Classifier handle context tiles automatically.
