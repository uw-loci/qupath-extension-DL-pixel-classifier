/**
 * Batch Annotation Adjustment from DL Predictions
 *
 * WARNING: This script modifies annotations across MULTIPLE tiles at once.
 * It is intended for advanced users who have already reviewed individual tiles
 * using the Training Area Issues dialog and understand the model's behavior.
 *
 * The recommended workflow is:
 *   1. Train a model
 *   2. Run evaluation (Training Area Issues dialog)
 *   3. Review tiles one-by-one using the interactive dialog
 *   4. ONLY if the model's predictions are consistently reliable should you
 *      use this script for bulk adjustment
 *
 * What this script does:
 *   - Loads prediction and confidence maps from a saved evaluation session
 *   - For each tile above a loss threshold, adjusts annotations where the
 *     model's confidence exceeds the given threshold
 *   - Only modifies annotation geometry WITHIN tile boundaries
 *   - Annotations extending beyond tiles are preserved outside
 *
 * Prerequisites:
 *   - A Training Area Issues session must be saved (contains prediction,
 *     confidence, and ground truth maps)
 *   - The project and images must be open
 *
 * Safety:
 *   - SAVE YOUR PROJECT before running this script
 *   - There is no batch undo -- individual tile adjustments can be undone
 *     one at a time, but not the entire batch
 *   - Start with a HIGH confidence threshold (0.90+) and only lower it
 *     after verifying results on a few tiles first
 *
 * Usage:
 *   1. Save your QuPath project (File > Save)
 *   2. Update the configuration below
 *   3. Run this script from the Script Editor
 */

import qupath.ext.dlclassifier.service.AnnotationAdjuster
import qupath.ext.dlclassifier.service.ClassifierClient
import qupath.ext.dlclassifier.service.TrainingIssuesSessionStore
import qupath.ext.dlclassifier.model.ClassifierMetadata
import qupath.lib.gui.QuPathGUI
import qupath.lib.common.ColorTools

import java.nio.file.Path

// ============ Configuration ============

// Path to the model directory (contains model.pt and training_issues_sessions/)
def modelDirPath = "/path/to/your/model/directory"

// Confidence threshold: only accept model predictions at or above this level.
// RECOMMENDATION: Start at 0.90 and lower only after reviewing results.
//   0.95 = very conservative (only fix blatant errors)
//   0.90 = conservative (recommended starting point)
//   0.80 = moderate (same as dialog default)
//   0.70 = aggressive (changes many borders -- use with caution)
def confidenceThreshold = 0.90

// Minimum loss threshold: only adjust tiles with loss above this value.
// Tiles with low loss are already well-predicted and don't need correction.
def minLossThreshold = 1.0

// Which splits to adjust: "all", "train", or "val"
def splitFilter = "all"

// Dry run: if true, only reports what would change without modifying anything
def dryRun = true

// ============ End Configuration ============

def modelDir = Path.of(modelDirPath)
if (!modelDir.toFile().exists()) {
    println "ERROR: Model directory not found: ${modelDirPath}"
    println "Update modelDirPath in the script configuration."
    return
}

// Load classifier metadata
def metadataPath = modelDir.resolve("metadata.json")
ClassifierMetadata metadata = null
if (metadataPath.toFile().exists()) {
    try {
        metadata = ClassifierMetadata.loadFromPath(metadataPath)
    } catch (Exception e) {
        println "WARNING: Could not load classifier metadata: ${e.message}"
    }
}

if (metadata == null) {
    println "ERROR: No classifier metadata found at ${metadataPath}"
    println "Ensure the model directory contains metadata.json from training."
    return
}

// List available sessions
def sessions = TrainingIssuesSessionStore.listSessions(metadata, modelDir)
if (sessions.isEmpty()) {
    println "ERROR: No saved Training Area Issues sessions found."
    println "Run evaluation and save a session from the Training Area Issues dialog first."
    return
}

// Use the most recent session
def session = sessions.get(0)
println "Using session: ${session.sessionId()} (${session.tileCount()} tiles)"
if (session.stale()) {
    println "WARNING: Session is STALE (${session.stalenessReason()})"
    println "Results may not reflect the current model. Consider re-evaluating."
}

// Load the session
def loaded = TrainingIssuesSessionStore.load(session.dir())
def results = loaded.results()
def downsample = loaded.downsample()
def patchSize = loaded.patchSize()

// Extract class names and colors from metadata
def classNames = []
def classColors = [:]
if (metadata.getClasses() != null) {
    for (def c : metadata.getClasses()) {
        classNames.add(c.name())
        if (c.color() != null && !c.color().isEmpty()) {
            def hex = c.color().replace("#", "")
            try {
                classColors[c.name()] = Integer.parseInt(hex, 16) & 0xFFFFFF
            } catch (NumberFormatException ignored) {}
        }
    }
}

if (classNames.isEmpty()) {
    println "ERROR: No class information found in classifier metadata."
    return
}

println ""
println "=== Batch Annotation Adjustment ==="
println "Classes: ${classNames.join(', ')}"
println "Confidence threshold: ${String.format('%.0f%%', confidenceThreshold * 100)}"
println "Min loss threshold: ${minLossThreshold}"
println "Split filter: ${splitFilter}"
println "Dry run: ${dryRun}"
println ""

// Filter tiles by loss and split
def eligibleTiles = results.findAll { r ->
    if (r.loss() < minLossThreshold) return false
    if (splitFilter != "all" && r.split() != splitFilter) return false
    // Check that prediction data exists
    if (r.predictionMapPath() == null || r.confidenceMapPath() == null ||
        r.groundTruthMaskPath() == null) return false
    return true
}

println "Eligible tiles: ${eligibleTiles.size()} / ${results.size()} total"
println ""

if (eligibleTiles.isEmpty()) {
    println "No tiles match the filter criteria. Nothing to adjust."
    return
}

// Create the adjuster
def adjuster = new AnnotationAdjuster(downsample, patchSize, classNames)

// Get the viewer
def qupath = QuPathGUI.getInstance()
if (qupath == null) {
    println "ERROR: QuPath GUI not available. Run this from the QuPath Script Editor."
    return
}
def viewer = qupath.getViewer()
if (viewer == null || viewer.getImageData() == null) {
    println "ERROR: No image open in the viewer."
    return
}

// Process each eligible tile
int tilesProcessed = 0
int tilesSkipped = 0
int totalPixelsChanged = 0
int totalAnnotationsModified = 0
int totalAnnotationsAdded = 0

for (def tile : eligibleTiles) {
    def stem = tile.filename().replaceFirst("\\.(tiff?|raw)\$", "")
    print "  Tile ${stem} (loss=${String.format('%.3f', tile.loss())}, " +
          "split=${tile.split()}): "

    try {
        def preview = adjuster.computePreview(
            tile.predictionMapPath(),
            tile.confidenceMapPath(),
            tile.groundTruthMaskPath(),
            confidenceThreshold,
            classColors)

        if (preview.totalChangedPixels() == 0) {
            println "no changes needed"
            tilesSkipped++
            continue
        }

        def changeDesc = preview.changesPerClass().collect { k, v -> "${k}:${v}px" }.join(", ")
        print "${preview.totalChangedPixels()} pixels (${changeDesc})"

        if (dryRun) {
            println " [DRY RUN - not applied]"
        } else {
            def result = adjuster.applyAdjustment(viewer, tile.x(), tile.y(), preview.adjustedMask())
            println " -> ${result.summary()}"
            totalAnnotationsModified += result.annotationsModified()
            totalAnnotationsAdded += result.annotationsAdded()
        }

        totalPixelsChanged += preview.totalChangedPixels()
        tilesProcessed++

    } catch (Exception e) {
        println "ERROR: ${e.message}"
        tilesSkipped++
    }
}

println ""
println "=== Summary ==="
println "Tiles processed: ${tilesProcessed}"
println "Tiles skipped: ${tilesSkipped}"
println "Total pixels changed: ${totalPixelsChanged}"
if (!dryRun) {
    println "Annotations modified: ${totalAnnotationsModified}"
    println "Annotations added: ${totalAnnotationsAdded}"
    println ""
    println "IMPORTANT: Save your project to preserve these changes."
    println "If results are unexpected, close the project WITHOUT saving"
    println "and reopen to restore the original annotations."
} else {
    println ""
    println "This was a DRY RUN. No annotations were modified."
    println "To apply changes, set dryRun = false and re-run the script."
}
