package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.lib.common.ColorTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Extracts annotated regions for deep learning training.
 * <p>
 * Supports both sparse annotations (lines, brushes) and dense annotations
 * (filled polygons). For sparse annotations, unlabeled pixels are marked with
 * an ignore index (255) so the training loss only computes on annotated pixels.
 *
 * <h3>Sparse Annotation Handling</h3>
 * In pixel classification, users typically draw thin lines or brush strokes
 * over different tissue types. This creates sparse labels where most pixels
 * in a training tile are unlabeled. The extractor:
 * <ul>
 *   <li>Renders line annotations with a configurable stroke width</li>
 *   <li>Marks unlabeled pixels as 255 (ignore_index)</li>
 *   <li>Combines all overlapping annotations from different classes into one mask</li>
 *   <li>Reports class pixel counts for weight balancing</li>
 * </ul>
 *
 * <h3>Export Format</h3>
 * <pre>
 * output_dir/
 *   train/
 *     images/
 *       patch_0000.tiff
 *     masks/
 *       patch_0000.png
 *   validation/
 *     images/
 *     masks/
 *   config.json
 * </pre>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class AnnotationExtractor {

    private static final Logger logger = LoggerFactory.getLogger(AnnotationExtractor.class);

    /**
     * Value used in masks for unlabeled pixels.
     * Training loss should use ignore_index=255 to skip these pixels.
     */
    public static final int UNLABELED_INDEX = 255;

    /**
     * Default stroke width for rendering line annotations (in pixels).
     */
    private static final int DEFAULT_LINE_STROKE_WIDTH = 5;

    private final ImageData<BufferedImage> imageData;
    private final ImageServer<BufferedImage> server;
    private final int patchSize;
    private final ChannelConfiguration channelConfig;
    private final int lineStrokeWidth;
    private final double downsample;
    private final int contextScale;

    /**
     * Creates a new annotation extractor.
     *
     * @param imageData     the image data
     * @param patchSize     the patch size to extract
     * @param channelConfig channel configuration
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig) {
        this(imageData, patchSize, channelConfig, DEFAULT_LINE_STROKE_WIDTH, 1.0);
    }

    /**
     * Creates a new annotation extractor with custom line stroke width.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth) {
        this(imageData, patchSize, channelConfig, lineStrokeWidth, 1.0);
    }

    /**
     * Creates a new annotation extractor with custom line stroke width and downsample.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract (output size in pixels)
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @param downsample      downsample factor (1.0 = full resolution, 4.0 = quarter resolution)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth,
                               double downsample) {
        this(imageData, patchSize, channelConfig, lineStrokeWidth, downsample, 1);
    }

    /**
     * Creates a new annotation extractor with multi-scale context support.
     *
     * @param imageData       the image data
     * @param patchSize       the patch size to extract (output size in pixels)
     * @param channelConfig   channel configuration
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @param downsample      downsample factor (1.0 = full resolution)
     * @param contextScale    context scale factor (1 = disabled, 2/4/8 = context tile extracted)
     */
    public AnnotationExtractor(ImageData<BufferedImage> imageData,
                               int patchSize,
                               ChannelConfiguration channelConfig,
                               int lineStrokeWidth,
                               double downsample,
                               int contextScale) {
        this.imageData = imageData;
        this.server = imageData.getServer();
        this.patchSize = patchSize;
        this.channelConfig = channelConfig;
        this.lineStrokeWidth = lineStrokeWidth;
        this.downsample = downsample;
        this.contextScale = contextScale;
    }

    /**
     * Exports training data from annotations.
     *
     * @param outputDir      output directory
     * @param classNames     list of class names to export
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @return export statistics including per-class pixel counts
     * @throws IOException if export fails
     */
    public ExportResult exportTrainingData(Path outputDir, List<String> classNames,
                                           double validationSplit) throws IOException {
        return exportTrainingData(outputDir, classNames, validationSplit, Collections.emptyMap());
    }

    /**
     * Exports training data from annotations with class weight multipliers.
     * <p>
     * This method handles both sparse (line/brush) and dense (polygon/area)
     * annotations. For sparse annotations, masks use 255 for unlabeled pixels.
     *
     * @param outputDir              output directory
     * @param classNames             list of class names to export
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @return export statistics including per-class pixel counts
     * @throws IOException if export fails
     */
    public ExportResult exportTrainingData(Path outputDir, List<String> classNames,
                                           double validationSplit,
                                           Map<String, Double> classWeightMultipliers) throws IOException {
        logger.info("Exporting training data to: {}", outputDir);

        // Create directories
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");

        Files.createDirectories(trainImages);
        Files.createDirectories(trainMasks);
        Files.createDirectories(valImages);
        Files.createDirectories(valMasks);

        // Context tile directories (only when multi-scale is enabled)
        Path trainContext = null;
        Path valContext = null;
        if (contextScale > 1) {
            trainContext = outputDir.resolve("train/context");
            valContext = outputDir.resolve("validation/context");
            Files.createDirectories(trainContext);
            Files.createDirectories(valContext);
            logger.info("Multi-scale context enabled: contextScale={}", contextScale);
        }

        // Build class index map (class 0, 1, 2, ...; 255 = unlabeled)
        Map<String, Integer> classIndex = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            classIndex.put(classNames.get(i), i);
        }

        // Collect all annotations with their class info and extract class colors
        List<AnnotationInfo> allAnnotations = new ArrayList<>();
        Map<String, String> classColorMap = new LinkedHashMap<>();
        for (PathObject annotation : imageData.getHierarchy().getAnnotationObjects()) {
            if (annotation.getPathClass() == null) continue;
            String className = annotation.getPathClass().getName();
            if (classIndex.containsKey(className)) {
                allAnnotations.add(new AnnotationInfo(
                        annotation,
                        annotation.getROI(),
                        classIndex.get(className),
                        isSparseROI(annotation.getROI())
                ));
                // Extract color from PathClass (first annotation of each class wins)
                if (!classColorMap.containsKey(className)) {
                    int color = annotation.getPathClass().getColor();
                    classColorMap.put(className, String.format("#%02X%02X%02X",
                            ColorTools.red(color), ColorTools.green(color), ColorTools.blue(color)));
                }
            }
        }

        logger.info("Found {} annotations across {} classes",
                allAnnotations.size(), classIndex.size());

        if (allAnnotations.isEmpty()) {
            throw new IOException("No annotations found for the specified classes");
        }

        // Determine patch locations based on annotation locations
        List<PatchLocation> patchLocations = generatePatchLocations(allAnnotations);
        logger.info("Generated {} candidate patch locations", patchLocations.size());

        // Log context-vs-image size for diagnostics
        int regionSize = (int) (patchSize * downsample);
        if (contextScale > 1) {
            int contextRegionSize = regionSize * contextScale;
            boolean imageFitsContext = server.getWidth() >= contextRegionSize
                    && server.getHeight() >= contextRegionSize;
            if (!imageFitsContext) {
                logger.warn("Image {}x{} is smaller than context region {}x{} ({}x scale). "
                                + "Context tiles will be resized from available area.",
                        server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize, contextScale);
            } else {
                logger.info("Context {}x scale: image {}x{} is large enough (context region {}x{}). "
                                + "Edge patches will use clamped (shifted) context.",
                        contextScale, server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize);
            }
        }

        // Phase 1: Collect all masks and determine class presence per patch
        List<PendingPatch> pendingPatches = new ArrayList<>();
        for (PatchLocation loc : patchLocations) {
            MaskResult maskResult = createCombinedMask(loc.x, loc.y, allAnnotations, classIndex.size());
            if (maskResult.labeledPixelCount == 0) continue;

            Set<Integer> presentClasses = new HashSet<>();
            for (int i = 0; i < classNames.size(); i++) {
                if (maskResult.classPixelCounts[i] > 0) presentClasses.add(i);
            }
            pendingPatches.add(new PendingPatch(loc, maskResult, presentClasses));
        }

        // Phase 2: Compute stratified train/validation split
        boolean[] isValidationArr = computeStratifiedSplit(pendingPatches, validationSplit, classNames.size());
        logSplitStatistics(pendingPatches, isValidationArr, classNames);

        // Phase 3: Read images and write files based on stratified assignment
        int patchIndex = 0;
        int trainCount = 0;
        int valCount = 0;
        long[] classPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;

        for (int p = 0; p < pendingPatches.size(); p++) {
            PendingPatch pp = pendingPatches.get(p);
            boolean isValidation = isValidationArr[p];

            RegionRequest request = RegionRequest.createInstance(
                    server.getPath(), downsample,
                    pp.location().x(), pp.location().y(), regionSize, regionSize,
                    0, 0);
            BufferedImage image = server.readRegion(request);

            Path imgDir = isValidation ? valImages : trainImages;
            Path maskDir = isValidation ? valMasks : trainMasks;

            savePatch(image, imgDir.resolve(String.format("patch_%04d.tiff", patchIndex)));
            saveMask(pp.maskResult().mask(), maskDir.resolve(String.format("patch_%04d.png", patchIndex)));

            if (contextScale > 1) {
                Path ctxDir = isValidation ? valContext : trainContext;
                Path ctxPath = ctxDir.resolve(String.format("patch_%04d.tiff", patchIndex));
                BufferedImage contextImage = readContextTile(
                        pp.location().x(), pp.location().y(), regionSize);
                savePatch(contextImage, ctxPath);
            }

            for (int i = 0; i < classNames.size(); i++) {
                classPixelCounts[i] += pp.maskResult().classPixelCounts()[i];
            }
            totalLabeledPixels += pp.maskResult().labeledPixelCount();

            if (isValidation) valCount++;
            else trainCount++;
            patchIndex++;
        }

        logger.info("Exported {} patches ({} train, {} validation)",
                patchIndex, trainCount, valCount);

        if (patchIndex == 0) {
            throw new IOException("No valid training patches could be extracted. "
                    + "This usually means the annotations are too small to produce "
                    + "any tiles at the current downsample level. "
                    + "Try: (1) using a lower downsample value, "
                    + "(2) making annotations larger, or "
                    + "(3) adding annotations to more images.");
        }
        if (trainCount == 0) {
            throw new IOException("All " + patchIndex + " exported patches were assigned "
                    + "to validation, leaving 0 for training. "
                    + "This happens when there are very few patches and the validation "
                    + "split requires at least one patch per class. "
                    + "Try: (1) reducing the downsample to generate more patches, "
                    + "(2) annotating more regions, or "
                    + "(3) reducing the validation split percentage.");
        }

        // Calculate class weights
        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), classPixelCounts[i]);
            logger.info("  Class '{}': {} labeled pixels", classNames.get(i), classPixelCounts[i]);
        }

        // Save configuration with class distribution, colors, and metadata
        saveConfig(outputDir, classNames, classPixelCounts, totalLabeledPixels,
                trainCount, valCount, allAnnotations.size(), classWeightMultipliers, classColorMap);

        return new ExportResult(patchIndex, trainCount, valCount,
                pixelCounts, totalLabeledPixels);
    }

    /**
     * Overload for backward compatibility.
     */
    public void exportTrainingData(Path outputDir, List<String> classNames) throws IOException {
        exportTrainingData(outputDir, classNames, 0.2);
    }

    /**
     * Exports training data with a patch numbering offset, for use in multi-image export.
     * <p>
     * Assumes output directories already exist. Does not write config.json (caller handles that).
     *
     * @param outputDir       output directory (with train/images, train/masks, etc.)
     * @param classNames      list of class names
     * @param validationSplit fraction for validation
     * @param startIndex      starting patch index for sequential numbering
     * @return export statistics for this image
     * @throws IOException if export fails
     */
    ExportResult exportTrainingDataWithOffset(Path outputDir, List<String> classNames,
                                              double validationSplit, int startIndex) throws IOException {
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");

        // Context tile directories
        Path trainContext = contextScale > 1 ? outputDir.resolve("train/context") : null;
        Path valContext = contextScale > 1 ? outputDir.resolve("validation/context") : null;

        // Build class index map
        Map<String, Integer> classIndex = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            classIndex.put(classNames.get(i), i);
        }

        // Collect annotations
        List<AnnotationInfo> allAnnotations = new ArrayList<>();
        for (PathObject annotation : imageData.getHierarchy().getAnnotationObjects()) {
            if (annotation.getPathClass() == null) continue;
            String className = annotation.getPathClass().getName();
            if (classIndex.containsKey(className)) {
                allAnnotations.add(new AnnotationInfo(
                        annotation, annotation.getROI(),
                        classIndex.get(className),
                        isSparseROI(annotation.getROI())
                ));
            }
        }

        if (allAnnotations.isEmpty()) {
            logger.info("No annotations found in this image, skipping");
            return new ExportResult(0, 0, 0, new LinkedHashMap<>(), 0);
        }

        List<PatchLocation> patchLocations = generatePatchLocations(allAnnotations);
        int regionSize = (int) (patchSize * downsample);

        // Log context-vs-image size for diagnostics
        if (contextScale > 1) {
            int contextRegionSize = regionSize * contextScale;
            if (server.getWidth() < contextRegionSize || server.getHeight() < contextRegionSize) {
                logger.warn("Image {}x{} is smaller than context region {}x{} ({}x scale). "
                                + "Context tiles will be resized from available area.",
                        server.getWidth(), server.getHeight(),
                        contextRegionSize, contextRegionSize, contextScale);
            }
        }

        // Phase 1: Collect all masks and determine class presence per patch
        List<PendingPatch> pendingPatches = new ArrayList<>();
        for (PatchLocation loc : patchLocations) {
            MaskResult maskResult = createCombinedMask(loc.x, loc.y, allAnnotations, classIndex.size());
            if (maskResult.labeledPixelCount == 0) continue;

            Set<Integer> presentClasses = new HashSet<>();
            for (int i = 0; i < classNames.size(); i++) {
                if (maskResult.classPixelCounts[i] > 0) presentClasses.add(i);
            }
            pendingPatches.add(new PendingPatch(loc, maskResult, presentClasses));
        }

        // Phase 2: Compute stratified train/validation split
        boolean[] isValidationArr = computeStratifiedSplit(pendingPatches, validationSplit, classNames.size());
        logSplitStatistics(pendingPatches, isValidationArr, classNames);

        // Phase 3: Read images and write files based on stratified assignment
        int patchIndex = startIndex;
        int trainCount = 0;
        int valCount = 0;
        long[] classPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;

        for (int p = 0; p < pendingPatches.size(); p++) {
            PendingPatch pp = pendingPatches.get(p);
            boolean isValidation = isValidationArr[p];

            RegionRequest request = RegionRequest.createInstance(
                    server.getPath(), downsample,
                    pp.location().x(), pp.location().y(), regionSize, regionSize, 0, 0);
            BufferedImage image = server.readRegion(request);

            Path imgDir = isValidation ? valImages : trainImages;
            Path maskDir = isValidation ? valMasks : trainMasks;

            savePatch(image, imgDir.resolve(String.format("patch_%04d.tiff", patchIndex)));
            saveMask(pp.maskResult().mask(), maskDir.resolve(String.format("patch_%04d.png", patchIndex)));

            if (contextScale > 1) {
                Path ctxDir = isValidation ? valContext : trainContext;
                Path ctxPath = ctxDir.resolve(String.format("patch_%04d.tiff", patchIndex));
                BufferedImage contextImage = readContextTile(
                        pp.location().x(), pp.location().y(), regionSize);
                savePatch(contextImage, ctxPath);
            }

            for (int i = 0; i < classNames.size(); i++) {
                classPixelCounts[i] += pp.maskResult().classPixelCounts()[i];
            }
            totalLabeledPixels += pp.maskResult().labeledPixelCount();

            if (isValidation) valCount++;
            else trainCount++;
            patchIndex++;
        }

        int exportedCount = patchIndex - startIndex;
        logger.info("Exported {} patches from this image ({} train, {} val)",
                exportedCount, trainCount, valCount);

        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), classPixelCounts[i]);
        }

        return new ExportResult(exportedCount, trainCount, valCount, pixelCounts, totalLabeledPixels);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries         project image entries to export from
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param classNames      list of class names to export
     * @param outputDir       output directory for combined training data
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, DEFAULT_LINE_STROKE_WIDTH, Collections.emptyMap(), 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries         project image entries to export from
     * @param patchSize       the patch size to extract
     * @param channelConfig   channel configuration
     * @param classNames      list of class names to export
     * @param outputDir       output directory for combined training data
     * @param validationSplit fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth stroke width for rendering line annotations (pixels)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, Collections.emptyMap(), 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, classWeightMultipliers, 1.0);
    }

    /**
     * Exports training data from multiple project images into a single training directory.
     * <p>
     * Each image's annotations are exported with sequential patch numbering across all images.
     * A combined config.json with aggregated class statistics is written at the end.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @param downsample             downsample factor (1.0 = full resolution)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers,
            double downsample) throws IOException {
        return exportFromProject(entries, patchSize, channelConfig, classNames,
                outputDir, validationSplit, lineStrokeWidth, classWeightMultipliers, downsample, 1);
    }

    /**
     * Exports training data from multiple project images with multi-scale context support.
     *
     * @param entries                project image entries to export from
     * @param patchSize              the patch size to extract
     * @param channelConfig          channel configuration
     * @param classNames             list of class names to export
     * @param outputDir              output directory for combined training data
     * @param validationSplit        fraction of data for validation (0.0-1.0)
     * @param lineStrokeWidth        stroke width for rendering line annotations (pixels)
     * @param classWeightMultipliers user multipliers on auto-computed weights (empty = no modification)
     * @param downsample             downsample factor (1.0 = full resolution)
     * @param contextScale           context scale factor (1 = disabled, 2/4/8 = context)
     * @return combined export statistics
     * @throws IOException if export fails
     */
    public static ExportResult exportFromProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            int patchSize,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path outputDir,
            double validationSplit,
            int lineStrokeWidth,
            Map<String, Double> classWeightMultipliers,
            double downsample,
            int contextScale) throws IOException {

        logger.info("Exporting training data from {} project images to: {}", entries.size(), outputDir);

        // Create shared directories
        Path trainImages = outputDir.resolve("train/images");
        Path trainMasks = outputDir.resolve("train/masks");
        Path valImages = outputDir.resolve("validation/images");
        Path valMasks = outputDir.resolve("validation/masks");
        Files.createDirectories(trainImages);
        Files.createDirectories(trainMasks);
        Files.createDirectories(valImages);
        Files.createDirectories(valMasks);

        // Context tile directories (only when multi-scale is enabled)
        if (contextScale > 1) {
            Files.createDirectories(outputDir.resolve("train/context"));
            Files.createDirectories(outputDir.resolve("validation/context"));
            logger.info("Multi-scale context enabled: contextScale={}", contextScale);
        }

        // Accumulators across all images
        int totalPatchIndex = 0;
        int totalTrainCount = 0;
        int totalValCount = 0;
        long[] totalClassPixelCounts = new long[classNames.size()];
        long totalLabeledPixels = 0;
        int totalAnnotationCount = 0;
        List<String> sourceImages = new ArrayList<>();

        for (ProjectImageEntry<BufferedImage> entry : entries) {
            logger.info("Processing image: {}", entry.getImageName());
            try {
                ImageData<BufferedImage> imageData = entry.readImageData();
                AnnotationExtractor extractor = new AnnotationExtractor(imageData, patchSize, channelConfig, lineStrokeWidth, downsample, contextScale);

                ExportResult result = extractor.exportTrainingDataWithOffset(
                        outputDir, classNames, validationSplit, totalPatchIndex);

                // Accumulate statistics
                totalPatchIndex += result.totalPatches();
                totalTrainCount += result.trainPatches();
                totalValCount += result.validationPatches();
                totalLabeledPixels += result.totalLabeledPixels();

                for (int i = 0; i < classNames.size(); i++) {
                    String className = classNames.get(i);
                    totalClassPixelCounts[i] += result.classPixelCounts().getOrDefault(className, 0L);
                }

                totalAnnotationCount += imageData.getHierarchy().getAnnotationObjects().stream()
                        .filter(a -> a.getPathClass() != null)
                        .count();
                sourceImages.add(entry.getImageName());

                imageData.getServer().close();
            } catch (Exception e) {
                logger.warn("Failed to export from image '{}': {}",
                        entry.getImageName(), e.getMessage());
            }
        }

        if (totalPatchIndex == 0) {
            throw new IOException("No valid training patches could be extracted from any image. "
                    + "This usually means the annotations are too small to produce "
                    + "any tiles at the current downsample level. "
                    + "Try: (1) using a lower downsample value, "
                    + "(2) making annotations larger, or "
                    + "(3) adding annotations to more images.");
        }
        if (totalTrainCount == 0) {
            throw new IOException("All " + totalPatchIndex + " exported patches were assigned "
                    + "to validation, leaving 0 for training. "
                    + "This happens when there are very few patches and the validation "
                    + "split requires at least one patch per class. "
                    + "Try: (1) reducing the downsample to generate more patches, "
                    + "(2) annotating more regions, or "
                    + "(3) reducing the validation split percentage.");
        }

        // Build combined pixel counts map
        Map<String, Long> pixelCounts = new LinkedHashMap<>();
        for (int i = 0; i < classNames.size(); i++) {
            pixelCounts.put(classNames.get(i), totalClassPixelCounts[i]);
            logger.info("  Class '{}': {} labeled pixels (combined)", classNames.get(i), totalClassPixelCounts[i]);
        }

        // Save combined config.json
        saveProjectConfig(outputDir, classNames, totalClassPixelCounts, totalLabeledPixels,
                channelConfig, patchSize, totalTrainCount, totalValCount,
                totalAnnotationCount, sourceImages, classWeightMultipliers, downsample, contextScale);

        logger.info("Multi-image export complete: {} patches ({} train, {} val) from {} images",
                totalPatchIndex, totalTrainCount, totalValCount, entries.size());

        return new ExportResult(totalPatchIndex, totalTrainCount, totalValCount,
                pixelCounts, totalLabeledPixels);
    }

    /**
     * Saves a combined config.json for multi-image project export.
     */
    private static void saveProjectConfig(Path outputDir, List<String> classNames,
                                           long[] classPixelCounts, long totalLabeledPixels,
                                           ChannelConfiguration channelConfig, int patchSize,
                                           int trainCount, int valCount,
                                           int annotationCount, List<String> sourceImages,
                                           Map<String, Double> classWeightMultipliers,
                                           double downsample, int contextScale)
            throws IOException {
        Path configPath = outputDir.resolve("config.json");
        List<String> channelNames = channelConfig.getChannelNames();
        String normStrategy = channelConfig.getNormalizationStrategy().name().toLowerCase();

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"patch_size\": ").append(patchSize).append(",\n");
        json.append("  \"downsample\": ").append(downsample).append(",\n");
        json.append("  \"unlabeled_index\": ").append(UNLABELED_INDEX).append(",\n");
        json.append("  \"total_labeled_pixels\": ").append(totalLabeledPixels).append(",\n");
        json.append("  \"classes\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            String color = getDefaultClassColor(i);
            json.append("    {\"index\": ").append(i)
                    .append(", \"name\": \"").append(classNames.get(i))
                    .append("\", \"color\": \"").append(color)
                    .append("\", \"pixel_count\": ").append(classPixelCounts[i]).append("}");
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"class_weights\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            double weight = classPixelCounts[i] > 0 ?
                    (double) totalLabeledPixels / (classNames.size() * classPixelCounts[i]) : 1.0;
            double multiplier = classWeightMultipliers.getOrDefault(classNames.get(i), 1.0);
            weight *= multiplier;
            json.append("    ").append(String.format("%.6f", weight));
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"channel_config\": {\n");
        int effectiveChannels = contextScale > 1
                ? channelConfig.getNumChannels() * 2
                : channelConfig.getNumChannels();
        json.append("    \"num_channels\": ").append(effectiveChannels).append(",\n");
        json.append("    \"detail_channels\": ").append(channelConfig.getNumChannels()).append(",\n");
        json.append("    \"context_scale\": ").append(contextScale).append(",\n");
        json.append("    \"channel_names\": [");
        for (int i = 0; i < channelNames.size(); i++) {
            json.append("\"").append(channelNames.get(i)).append("\"");
            if (i < channelNames.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"bit_depth\": ").append(channelConfig.getBitDepth()).append(",\n");
        json.append("    \"normalization\": {\n");
        json.append("      \"strategy\": \"").append(normStrategy).append("\",\n");
        json.append("      \"per_channel\": false,\n");
        json.append("      \"clip_percentile\": 99.0\n");
        json.append("    }\n");
        json.append("  },\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"source_images\": [");
        for (int i = 0; i < sourceImages.size(); i++) {
            json.append("\"").append(sourceImages.get(i).replace("\"", "\\\"")).append("\"");
            if (i < sourceImages.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"train_count\": ").append(trainCount).append(",\n");
        json.append("    \"validation_count\": ").append(valCount).append(",\n");
        json.append("    \"annotation_count\": ").append(annotationCount).append(",\n");
        json.append("    \"export_date\": \"").append(
                java.time.LocalDateTime.now().format(
                        java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\"\n");
        json.append("  }\n");
        json.append("}\n");

        Files.writeString(configPath, json.toString());
        logger.info("Saved combined config to: {}", configPath);
    }

    /**
     * Reads a context tile centered on the same location as a detail tile.
     * <p>
     * The context region covers {@code contextScale} times the area of the detail
     * region, read at {@code downsample * contextScale} so the output has the same
     * pixel dimensions (patchSize x patchSize) as the detail tile.
     *
     * @param detailX    top-left X of the detail tile region (full-res coords)
     * @param detailY    top-left Y of the detail tile region (full-res coords)
     * @param detailSize size of the detail region in full-res coords (patchSize * downsample)
     * @return context tile image (patchSize x patchSize pixels)
     * @throws IOException if reading fails
     */
    /**
     * Reads a context tile centered on the same location as a detail tile.
     * <p>
     * Uses a three-tier strategy:
     * <ol>
     *   <li><b>Ideal</b>: context region fits entirely -- read it directly.</li>
     *   <li><b>Clamped</b>: image is large enough but patch is near edge --
     *       shift context to the nearest valid position (slightly off-center but
     *       all real data, no padding).</li>
     *   <li><b>Resized</b>: image is smaller than the context region in at least
     *       one dimension -- read whatever fits and resize to patchSize.</li>
     * </ol>
     *
     * @param detailX    top-left X of the detail tile region (full-res coords)
     * @param detailY    top-left Y of the detail tile region (full-res coords)
     * @param detailSize size of the detail region in full-res coords (patchSize * downsample)
     * @return context tile image (patchSize x patchSize pixels), never null
     * @throws IOException if reading fails
     */
    private BufferedImage readContextTile(int detailX, int detailY, int detailSize) throws IOException {
        int contextRegionSize = detailSize * contextScale;
        int centerX = detailX + detailSize / 2;
        int centerY = detailY + detailSize / 2;

        int imgW = server.getWidth();
        int imgH = server.getHeight();

        int cx, cy, readW, readH;

        if (imgW >= contextRegionSize && imgH >= contextRegionSize) {
            // Image is large enough: clamp context position to fit (may shift off-center)
            cx = centerX - contextRegionSize / 2;
            cy = centerY - contextRegionSize / 2;
            cx = Math.max(0, Math.min(cx, imgW - contextRegionSize));
            cy = Math.max(0, Math.min(cy, imgH - contextRegionSize));
            readW = contextRegionSize;
            readH = contextRegionSize;
        } else {
            // Image smaller than context region: read the entire image
            cx = 0;
            cy = 0;
            readW = imgW;
            readH = imgH;
        }

        double contextDownsample = downsample * contextScale;
        RegionRequest contextRequest = RegionRequest.createInstance(
                server.getPath(), contextDownsample,
                cx, cy, readW, readH, 0, 0);
        BufferedImage contextImage = server.readRegion(contextRequest);

        // Resize to patchSize if the read region was smaller than expected
        if (contextImage.getWidth() != patchSize || contextImage.getHeight() != patchSize) {
            BufferedImage resized = new BufferedImage(patchSize, patchSize, contextImage.getType());
            java.awt.Graphics2D g = resized.createGraphics();
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(contextImage, 0, 0, patchSize, patchSize, null);
            g.dispose();
            contextImage = resized;
        }
        return contextImage;
    }

    /**
     * Determines if an ROI represents sparse annotation (line, polyline, etc.).
     */
    private boolean isSparseROI(ROI roi) {
        // Lines, polylines, and very thin shapes are sparse
        if (roi.isLine()) return true;

        // Check if the shape has very small area relative to its bounding box
        double bounds = roi.getBoundsWidth() * roi.getBoundsHeight();
        if (bounds > 0) {
            double areaRatio = roi.getArea() / bounds;
            // If area is less than 5% of bounding box, treat as sparse
            return areaRatio < 0.05;
        }
        return false;
    }

    /**
     * Generate patch locations based on annotation positions.
     * <p>
     * For sparse annotations (lines), we generate patches centered on points
     * along the line. For area annotations, we tile the bounding box.
     */
    private List<PatchLocation> generatePatchLocations(List<AnnotationInfo> annotations) {
        Set<String> locationKeys = new HashSet<>();
        List<PatchLocation> locations = new ArrayList<>();

        // Coverage per patch in full-res coordinates
        int coverage = (int) (patchSize * downsample);
        int step = coverage / 2; // 50% overlap between patches

        for (AnnotationInfo ann : annotations) {
            ROI roi = ann.roi;
            int x0 = (int) roi.getBoundsX();
            int y0 = (int) roi.getBoundsY();
            int w = (int) roi.getBoundsWidth();
            int h = (int) roi.getBoundsHeight();

            if (ann.isSparse) {
                // For sparse annotations, sample points along the ROI
                List<double[]> points = samplePointsAlongROI(roi);
                for (double[] pt : points) {
                    // Center patch on the sampled point
                    int px = (int) pt[0] - coverage / 2;
                    int py = (int) pt[1] - coverage / 2;

                    // Clip to image bounds
                    px = Math.max(0, Math.min(px, server.getWidth() - coverage));
                    py = Math.max(0, Math.min(py, server.getHeight() - coverage));

                    // Snap to grid to avoid too many overlapping patches
                    int snapStep = Math.max(1, step / 2);
                    px = (px / snapStep) * snapStep;
                    py = (py / snapStep) * snapStep;

                    String key = px + "," + py;
                    if (!locationKeys.contains(key)) {
                        locationKeys.add(key);
                        locations.add(new PatchLocation(px, py));
                    }
                }
            } else {
                // For area annotations, tile the bounding box
                for (int py = y0 - coverage / 4; py < y0 + h; py += step) {
                    for (int px = x0 - coverage / 4; px < x0 + w; px += step) {
                        int clippedX = Math.max(0, Math.min(px, server.getWidth() - coverage));
                        int clippedY = Math.max(0, Math.min(py, server.getHeight() - coverage));

                        String key = clippedX + "," + clippedY;
                        if (!locationKeys.contains(key)) {
                            locationKeys.add(key);
                            locations.add(new PatchLocation(clippedX, clippedY));
                        }
                    }
                }
            }
        }

        return locations;
    }

    /**
     * Sample points along a ROI for patch generation.
     * Works for lines, polylines, and thin shapes.
     */
    private List<double[]> samplePointsAlongROI(ROI roi) {
        List<double[]> points = new ArrayList<>();

        // Get all polygon points from the ROI
        List<qupath.lib.geom.Point2> roiPoints = roi.getAllPoints();

        if (roiPoints.isEmpty()) {
            // Fallback: use center of bounding box
            points.add(new double[]{
                    roi.getBoundsX() + roi.getBoundsWidth() / 2,
                    roi.getBoundsY() + roi.getBoundsHeight() / 2
            });
            return points;
        }

        // Sample at intervals along the ROI path (in full-res coords)
        double sampleInterval = patchSize * downsample / 4.0; // Sample every quarter-patch

        for (int i = 0; i < roiPoints.size() - 1; i++) {
            double x1 = roiPoints.get(i).getX();
            double y1 = roiPoints.get(i).getY();
            double x2 = roiPoints.get(i + 1).getX();
            double y2 = roiPoints.get(i + 1).getY();

            double segLength = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            int numSamples = Math.max(1, (int) (segLength / sampleInterval));

            for (int s = 0; s <= numSamples; s++) {
                double t = (double) s / numSamples;
                double x = x1 + t * (x2 - x1);
                double y = y1 + t * (y2 - y1);
                points.add(new double[]{x, y});
            }
        }

        // Also add the last point
        if (!roiPoints.isEmpty()) {
            qupath.lib.geom.Point2 last = roiPoints.get(roiPoints.size() - 1);
            points.add(new double[]{last.getX(), last.getY()});
        }

        return points;
    }

    /**
     * Creates a combined mask from all annotations overlapping a patch region.
     * <p>
     * Unlabeled pixels are set to 255 (UNLABELED_INDEX).
     * Labeled pixels are set to their class index (0, 1, 2, ...).
     */
    private MaskResult createCombinedMask(int offsetX, int offsetY,
                                          List<AnnotationInfo> annotations,
                                          int numClasses) {
        // The mask is always patchSize x patchSize (output resolution)
        BufferedImage mask = new BufferedImage(patchSize, patchSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = mask.createGraphics();

        // Fill with unlabeled value (255)
        g2d.setColor(new Color(UNLABELED_INDEX, UNLABELED_INDEX, UNLABELED_INDEX));
        g2d.fillRect(0, 0, patchSize, patchSize);

        // Set rendering hints for smooth lines
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        // Translate to patch coordinates
        AffineTransform originalTransform = g2d.getTransform();

        // Patch bounds in full-res coordinates
        int coverage = (int) (patchSize * downsample);
        Rectangle patchBounds = new Rectangle(offsetX, offsetY, coverage, coverage);

        for (AnnotationInfo ann : annotations) {
            ROI roi = ann.roi;

            // Quick check: does this annotation's bounding box overlap the patch?
            // Use scaled stroke width for bounding box expansion
            int expandedStroke = (int) Math.ceil(lineStrokeWidth * downsample);
            Rectangle annBounds = new Rectangle(
                    (int) roi.getBoundsX(), (int) roi.getBoundsY(),
                    (int) roi.getBoundsWidth() + expandedStroke,
                    (int) roi.getBoundsHeight() + expandedStroke
            );

            if (!patchBounds.intersects(annBounds)) continue;

            // Set class color
            int classIdx = ann.classIndex;
            g2d.setColor(new Color(classIdx, classIdx, classIdx));

            // Transform from full-res coords to mask pixel coords:
            // 1. Translate to patch origin in full-res space
            // 2. Scale down by downsample factor to get mask pixels
            g2d.setTransform(originalTransform);
            g2d.scale(1.0 / downsample, 1.0 / downsample);
            g2d.translate(-offsetX, -offsetY);

            Shape shape = roi.getShape();

            if (ann.isSparse || roi.isLine()) {
                // For sparse/line annotations: DRAW with stroke width
                // Scale stroke to maintain consistent visual width in mask space
                g2d.setStroke(new BasicStroke(
                        (float) (lineStrokeWidth * downsample),
                        BasicStroke.CAP_ROUND,
                        BasicStroke.JOIN_ROUND
                ));
                g2d.draw(shape);
            } else {
                // For area annotations: FILL the shape
                g2d.fill(shape);
            }
        }

        g2d.dispose();

        // Count pixels per class
        long[] classPixelCounts = new long[numClasses];
        long labeledPixelCount = 0;

        int[] pixels = new int[patchSize * patchSize];
        mask.getRaster().getPixels(0, 0, patchSize, patchSize, pixels);

        for (int pixel : pixels) {
            if (pixel != UNLABELED_INDEX && pixel < numClasses) {
                classPixelCounts[pixel]++;
                labeledPixelCount++;
            }
        }

        return new MaskResult(mask, classPixelCounts, labeledPixelCount);
    }

    /**
     * Saves a patch image. Uses TIFF for simple 8-bit images (<=4 bands)
     * and a raw float32 format for multi-channel or high-bit-depth images.
     */
    private void savePatch(BufferedImage image, Path path) throws IOException {
        int numBands = image.getRaster().getNumBands();
        int dataType = image.getRaster().getDataBuffer().getDataType();
        if (numBands <= 4 && dataType == DataBuffer.TYPE_BYTE) {
            ImageIO.write(image, "TIFF", path.toFile());
        } else {
            // N-channel or high-bit-depth: write as raw float32 with header
            Path rawPath = path.resolveSibling(
                    path.getFileName().toString()
                            .replaceFirst("\\.(tiff?|png)$", ".raw"));
            writeRawFloat(BitDepthConverter.toFloatArray(image), rawPath);
        }
    }

    /**
     * Writes float data as a raw binary file with a 12-byte header.
     * <p>
     * Format: 3 x int32 (height, width, channels) followed by
     * H*W*C float32 values in HWC order, all little-endian.
     *
     * @param data    float array [height][width][channels]
     * @param outPath output file path
     * @throws IOException if writing fails
     */
    static void writeRawFloat(float[][][] data, Path outPath) throws IOException {
        int h = data.length;
        int w = data[0].length;
        int c = data[0][0].length;
        ByteBuffer header = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(h);
        header.putInt(w);
        header.putInt(c);
        ByteBuffer body = ByteBuffer.allocate(h * w * c * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    body.putFloat(data[y][x][ch]);
                }
            }
        }
        try (OutputStream os = Files.newOutputStream(outPath)) {
            os.write(header.array());
            os.write(body.array());
        }
    }

    /**
     * Saves a mask image.
     */
    private void saveMask(BufferedImage mask, Path path) throws IOException {
        ImageIO.write(mask, "PNG", path.toFile());
    }

    /**
     * Saves configuration files including class distribution for weight balancing.
     */
    private void saveConfig(Path outputDir, List<String> classNames,
                            long[] classPixelCounts, long totalLabeledPixels) throws IOException {
        saveConfig(outputDir, classNames, classPixelCounts, totalLabeledPixels, 0, 0, 0,
                Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * Saves configuration files including class distribution, channel info, and metadata.
     *
     * @param outputDir              output directory
     * @param classNames             list of class names
     * @param classPixelCounts       per-class pixel counts
     * @param totalLabeledPixels     total labeled pixel count
     * @param trainCount             number of training patches
     * @param valCount               number of validation patches
     * @param annotationCount        number of annotations processed
     * @param classWeightMultipliers user-supplied multipliers on auto-computed weights (empty = no modification)
     * @param classColors            map of class name to hex color string (e.g. "#FF0000"), or empty
     */
    private void saveConfig(Path outputDir, List<String> classNames,
                            long[] classPixelCounts, long totalLabeledPixels,
                            int trainCount, int valCount, int annotationCount,
                            Map<String, Double> classWeightMultipliers,
                            Map<String, String> classColors) throws IOException {
        Path configPath = outputDir.resolve("config.json");

        List<String> channelNames = channelConfig.getChannelNames();
        String normStrategy = channelConfig.getNormalizationStrategy().name().toLowerCase();

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"patch_size\": ").append(patchSize).append(",\n");
        json.append("  \"downsample\": ").append(downsample).append(",\n");
        json.append("  \"unlabeled_index\": ").append(UNLABELED_INDEX).append(",\n");
        json.append("  \"line_stroke_width\": ").append(lineStrokeWidth).append(",\n");
        json.append("  \"total_labeled_pixels\": ").append(totalLabeledPixels).append(",\n");
        json.append("  \"classes\": [\n");
        for (int i = 0; i < classNames.size(); i++) {
            String color = classColors.getOrDefault(classNames.get(i), getDefaultClassColor(i));
            json.append("    {\"index\": ").append(i)
                    .append(", \"name\": \"").append(classNames.get(i))
                    .append("\", \"color\": \"").append(color)
                    .append("\", \"pixel_count\": ").append(classPixelCounts[i]).append("}");
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"class_weights\": [\n");
        // Calculate inverse frequency weights, then apply user multipliers
        for (int i = 0; i < classNames.size(); i++) {
            double weight = classPixelCounts[i] > 0 ?
                    (double) totalLabeledPixels / (classNames.size() * classPixelCounts[i]) : 1.0;
            double multiplier = classWeightMultipliers.getOrDefault(classNames.get(i), 1.0);
            weight *= multiplier;
            json.append("    ").append(String.format("%.6f", weight));
            if (i < classNames.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        json.append("  \"channel_config\": {\n");
        int effectiveChannels = contextScale > 1
                ? channelConfig.getNumChannels() * 2
                : channelConfig.getNumChannels();
        json.append("    \"num_channels\": ").append(effectiveChannels).append(",\n");
        json.append("    \"detail_channels\": ").append(channelConfig.getNumChannels()).append(",\n");
        json.append("    \"context_scale\": ").append(contextScale).append(",\n");
        json.append("    \"channel_names\": [");
        for (int i = 0; i < channelNames.size(); i++) {
            json.append("\"").append(channelNames.get(i)).append("\"");
            if (i < channelNames.size() - 1) json.append(", ");
        }
        json.append("],\n");
        json.append("    \"bit_depth\": ").append(channelConfig.getBitDepth()).append(",\n");
        json.append("    \"normalization\": {\n");
        json.append("      \"strategy\": \"").append(normStrategy).append("\",\n");
        json.append("      \"per_channel\": false,\n");
        json.append("      \"clip_percentile\": 99.0\n");
        json.append("    }\n");
        json.append("  },\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"source_image\": \"").append(escapeJson(server.getMetadata().getName())).append("\",\n");
        json.append("    \"train_count\": ").append(trainCount).append(",\n");
        json.append("    \"validation_count\": ").append(valCount).append(",\n");
        json.append("    \"annotation_count\": ").append(annotationCount).append(",\n");
        json.append("    \"export_date\": \"").append(
                java.time.LocalDateTime.now().format(
                        java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\"\n");
        json.append("  }\n");
        json.append("}\n");

        Files.writeString(configPath, json.toString());
        logger.info("Saved config to: {}", configPath);
    }

    /**
     * Returns a distinct default color for a class index.
     * Used when class colors are not available from annotations.
     */
    private static String getDefaultClassColor(int classIndex) {
        String[] palette = {
                "#FF0000", "#00AA00", "#0000FF", "#FFFF00",
                "#FF00FF", "#00FFFF", "#FF8800", "#8800FF"
        };
        return palette[classIndex % palette.length];
    }

    /**
     * Escapes special characters for JSON string values.
     */
    private static String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t");
    }

    // ==================== Data Classes ====================

    /**
     * Information about an annotation for processing.
     */
    private record AnnotationInfo(PathObject annotation, ROI roi, int classIndex, boolean isSparse) {}

    /**
     * A candidate patch location.
     */
    private record PatchLocation(int x, int y) {}

    /**
     * Result of creating a combined mask.
     */
    private record MaskResult(BufferedImage mask, long[] classPixelCounts, long labeledPixelCount) {}

    /**
     * A patch awaiting train/val assignment during the collection phase.
     * Holds the mask and class presence info but not the image (read later during write phase).
     */
    private record PendingPatch(
            PatchLocation location,
            MaskResult maskResult,
            Set<Integer> presentClasses
    ) {}

    /**
     * Computes a stratified train/validation split that guarantees every class present
     * in the dataset is represented in the validation set.
     * <p>
     * Algorithm:
     * <ol>
     *   <li>Build an inverted index: classIndex -> list of patch indices containing that class</li>
     *   <li>Sort classes by ascending frequency (rarest first)</li>
     *   <li>Guarantee phase: for each class not yet in validation, pick the patch covering
     *       the most still-unrepresented classes (greedy set-cover) and assign to validation</li>
     *   <li>Fill phase: if more validation slots remain, fill randomly from unassigned patches</li>
     * </ol>
     * If the guarantee phase assigns more patches than the target count, all are kept
     * (class coverage takes priority over exact split ratio).
     *
     * @param patches         collected patches with class presence info
     * @param validationSplit fraction of patches for validation (0.0-1.0)
     * @param numClasses      total number of classes
     * @return boolean array where true = validation patch
     */
    private static boolean[] computeStratifiedSplit(List<PendingPatch> patches,
                                                     double validationSplit,
                                                     int numClasses) {
        int total = patches.size();
        boolean[] isValidation = new boolean[total];

        if (validationSplit <= 0.0 || total == 0) {
            return isValidation; // all false
        }

        int targetValCount = Math.max(1, (int) Math.round(total * validationSplit));

        // Build inverted index: classIndex -> list of patch indices containing that class
        Map<Integer, List<Integer>> classToPatchIndices = new HashMap<>();
        for (int i = 0; i < total; i++) {
            for (int classIdx : patches.get(i).presentClasses()) {
                classToPatchIndices.computeIfAbsent(classIdx, k -> new ArrayList<>()).add(i);
            }
        }

        // Sort classes by ascending frequency (rarest first)
        List<Integer> classesByFrequency = new ArrayList<>(classToPatchIndices.keySet());
        classesByFrequency.sort(Comparator.comparingInt(c -> classToPatchIndices.get(c).size()));

        // Guarantee phase: ensure every class has at least one validation patch
        Set<Integer> coveredClasses = new HashSet<>();
        int valAssigned = 0;

        for (int classIdx : classesByFrequency) {
            if (coveredClasses.contains(classIdx)) continue;

            // Find the unassigned patch that covers the most still-uncovered classes
            int bestPatch = -1;
            int bestCoverage = 0;

            for (int patchIdx : classToPatchIndices.get(classIdx)) {
                if (isValidation[patchIdx]) continue; // already assigned

                int coverage = 0;
                for (int c : patches.get(patchIdx).presentClasses()) {
                    if (!coveredClasses.contains(c)) coverage++;
                }
                if (coverage > bestCoverage) {
                    bestCoverage = coverage;
                    bestPatch = patchIdx;
                }
            }

            if (bestPatch >= 0) {
                isValidation[bestPatch] = true;
                valAssigned++;
                coveredClasses.addAll(patches.get(bestPatch).presentClasses());
            }
        }

        // Fill phase: if more validation slots remain, fill randomly from unassigned
        if (valAssigned < targetValCount) {
            List<Integer> unassigned = new ArrayList<>();
            for (int i = 0; i < total; i++) {
                if (!isValidation[i]) unassigned.add(i);
            }
            Collections.shuffle(unassigned, new Random(42));

            int remaining = targetValCount - valAssigned;
            for (int i = 0; i < remaining && i < unassigned.size(); i++) {
                isValidation[unassigned.get(i)] = true;
            }
        }

        return isValidation;
    }

    /**
     * Logs per-class patch distribution across train and validation sets.
     * Warns if any class has zero patches in either set.
     *
     * @param patches      collected patches with class presence info
     * @param isValidation boolean array from computeStratifiedSplit
     * @param classNames   ordered list of class names
     */
    private static void logSplitStatistics(List<PendingPatch> patches,
                                            boolean[] isValidation,
                                            List<String> classNames) {
        int numClasses = classNames.size();
        int[] trainCounts = new int[numClasses];
        int[] valCounts = new int[numClasses];

        for (int i = 0; i < patches.size(); i++) {
            for (int classIdx : patches.get(i).presentClasses()) {
                if (classIdx < numClasses) {
                    if (isValidation[i]) valCounts[classIdx]++;
                    else trainCounts[classIdx]++;
                }
            }
        }

        int totalTrain = 0, totalVal = 0;
        for (int i = 0; i < patches.size(); i++) {
            if (isValidation[i]) totalVal++;
            else totalTrain++;
        }

        logger.info("Stratified split: {} train, {} validation patches", totalTrain, totalVal);
        for (int i = 0; i < numClasses; i++) {
            logger.info("  Class '{}': {} train patches, {} val patches",
                    classNames.get(i), trainCounts[i], valCounts[i]);
            if (valCounts[i] == 0) {
                logger.warn("  WARNING: Class '{}' has ZERO validation patches - "
                        + "validation metrics for this class will be unreliable", classNames.get(i));
            }
            if (trainCounts[i] == 0) {
                logger.warn("  WARNING: Class '{}' has ZERO training patches - "
                        + "model cannot learn this class", classNames.get(i));
            }
        }
    }

    /**
     * Result of the export operation, including class distribution statistics.
     */
    public record ExportResult(
            int totalPatches,
            int trainPatches,
            int validationPatches,
            Map<String, Long> classPixelCounts,
            long totalLabeledPixels
    ) {
        /**
         * Calculate inverse-frequency class weights for balanced training.
         *
         * @return map of class name to weight
         */
        public Map<String, Double> calculateClassWeights() {
            Map<String, Double> weights = new LinkedHashMap<>();
            int numClasses = classPixelCounts.size();

            for (Map.Entry<String, Long> entry : classPixelCounts.entrySet()) {
                double weight = entry.getValue() > 0 ?
                        (double) totalLabeledPixels / (numClasses * entry.getValue()) : 1.0;
                weights.put(entry.getKey(), weight);
            }
            return weights;
        }
    }
}
