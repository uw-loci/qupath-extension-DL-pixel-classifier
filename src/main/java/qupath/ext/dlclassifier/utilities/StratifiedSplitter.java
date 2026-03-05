package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Computes stratified train/validation splits for patch-based training data.
 * <p>
 * Guarantees every class present in the dataset is represented in the
 * validation set using a greedy set-cover algorithm, then fills remaining
 * validation slots randomly.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class StratifiedSplitter {

    private static final Logger logger = LoggerFactory.getLogger(StratifiedSplitter.class);

    private StratifiedSplitter() {
        // Static utility class
    }

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
     * @param classPresenceSets per-patch set of class indices present in that patch
     * @param validationSplit   fraction of patches for validation (0.0-1.0)
     * @param numClasses        total number of classes
     * @return boolean array where true = validation patch
     */
    public static boolean[] computeStratifiedSplit(List<Set<Integer>> classPresenceSets,
                                                    double validationSplit,
                                                    int numClasses) {
        int total = classPresenceSets.size();
        boolean[] isValidation = new boolean[total];

        if (validationSplit <= 0.0 || total == 0) {
            return isValidation; // all false
        }

        // Ensure at least 1 patch remains for training
        int targetValCount = Math.min(
                Math.max(1, (int) Math.round(total * validationSplit)),
                total - 1);

        // Build inverted index: classIndex -> list of patch indices containing that class
        Map<Integer, List<Integer>> classToPatchIndices = new HashMap<>();
        for (int i = 0; i < total; i++) {
            for (int classIdx : classPresenceSets.get(i)) {
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

            // Stop if assigning another validation patch would leave zero for training
            if (valAssigned >= total - 1) {
                logger.warn("Cannot guarantee class {} in validation: "
                        + "would leave 0 training patches", classIdx);
                break;
            }

            // Find the unassigned patch that covers the most still-uncovered classes
            int bestPatch = -1;
            int bestCoverage = 0;

            for (int patchIdx : classToPatchIndices.get(classIdx)) {
                if (isValidation[patchIdx]) continue; // already assigned

                int coverage = 0;
                for (int c : classPresenceSets.get(patchIdx)) {
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
                coveredClasses.addAll(classPresenceSets.get(bestPatch));
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
     * @param classPresenceSets per-patch set of class indices present in that patch
     * @param isValidation      boolean array from computeStratifiedSplit
     * @param classNames        ordered list of class names
     */
    public static void logSplitStatistics(List<Set<Integer>> classPresenceSets,
                                           boolean[] isValidation,
                                           List<String> classNames) {
        int numClasses = classNames.size();
        int[] trainCounts = new int[numClasses];
        int[] valCounts = new int[numClasses];

        for (int i = 0; i < classPresenceSets.size(); i++) {
            for (int classIdx : classPresenceSets.get(i)) {
                if (classIdx < numClasses) {
                    if (isValidation[i]) valCounts[classIdx]++;
                    else trainCounts[classIdx]++;
                }
            }
        }

        int totalTrain = 0, totalVal = 0;
        for (int i = 0; i < classPresenceSets.size(); i++) {
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
}
