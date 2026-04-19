package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.warnings.InteractionWarning;
import qupath.ext.dlclassifier.service.warnings.TrainingWarning;

/**
 * Fires when tile overlap is greater than zero AND no image has an
 * explicit TRAIN_ONLY or VAL_ONLY role. In that case the stratified
 * split operates on overlapping patches generated from the same
 * region, so pixels that appear in a validation tile can also
 * appear (shifted) in a training tile -- classic pixel-level
 * leakage across the train/val boundary. Severity is BLOCKING
 * because the resulting validation metrics are untrustworthy.
 * <p>
 * Implements row "Tile overlap > 0 + Stratified validation split"
 * in section 6 of the interaction table
 * (claude-reports/2026-04-19_option-interaction-table.md).
 */
public final class TileOverlapSplitWatcher implements TrainingWarning {

    public static final String ID = "overlap-split-leakage";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "Tile overlap + auto split would leak pixels "
                + "across train/val";
    }

    @Override
    public String getDescription() {
        return "Tile overlap is greater than zero, but no image has "
                + "been assigned a Train-only or Val-only role. "
                + "The stratified split operates on per-tile indices "
                + "AFTER overlapping patches are generated, so a "
                + "training tile can overlap a validation tile from "
                + "the same region -- validation metrics become "
                + "artificially high. Fix: set overlap to 0, or "
                + "assign at least one image an explicit Train-only "
                + "or Val-only split role in the dialog's image list.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-6-tile-overlap-stratified-split";
    }

    @Override
    public Severity getSeverity() {
        return Severity.BLOCKING;
    }

    @Override
    public boolean check(TrainingConfig config) {
        if (config == null) return false;
        return config.getOverlap() > 0
                && !config.isHasPerImageSplitRoles();
    }
}
