package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.warnings.InteractionWarning;
import qupath.ext.dlclassifier.service.warnings.TrainingWarning;

/**
 * Fires when OHEM is active AND a focal-loss variant is selected.
 * Before the A.3 fix, the factory in training_service.py replaced
 * the focal loss with plain OHEM-CE when wrapping with OHEM,
 * silently discarding the user's focal_gamma. The A.3 fix adds
 * OHEMFocalLoss so focal modulation is preserved inside the hard-
 * pixel set; this watcher stays as a regression tripwire.
 * <p>
 * Severity is WARN until A.3 is confirmed landed in the running
 * Python env (the watcher is pre-training, Java-side -- it cannot
 * introspect the Python criterion object). After A.3 the
 * composition is correct; the popup can then be suppressed or
 * downgraded to INFO.
 */
public final class OhemFocalWatcher implements TrainingWarning {

    public static final String ID = "ohem-focal-gamma";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "OHEM with Focal loss -- focal_gamma preserved by "
                + "OHEMFocalLoss";
    }

    @Override
    public String getDescription() {
        return "OHEM hard-pixel mining combined with Focal loss is "
                + "handled by the OHEMFocalLoss composition so that "
                + "focal_gamma is applied BEFORE the top-K hard "
                + "selection. This is an informational notice to "
                + "confirm the composition is active; if your "
                + "training log shows 'OHEM active: ...' but does "
                + "NOT mention focal gamma, the fix has regressed.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-4-ohem-focal";
    }

    @Override
    public Severity getSeverity() {
        return Severity.INFO;
    }

    @Override
    public boolean check(TrainingConfig config) {
        if (config == null) return false;
        String lf = config.getLossFunction();
        boolean focal = "focal".equalsIgnoreCase(lf)
                || "focal_dice".equalsIgnoreCase(lf);
        return focal && config.getOhemHardRatio() < 1.0;
    }
}
