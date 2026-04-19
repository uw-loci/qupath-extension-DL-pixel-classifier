package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.warnings.InteractionWarning;
import qupath.ext.dlclassifier.service.warnings.TrainingWarning;

/**
 * Fires when the plateau scheduler is selected alongside
 * early-stopping metric "val_loss". Before the B.1 fix
 * (training_service.py _create_scheduler), plateau always got
 * {@code mode="max"} while val_loss decreases when improving, so
 * plateau aggressively reduced LR on every actual improvement. The
 * fix auto-derives mode from the metric; this watcher stays behind
 * as a regression tripwire in case a future scheduler refactor
 * drops that mapping.
 * <p>
 * Severity is WARN rather than BLOCKING: after B.1 lands, the
 * condition produces correct behaviour -- the popup is
 * informational ("this used to be broken; we derive the correct
 * mode automatically now"). A future deprecation can drop the
 * watcher once B.1 has been in production for a couple of
 * releases.
 */
public final class PlateauValLossWatcher implements TrainingWarning {

    public static final String ID = "plateau-val-loss-mode";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "Plateau scheduler with val_loss -- mode must be 'min'";
    }

    @Override
    public String getDescription() {
        return "ReduceLROnPlateau combined with early stopping on "
                + "val_loss requires mode='min' (loss decreases when "
                + "improving). The extension auto-derives this mode "
                + "from the ES metric, so this combination is safe. "
                + "You can suppress this notice; it exists so future "
                + "regressions in the scheduler-mode derivation are "
                + "caught at training time.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-4-plateau-val-loss";
    }

    @Override
    public Severity getSeverity() {
        return Severity.INFO;
    }

    @Override
    public boolean check(TrainingConfig config) {
        if (config == null) return false;
        return "plateau".equalsIgnoreCase(config.getSchedulerType())
                && "val_loss".equalsIgnoreCase(config.getEarlyStoppingMetric());
    }
}
