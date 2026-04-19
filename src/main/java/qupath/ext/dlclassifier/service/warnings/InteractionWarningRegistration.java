package qupath.ext.dlclassifier.service.warnings;

import qupath.ext.dlclassifier.service.warnings.watchers.BrnFoldConvergenceWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.ChannelsLastBrnWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.ExperimentalProvidersToggleWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.InMemoryCacheWorkersWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.OhemFocalWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.PlateauValLossWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.TileOverlapSplitWatcher;

/**
 * Static startup hook that registers every
 * {@link InteractionWarning} shipped with the extension.
 * <p>
 * Called from {@code SetupDLClassifier.installExtension}. Add new
 * watchers here as they are authored; the list stays short because
 * each watcher is a one-line register call and the watchers
 * themselves own their titles, descriptions, and check logic.
 * <p>
 * When removing a deprecated watcher, keep its id reserved by
 * listing it in a source-level comment so future additions do not
 * accidentally reuse the id (which would also reuse the stored
 * "Don't show again" preference).
 */
public final class InteractionWarningRegistration {

    private InteractionWarningRegistration() {}

    public static void registerAll() {
        // Training-scope watchers.
        InteractionWarningService.register(new OhemFocalWatcher());
        InteractionWarningService.register(new InMemoryCacheWorkersWatcher());
        InteractionWarningService.register(new TileOverlapSplitWatcher());
        InteractionWarningService.register(new PlateauValLossWatcher());

        // Inference-scope watchers.
        InteractionWarningService.register(new ChannelsLastBrnWatcher());

        // Preference-toggle watchers.
        InteractionWarningService.register(new BrnFoldConvergenceWatcher());
        InteractionWarningService.register(new ExperimentalProvidersToggleWatcher());
    }
}
