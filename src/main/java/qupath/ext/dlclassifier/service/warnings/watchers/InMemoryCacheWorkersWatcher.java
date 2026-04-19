package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.warnings.InteractionWarning;
import qupath.ext.dlclassifier.service.warnings.TrainingWarning;

/**
 * Fires when the in-memory dataset cache is requested AND more
 * than zero DataLoader workers are configured. On Windows/Appose
 * the worker processes are spawned, which pickles the dataset
 * (including the cache list) and copies it into every worker --
 * RAM usage scales linearly with worker count.
 * <p>
 * The D.1 fix auto-downgrades workers to 0 whenever the cache is
 * active; this watcher surfaces the downgrade to the user so they
 * see it in the GUI instead of only in the log.
 */
public final class InMemoryCacheWorkersWatcher implements TrainingWarning {

    public static final String ID = "cache-workers-downgrade";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "In-memory cache active: DataLoader workers "
                + "will be set to 0";
    }

    @Override
    public String getDescription() {
        return "With the in-memory dataset cache enabled, setting "
                + "data loader workers > 0 would cause the cache to "
                + "be copied into every worker process on Windows, "
                + "multiplying RAM usage. Training will continue "
                + "with num_workers=0 (log message "
                + "'forcing data_loader_workers=0 because in-memory "
                + "cache is active' in the Python log). If you need "
                + "multiple workers, disable the cache first.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-6-cache-workers";
    }

    @Override
    public Severity getSeverity() {
        return Severity.INFO;
    }

    @Override
    public boolean check(TrainingConfig config) {
        if (config == null) return false;
        String cache = config.getInMemoryDataset();
        if (cache == null || "off".equalsIgnoreCase(cache)) {
            return false;
        }
        return config.getDataLoaderWorkers() > 0;
    }
}
