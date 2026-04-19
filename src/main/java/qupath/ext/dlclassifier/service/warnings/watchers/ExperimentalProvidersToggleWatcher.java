package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.warnings.InteractionWarning;
import qupath.ext.dlclassifier.service.warnings.PreferenceWarning;

/**
 * Fires on TensorRT / INT8 preference toggle. The Python-side
 * inference service caches ONNX Runtime sessions per model, and
 * toggling provider flags ({@code _use_tensorrt}, {@code _use_int8})
 * does NOT automatically invalidate those cached sessions. Until
 * the E.4 fix lands, users who flip the preference mid-session
 * will see no change until they reopen the classifier.
 * <p>
 * This watcher pops once on toggle with a reminder to restart
 * inference (close + reopen the overlay, or use the "Reload all
 * models" action once E.4 ships).
 */
public final class ExperimentalProvidersToggleWatcher
        implements PreferenceWarning {

    public static final String ID = "experimental-providers-toggle";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "Experimental provider toggled -- reload models to apply";
    }

    @Override
    public String getDescription() {
        return "TensorRT / INT8 preferences affect which ONNX "
                + "Runtime execution provider is used at inference. "
                + "Models loaded before the toggle keep their "
                + "original provider until they are reloaded. Close "
                + "and reopen the overlay, or use 'Reload all "
                + "models' in preferences (once available), to make "
                + "the change take effect.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-7-experimental-providers-toggle";
    }

    @Override
    public Severity getSeverity() {
        return Severity.WARN;
    }

    @Override
    public boolean check() {
        // The watcher is invoked from a preference-change listener.
        // It fires unconditionally when invoked -- the listener
        // itself already knows a toggle just happened.
        return DLClassifierPreferences.isExperimentalTensorRT()
                || DLClassifierPreferences.isExperimentalInt8();
    }
}
