package qupath.ext.dlclassifier.service;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Detects an NVIDIA GPU <em>without</em> a Python environment.
 * <p>
 * This has to work before anything is installed, which rules out the usual
 * check: {@link ApposeService#getGpuType()} reports
 * {@code torch.cuda.is_available()} from inside the installed environment, so
 * it answers "cpu" on a CPU build no matter what hardware is present, and it
 * cannot answer at all before the first install. The setup wizard needs the
 * answer at exactly that moment, to steer the user to the GPU variant.
 * <p>
 * The probe shells out to {@code nvidia-smi -L}, which ships with the NVIDIA
 * driver and is on PATH on both Windows and Linux when a driver is installed.
 * <p>
 * <b>It fails safe.</b> A missing binary, a non-zero exit, a timeout, or any
 * exception all report "no GPU detected", because the consequence of a false
 * positive is severe and the consequence of a false negative is not: pixi
 * validates the {@code __cuda} virtual package on every install, so choosing
 * the GPU environment on a machine without an NVIDIA GPU produces an
 * environment that cannot install at all, whereas choosing CPU on a machine
 * that has a GPU merely runs slowly until the user switches.
 */
public final class GpuProbe {

    private static final Logger logger = LoggerFactory.getLogger(GpuProbe.class);

    /** Generous enough for a cold driver load, short enough not to stall the wizard. */
    private static final long TIMEOUT_SECONDS = 10;

    private static volatile Result cached;

    private GpuProbe() {}

    /**
     * The outcome of a probe.
     *
     * @param nvidiaPresent whether at least one NVIDIA GPU was found
     * @param gpuNames      the detected adapter names, empty when none were found
     */
    public record Result(boolean nvidiaPresent, List<String> gpuNames) {

        /** A one-line human summary for the setup wizard. */
        public String summary() {
            if (!nvidiaPresent) {
                return "No NVIDIA GPU detected on this machine.";
            }
            if (gpuNames.size() == 1) {
                return "Detected NVIDIA GPU: " + gpuNames.get(0);
            }
            return "Detected " + gpuNames.size() + " NVIDIA GPUs: " + String.join(", ", gpuNames);
        }
    }

    /**
     * The cached result, or null when no probe has completed yet.
     * <p>
     * Callers on the JavaFX thread must use this rather than {@link #detect()}:
     * the probe shells out to a binary that can hang on a broken driver, and
     * blocking the FX thread on it freezes QuPath.
     *
     * @return the cached result, or null if not probed yet
     */
    public static Result cachedResult() {
        return cached;
    }

    /** Probes once per session and caches the answer. Blocks; never call on the FX thread. */
    public static Result detect() {
        Result local = cached;
        if (local == null) {
            synchronized (GpuProbe.class) {
                local = cached;
                if (local == null) {
                    local = runProbe();
                    cached = local;
                    logger.info("GPU probe: {}", local.summary());
                }
            }
        }
        return local;
    }

    private static Result runProbe() {
        Process process = null;
        try {
            process = new ProcessBuilder("nvidia-smi", "-L")
                    .redirectErrorStream(true)
                    .start();
            // Decode explicitly. The platform default is cp1252 on the
            // Windows machines this ships to, which makes the parse
            // vary by platform for no benefit -- adapter names are
            // ASCII, so UTF-8 is exact here and stable everywhere.
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                logger.debug("nvidia-smi timed out after {}s; assuming no GPU", TIMEOUT_SECONDS);
                return new Result(false, List.of());
            }
            return parse(process.exitValue(), output);
        } catch (Exception e) {
            // An absent binary lands here (IOException) and is the common case on
            // a machine with no NVIDIA driver -- debug, not warn.
            logger.debug("nvidia-smi unavailable ({}); assuming no GPU", e.toString());
            if (process != null) {
                process.destroyForcibly();
            }
            return new Result(false, List.of());
        }
    }

    /**
     * Parses {@code nvidia-smi -L} output. Package-private for testing.
     * <p>
     * Each adapter is one line of the form
     * {@code GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-abc...)}.
     *
     * @param exitCode the process exit code; anything non-zero means no GPU
     * @param output   the combined stdout/stderr text
     * @return the parsed result, never null
     */
    static Result parse(int exitCode, String output) {
        if (exitCode != 0 || output == null || output.isBlank()) {
            return new Result(false, List.of());
        }
        List<String> names = new ArrayList<>();
        for (String line : output.split("\\R")) {
            String trimmed = line.strip();
            if (!trimmed.startsWith("GPU ")) {
                continue;
            }
            int colon = trimmed.indexOf(':');
            if (colon < 0) {
                continue;
            }
            String name = trimmed.substring(colon + 1).strip();
            // Drop the trailing "(UUID: ...)" so the name reads cleanly in the UI.
            int uuid = name.lastIndexOf("(UUID:");
            if (uuid > 0) {
                name = name.substring(0, uuid).strip();
            }
            if (!name.isEmpty()) {
                names.add(name);
            }
        }
        return new Result(!names.isEmpty(), List.copyOf(names));
    }
}
