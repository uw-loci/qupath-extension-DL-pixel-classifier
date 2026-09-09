package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Pins the {@code nvidia-smi -L} parsing that decides which compute environment
 * the setup wizard recommends.
 * <p>
 * A false positive is the expensive direction: pixi validates the {@code __cuda}
 * virtual package on every install, so recommending GPU on a machine without one
 * produces an environment that cannot install at all. Every ambiguous input must
 * therefore resolve to "no GPU".
 */
class GpuProbeTest {

    @Test
    void parsesASingleAdapterAndDropsTheUuid() {
        GpuProbe.Result result = GpuProbe.parse(0, "GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-1a2b3c4d-5e6f)\n");

        assertThat(result.nvidiaPresent()).isTrue();
        assertThat(result.gpuNames()).containsExactly("NVIDIA RTX 6000 Ada Generation");
        assertThat(result.summary()).isEqualTo("Detected NVIDIA GPU: NVIDIA RTX 6000 Ada Generation");
    }

    @Test
    void parsesMultipleAdapters() {
        GpuProbe.Result result = GpuProbe.parse(
                0,
                "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-aaa)\r\n"
                        + "GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-bbb)\r\n");

        assertThat(result.nvidiaPresent()).isTrue();
        assertThat(result.gpuNames()).hasSize(2);
        assertThat(result.summary()).startsWith("Detected 2 NVIDIA GPUs:");
    }

    @Test
    void nonZeroExitMeansNoGpuEvenIfSomethingWasPrinted() {
        // nvidia-smi prints a diagnostic and exits non-zero when the driver is
        // installed but no device is usable.
        GpuProbe.Result result = GpuProbe.parse(9, "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-ccc)");

        assertThat(result.nvidiaPresent()).isFalse();
        assertThat(result.gpuNames()).isEmpty();
    }

    @Test
    void emptyOrUnrecognisedOutputMeansNoGpu() {
        assertThat(GpuProbe.parse(0, "").nvidiaPresent()).isFalse();
        assertThat(GpuProbe.parse(0, "   \n\n").nvidiaPresent()).isFalse();
        assertThat(GpuProbe.parse(0, null).nvidiaPresent()).isFalse();
        assertThat(GpuProbe.parse(0, "No devices were found\n").nvidiaPresent()).isFalse();
    }

    @Test
    void summaryIsHonestWhenNothingWasFound() {
        assertThat(GpuProbe.parse(0, "").summary()).isEqualTo("No NVIDIA GPU detected on this machine.");
    }

    @Test
    void detectNeverThrowsOnThisMachineWhateverIsInstalled() {
        // The probe shells out; the wizard must survive a missing binary, a
        // timeout, or a driver that errors, on any developer or CI machine.
        assertThat(GpuProbe.detect()).isNotNull();
    }
}
