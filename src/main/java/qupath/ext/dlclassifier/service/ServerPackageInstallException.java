package qupath.ext.dlclassifier.service;

import java.io.IOException;

/**
 * Thrown when the pixi environment built successfully but installing the
 * {@code dlclassifier-server} Python package into it failed.
 * <p>
 * This is deliberately distinct from an environment build failure. The server
 * package is fetched from a GitHub tag matching the extension version and is
 * identical for both compute variants, so a failure here says nothing about
 * whether the CPU or GPU environment was the right choice -- retrying under the
 * other variant cannot help, and would silently demote a GPU user to CPU for an
 * unrelated reason.
 * <p>
 * The common cause is a version whose release tag does not exist yet, which is
 * expected while testing a version-bumped build before its release is published.
 */
public class ServerPackageInstallException extends IOException {

    private static final long serialVersionUID = 1L;

    public ServerPackageInstallException(String message) {
        super(message);
    }
}
