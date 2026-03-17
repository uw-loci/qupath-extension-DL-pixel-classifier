package qupath.ext.dlclassifier.ui;

import javafx.scene.control.Control;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Tooltip;
import javafx.util.Duration;

/**
 * Centralized tooltip creation utility for the DL Pixel Classifier extension.
 * <p>
 * Provides consistent styling (500ms show delay, 30s duration, 400px max width,
 * text wrapping) across all dialogs. Optionally appends a "Learn more" URL.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class TooltipHelper {

    /** Default delay before tooltip appears (milliseconds). */
    private static final double SHOW_DELAY_MS = 500;

    /** Default duration the tooltip remains visible (seconds). */
    private static final double SHOW_DURATION_SEC = 30;

    /** Default maximum width for tooltip text wrapping (pixels). */
    private static final double MAX_WIDTH = 400;

    private TooltipHelper() {
        // Utility class - no instantiation
    }

    /**
     * Creates a styled tooltip with consistent settings.
     *
     * @param text the tooltip text
     * @return a styled Tooltip instance
     */
    public static Tooltip create(String text) {
        Tooltip tooltip = new Tooltip(text);
        applyStyle(tooltip);
        return tooltip;
    }

    /**
     * Creates a styled tooltip with a "Learn more" URL appended.
     *
     * @param text the tooltip text
     * @param url  the URL to append as a "Learn more" link
     * @return a styled Tooltip instance with link
     */
    public static Tooltip createWithLink(String text, String url) {
        Tooltip tooltip = new Tooltip(text + "\n\nLearn more: " + url);
        applyStyle(tooltip);
        return tooltip;
    }

    /**
     * Sets a styled tooltip on a JavaFX control.
     *
     * @param control the control to receive the tooltip
     * @param text    the tooltip text
     */
    public static void install(Control control, String text) {
        control.setTooltip(create(text));
    }

    /**
     * Sets the same styled tooltip on multiple JavaFX controls (e.g. a label
     * and its associated field so hovering either shows the tooltip).
     *
     * @param text     the tooltip text
     * @param controls the controls to receive the tooltip
     */
    public static void install(String text, Control... controls) {
        Tooltip tooltip = create(text);
        for (Control c : controls) {
            c.setTooltip(tooltip);
        }
    }

    /**
     * Sets a styled tooltip with a "Learn more" URL on a JavaFX control.
     *
     * @param control the control to receive the tooltip
     * @param text    the tooltip text
     * @param url     the URL to append as a "Learn more" link
     */
    public static void installWithLink(Control control, String text, String url) {
        control.setTooltip(createWithLink(text, url));
    }

    /**
     * Sets the same styled tooltip with a "Learn more" URL on multiple controls.
     *
     * @param text     the tooltip text
     * @param url      the URL to append
     * @param controls the controls to receive the tooltip
     */
    public static void installWithLink(String text, String url, Control... controls) {
        Tooltip tooltip = createWithLink(text, url);
        for (Control c : controls) {
            c.setTooltip(tooltip);
        }
    }

    /**
     * Installs a styled tooltip on a MenuItem using the JavaFX popup listener pattern.
     * <p>
     * MenuItems do not directly support tooltips, so this method attaches the tooltip
     * to the menu item's styleable node when the parent popup is shown.
     *
     * @param menuItem    the menu item to receive the tooltip
     * @param tooltipText the tooltip text
     */
    public static void installOnMenuItem(MenuItem menuItem, String tooltipText) {
        Tooltip tooltip = create(tooltipText);

        menuItem.parentPopupProperty().addListener((obs, oldPopup, newPopup) -> {
            if (newPopup != null) {
                newPopup.setOnShown(e -> {
                    var node = menuItem.getStyleableNode();
                    if (node != null) {
                        Tooltip.install(node, tooltip);
                    }
                });
            }
        });
    }

    /**
     * Installs a styled tooltip with a "Learn more" URL on a MenuItem.
     *
     * @param menuItem    the menu item to receive the tooltip
     * @param tooltipText the tooltip text
     * @param url         the URL to append as a "Learn more" link
     */
    public static void installOnMenuItemWithLink(MenuItem menuItem, String tooltipText, String url) {
        Tooltip tooltip = createWithLink(tooltipText, url);

        menuItem.parentPopupProperty().addListener((obs, oldPopup, newPopup) -> {
            if (newPopup != null) {
                newPopup.setOnShown(e -> {
                    var node = menuItem.getStyleableNode();
                    if (node != null) {
                        Tooltip.install(node, tooltip);
                    }
                });
            }
        });
    }

    /**
     * Applies consistent styling to a tooltip.
     */
    private static void applyStyle(Tooltip tooltip) {
        tooltip.setShowDelay(Duration.millis(SHOW_DELAY_MS));
        tooltip.setShowDuration(Duration.seconds(SHOW_DURATION_SEC));
        tooltip.setWrapText(true);
        tooltip.setMaxWidth(MAX_WIDTH);
    }
}
