package qupath.ext.dlclassifier.ui;

import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.layout.GridPane;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.OverlayService;

/**
 * Dialog for configuring overlay blend mode and tile overlap.
 * <p>
 * Changes are saved to preferences and, if an overlay is active,
 * the overlay is rebuilt immediately with the new settings.
 *
 * @author UW-LOCI
 * @since 0.3.3
 */
public class OverlaySettingsDialog {

    private static final Logger logger = LoggerFactory.getLogger(OverlaySettingsDialog.class);

    private final OverlayService overlayService;

    public OverlaySettingsDialog(OverlayService overlayService) {
        this.overlayService = overlayService;
    }

    /**
     * Shows the overlay settings dialog.
     */
    public void show() {
        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("Overlay Settings");
        dialog.setHeaderText("Configure prediction overlay blending");

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(15));

        int row = 0;

        // Blend Mode
        grid.add(new Label("Blend Mode:"), 0, row);
        ComboBox<InferenceConfig.BlendMode> blendModeCombo = new ComboBox<>();
        blendModeCombo.getItems().addAll(
                InferenceConfig.BlendMode.LINEAR,
                InferenceConfig.BlendMode.GAUSSIAN,
                InferenceConfig.BlendMode.CENTER_CROP,
                InferenceConfig.BlendMode.NONE);
        // Restore from preference
        InferenceConfig.BlendMode savedMode;
        try {
            savedMode = InferenceConfig.BlendMode.valueOf(DLClassifierPreferences.getLastBlendMode());
        } catch (IllegalArgumentException e) {
            savedMode = InferenceConfig.BlendMode.LINEAR;
        }
        blendModeCombo.setValue(savedMode);
        TooltipHelper.install(blendModeCombo,
                "LINEAR: smooth linear ramp at tile boundaries\n" +
                "GAUSSIAN: cosine bell blend (smoother, best for ViT models)\n" +
                "CENTER_CROP: only use center predictions (no artifacts, ~4x slower)\n" +
                "NONE: no blending, raw tile predictions");
        grid.add(blendModeCombo, 1, row);

        row++;

        // Tile Overlap %
        grid.add(new Label("Tile Overlap (%):"), 0, row);
        SpinnerValueFactory.DoubleSpinnerValueFactory overlapFactory =
                new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 50.0,
                        DLClassifierPreferences.getTileOverlapPercent(), 2.5);
        Spinner<Double> overlapSpinner = new Spinner<>(overlapFactory);
        overlapSpinner.setEditable(true);
        overlapSpinner.setPrefWidth(100);
        TooltipHelper.install(overlapSpinner,
                "Percentage of tile size used as overlap between adjacent tiles.\n" +
                "Higher overlap reduces edge artifacts but uses more memory.\n" +
                "12.5% is a good default. 25%+ provides best quality.");
        grid.add(overlapSpinner, 1, row);

        row++;

        // Warning label
        Label warningLabel = new Label();
        warningLabel.setWrapText(true);
        warningLabel.setMaxWidth(300);
        grid.add(warningLabel, 0, row, 2, 1);
        updateWarning(warningLabel, overlapSpinner.getValue());
        overlapSpinner.valueProperty().addListener((obs, oldVal, newVal) ->
                updateWarning(warningLabel, newVal));

        row++;

        // ViT note
        Label vitNote = new Label(
                "Note: ViT/MuViT models always use GAUSSIAN blending\n" +
                "regardless of the setting above.");
        vitNote.setStyle("-fx-font-size: 11px; -fx-text-fill: #666666;");
        vitNote.setWrapText(true);
        vitNote.setMaxWidth(300);
        grid.add(vitNote, 0, row, 2, 1);

        dialog.getDialogPane().setContent(grid);

        // Apply + Cancel buttons
        ButtonType applyType = new ButtonType("Apply", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(applyType, ButtonType.CANCEL);

        // Wire Apply button
        Button applyButton = (Button) dialog.getDialogPane().lookupButton(applyType);
        applyButton.setOnAction(event -> {
            InferenceConfig.BlendMode selectedMode = blendModeCombo.getValue();
            double selectedOverlap = overlapSpinner.getValue();

            // Save to preferences
            DLClassifierPreferences.setLastBlendMode(selectedMode.name());
            DLClassifierPreferences.setTileOverlapPercent(selectedOverlap);

            // Rebuild overlay if one is active
            if (overlayService.hasOverlay()) {
                boolean ok = overlayService.recreateOverlay(selectedMode, selectedOverlap);
                if (ok) {
                    logger.info("Overlay recreated: blend={}, overlap={}%",
                            selectedMode, selectedOverlap);
                } else {
                    logger.warn("Could not recreate overlay -- " +
                            "settings saved but will apply on next overlay creation");
                }
            }
        });

        dialog.showAndWait();
    }

    private void updateWarning(Label label, double overlapPercent) {
        if (overlapPercent == 0.0) {
            label.setText("WARNING: No overlap -- visible seams at tile boundaries");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #D32F2F;");
        } else if (overlapPercent < 10.0) {
            label.setText("Note: Low overlap may result in visible seams");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #F57C00;");
        } else if (overlapPercent >= 25.0) {
            label.setText("High overlap -- best quality, eliminates edge artifacts");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
        } else {
            label.setText("Good overlap for seamless blending");
            label.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
        }
    }
}
