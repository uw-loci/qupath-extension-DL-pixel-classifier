package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.collections.transformation.SortedList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.projects.ProjectImageEntry;

import java.util.List;
import java.util.Map;

/**
 * Modeless dialog showing per-tile evaluation results from post-training analysis.
 * <p>
 * Displays tiles sorted by loss (descending) to help users identify annotation
 * errors, hard cases, and model failures. Double-clicking a row navigates the
 * QuPath viewer to the tile location.
 *
 * @author UW-LOCI
 * @since 0.3.0
 */
public class TrainingAreaIssuesDialog {

    private static final Logger logger = LoggerFactory.getLogger(TrainingAreaIssuesDialog.class);

    private final Stage stage;
    private final TableView<TileRow> table;
    private final ObservableList<TileRow> allRows;
    private final FilteredList<TileRow> filteredRows;
    private final Label summaryLabel;
    private final double downsample;

    /**
     * Creates the training area issues dialog.
     *
     * @param classifierName name of the classifier for the title
     * @param results        per-tile evaluation results sorted by loss descending
     * @param downsample     downsample factor used during training
     */
    public TrainingAreaIssuesDialog(String classifierName,
                                    List<ClassifierClient.TileEvaluationResult> results,
                                    double downsample) {
        this.downsample = downsample;
        this.stage = new Stage();
        stage.initStyle(StageStyle.DECORATED);
        stage.setTitle("Training Area Issues - " + classifierName);
        stage.setResizable(true);

        // Convert results to observable rows
        allRows = FXCollections.observableArrayList();
        for (var r : results) {
            allRows.add(new TileRow(r));
        }
        filteredRows = new FilteredList<>(allRows, row -> true);
        SortedList<TileRow> sortedRows = new SortedList<>(filteredRows);

        // Summary label
        long highLoss = results.stream().filter(r -> r.loss() > 1.0).count();
        summaryLabel = new Label(String.format(
                "%d tiles evaluated | %d with loss > 1.0", results.size(), highLoss));
        summaryLabel.setStyle("-fx-font-weight: bold;");

        // Filter controls
        ComboBox<String> splitFilter = new ComboBox<>();
        splitFilter.getItems().addAll("All", "Train", "Val");
        splitFilter.setValue("All");

        Slider thresholdSlider = new Slider(0, 10, 0);
        thresholdSlider.setShowTickLabels(true);
        thresholdSlider.setShowTickMarks(true);
        thresholdSlider.setMajorTickUnit(2);
        thresholdSlider.setMinorTickCount(1);
        thresholdSlider.setPrefWidth(200);

        Label thresholdLabel = new Label("Min Loss: 0.00");
        thresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            thresholdLabel.setText(String.format("Min Loss: %.2f", newVal.doubleValue()));
            updateFilter(splitFilter.getValue(), newVal.doubleValue());
        });

        splitFilter.setOnAction(e -> updateFilter(splitFilter.getValue(),
                thresholdSlider.getValue()));

        HBox filterBox = new HBox(10,
                new Label("Filter:"), splitFilter,
                thresholdLabel, thresholdSlider);
        filterBox.setAlignment(Pos.CENTER_LEFT);

        // Table
        table = new TableView<>();
        sortedRows.comparatorProperty().bind(table.comparatorProperty());
        table.setItems(sortedRows);

        TableColumn<TileRow, String> imageCol = new TableColumn<>("Image");
        imageCol.setCellValueFactory(new PropertyValueFactory<>("sourceImage"));
        imageCol.setPrefWidth(120);

        TableColumn<TileRow, String> splitCol = new TableColumn<>("Split");
        splitCol.setCellValueFactory(new PropertyValueFactory<>("split"));
        splitCol.setPrefWidth(50);

        TableColumn<TileRow, Double> lossCol = new TableColumn<>("Loss");
        lossCol.setCellValueFactory(new PropertyValueFactory<>("loss"));
        lossCol.setPrefWidth(70);
        lossCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));
        lossCol.setSortType(TableColumn.SortType.DESCENDING);

        TableColumn<TileRow, Double> disagreeCol = new TableColumn<>("Disagree%");
        disagreeCol.setCellValueFactory(new PropertyValueFactory<>("disagreementPct"));
        disagreeCol.setPrefWidth(80);
        disagreeCol.setCellFactory(col -> new FormattedDoubleCell<>("%5.1f%%", 100.0));

        TableColumn<TileRow, Double> iouCol = new TableColumn<>("mIoU");
        iouCol.setCellValueFactory(new PropertyValueFactory<>("meanIoU"));
        iouCol.setPrefWidth(65);
        iouCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));

        TableColumn<TileRow, String> classesCol = new TableColumn<>("Classes");
        classesCol.setCellValueFactory(new PropertyValueFactory<>("classesPresent"));
        classesCol.setPrefWidth(150);

        table.getColumns().addAll(List.of(imageCol, splitCol, lossCol, disagreeCol, iouCol, classesCol));
        table.getSortOrder().add(lossCol);

        // Double-click to navigate
        table.setRowFactory(tv -> {
            TableRow<TileRow> row = new TableRow<>();
            row.setOnMouseClicked(event -> {
                if (event.getClickCount() == 2 && !row.isEmpty()) {
                    navigateToTile(row.getItem());
                }
            });
            return row;
        });

        // Status bar
        Label statusLabel = new Label("Double-click a row to navigate to the tile location");
        statusLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

        // Layout
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.getChildren().addAll(summaryLabel, filterBox, table, statusLabel);
        VBox.setVgrow(table, Priority.ALWAYS);

        Scene scene = new Scene(root, 650, 500);
        stage.setScene(scene);
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        Platform.runLater(() -> stage.show());
    }

    private void updateFilter(String splitValue, double minLoss) {
        filteredRows.setPredicate(row -> {
            if (!"All".equals(splitValue)) {
                String expected = splitValue.toLowerCase();
                if (!row.getSplit().equalsIgnoreCase(expected)) {
                    return false;
                }
            }
            return row.getLoss() >= minLoss;
        });

        long visible = filteredRows.size();
        long highLoss = filteredRows.stream().filter(r -> r.getLoss() > 1.0).count();
        summaryLabel.setText(String.format(
                "%d tiles shown | %d with loss > 1.0", visible, highLoss));
    }

    private void navigateToTile(TileRow row) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;

        // Try to switch to the correct image if needed
        String targetImageId = row.getSourceImageId();
        String targetImageName = row.getSourceImage();
        var project = qupath.getProject();

        if (project != null && targetImageName != null && !targetImageName.isEmpty()) {
            // Check if we're already on the correct image
            var currentImageData = qupath.getImageData();
            String currentImageName = currentImageData != null
                    ? currentImageData.getServer().getMetadata().getName() : null;
            boolean needsSwitch = !targetImageName.equals(currentImageName);

            if (needsSwitch) {
                // Find the target entry
                for (var entry : project.getImageList()) {
                    boolean match = targetImageId != null && !targetImageId.isEmpty()
                            ? targetImageId.equals(entry.getID())
                            : targetImageName.equals(entry.getImageName());
                    if (match) {
                        Platform.runLater(() -> {
                            try {
                                qupath.openImageEntry(entry);
                                // Navigate after image loads
                                Platform.runLater(() -> centerViewerOnTile(qupath, row));
                            } catch (Exception e) {
                                logger.warn("Failed to open image: {}", e.getMessage());
                            }
                        });
                        return;
                    }
                }
                logger.warn("Could not find image '{}' in project", targetImageName);
            }
        }

        // Already on the right image, just navigate
        Platform.runLater(() -> centerViewerOnTile(qupath, row));
    }

    private void centerViewerOnTile(QuPathGUI qupath, TileRow row) {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) return;

        // Center on the tile (x, y are top-left corner in full-res coordinates)
        var imageData = viewer.getImageData();
        if (imageData == null) return;

        int patchSize = (int) (imageData.getServer().getMetadata().getPreferredTileWidth());
        if (patchSize <= 0) patchSize = 512;

        double regionSize = patchSize * downsample;
        double centerX = row.getX() + regionSize / 2.0;
        double centerY = row.getY() + regionSize / 2.0;

        viewer.setCenterPixelLocation(centerX, centerY);
        viewer.setDownsampleFactor(downsample);
    }

    /**
     * Table cell that formats doubles with a format string.
     */
    private static class FormattedDoubleCell<S> extends TableCell<S, Double> {
        private final String format;
        private final double multiplier;

        FormattedDoubleCell(String format) {
            this(format, 1.0);
        }

        FormattedDoubleCell(String format, double multiplier) {
            this.format = format;
            this.multiplier = multiplier;
        }

        @Override
        protected void updateItem(Double item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setText(null);
            } else {
                setText(String.format(format, item * multiplier));
            }
        }
    }

    /**
     * Row model for the evaluation results table.
     */
    public static class TileRow {
        private final StringProperty sourceImage;
        private final StringProperty sourceImageId;
        private final StringProperty split;
        private final DoubleProperty loss;
        private final DoubleProperty disagreementPct;
        private final DoubleProperty meanIoU;
        private final StringProperty classesPresent;
        private final IntegerProperty x;
        private final IntegerProperty y;
        private final StringProperty filename;

        public TileRow(ClassifierClient.TileEvaluationResult result) {
            this.sourceImage = new SimpleStringProperty(result.sourceImage());
            this.sourceImageId = new SimpleStringProperty(result.sourceImageId());
            this.split = new SimpleStringProperty(result.split());
            this.loss = new SimpleDoubleProperty(result.loss());
            this.disagreementPct = new SimpleDoubleProperty(result.disagreementPct());
            this.meanIoU = new SimpleDoubleProperty(result.meanIoU());
            this.x = new SimpleIntegerProperty(result.x());
            this.y = new SimpleIntegerProperty(result.y());
            this.filename = new SimpleStringProperty(result.filename());

            // Build classes present string from per-class IoU
            StringBuilder classes = new StringBuilder();
            if (result.perClassIoU() != null) {
                for (Map.Entry<String, Double> entry : result.perClassIoU().entrySet()) {
                    if (entry.getValue() != null) {
                        if (classes.length() > 0) classes.append(", ");
                        classes.append(entry.getKey());
                    }
                }
            }
            this.classesPresent = new SimpleStringProperty(classes.toString());
        }

        public String getSourceImage() { return sourceImage.get(); }
        public StringProperty sourceImageProperty() { return sourceImage; }

        public String getSourceImageId() { return sourceImageId.get(); }

        public String getSplit() { return split.get(); }
        public StringProperty splitProperty() { return split; }

        public double getLoss() { return loss.get(); }
        public DoubleProperty lossProperty() { return loss; }

        public double getDisagreementPct() { return disagreementPct.get(); }
        public DoubleProperty disagreementPctProperty() { return disagreementPct; }

        public double getMeanIoU() { return meanIoU.get(); }
        public DoubleProperty meanIoUProperty() { return meanIoU; }

        public String getClassesPresent() { return classesPresent.get(); }
        public StringProperty classesPresentProperty() { return classesPresent; }

        public int getX() { return x.get(); }
        public int getY() { return y.get(); }

        public String getFilename() { return filename.get(); }
    }
}
