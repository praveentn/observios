# MLflow Tracking & Observability Enhancements

## Summary of Changes

I've significantly enhanced your Iris ML platform with comprehensive MLflow tracking and observability. Here's what's been added:

## 🎯 Key Improvements

### 1. **Separate Experiments for Training vs Inference**
- **Training Experiment**: `iris_classification` - Tracks all model training runs
- **Inference Experiment**: `iris_inference` - Tracks all prediction activities (single & batch)

### 2. **Enhanced Batch Prediction Tracking**
Previously, batch predictions had **NO** MLflow tracking. Now they log:
- ✅ Total samples processed
- ✅ Average, min, and max prediction confidence
- ✅ Prediction distribution across classes
- ✅ Percentage breakdown by class
- ✅ Complete predictions saved as CSV artifact
- ✅ Input file name and timestamp

### 3. **Improved Single Prediction Tracking**
Single predictions now log:
- ✅ All input features
- ✅ Predicted class name and index
- ✅ Prediction confidence score
- ✅ Probabilities for all classes
- ✅ Model path used
- ✅ Timestamp

### 4. **Enhanced Dashboard with Tabs**
The new HTML interface includes:
- **Three experiment tabs**: All Runs, Training Runs, Inference Runs
- **Summary statistics**: Total runs, training runs, inference runs, average accuracy
- **Visual run type indicators**: Color-coded badges for training vs inference
- **Hover effects**: Better UX with visual feedback
- **Detailed metrics display**: Shows relevant metrics based on run type

### 5. **Better Run Visualization**
- **Training runs** (green badge): Show accuracy, F1-score, samples
- **Inference runs** (blue badge): Show prediction type, confidence, sample counts
- **Batch statistics**: Display in result cards after predictions
- **Click to view details**: Popup with full run parameters and metrics

## 📊 What Gets Logged

### Training Runs
**Parameters:**
- model_type, n_estimators, test_size
- n_features, n_samples, n_classes
- data_source (sklearn_iris or uploaded_csv)

**Metrics:**
- accuracy, precision, recall, f1_score

**Artifacts:**
- Trained model (.pkl)
- Model metadata (.json)

### Batch Prediction Runs
**Parameters:**
- prediction_type: "batch"
- model_path, input_file, timestamp

**Metrics:**
- total_samples
- avg_confidence, min_confidence, max_confidence
- unique_classes_predicted
- count_[class_name] for each class
- percentage_[class_name] for each class

**Artifacts:**
- Complete predictions CSV with:
  - Original features
  - Predicted class
  - Confidence score
  - Probabilities for all classes

### Single Prediction Runs
**Parameters:**
- prediction_type: "single"
- model_path, timestamp
- sepal_length, sepal_width, petal_length, petal_width
- predicted_class (name)

**Metrics:**
- predicted_class_index
- prediction_confidence
- probability_[class_name] for each class

## 🚀 How to Use

### 1. Start the Application
```bash
python main.py
```

### 2. Train a Model
- Upload a CSV or use default Iris dataset
- Training metrics are automatically logged to `iris_classification` experiment

### 3. Make Predictions
- **Single**: Manual input → Logs to `iris_inference` experiment
- **Batch**: Upload CSV → Logs detailed statistics to `iris_inference` experiment

### 4. View MLflow Dashboard
In the web interface:
- Click tabs to switch between All/Training/Inference runs
- View summary statistics at the top
- Click any run to see full details
- Refresh button updates the view

### 5. Optional: MLflow UI
For the full MLflow experience:
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Then visit: http://localhost:5000

## 📁 File Structure

```
mlruns/
├── 0/                    # iris_classification experiment
│   ├── <run_id>/         # Training run artifacts
│   │   ├── metrics/
│   │   ├── params/
│   │   ├── artifacts/
│   │   │   ├── model.pkl
│   │   │   └── metadata.json
│   └── ...
├── 1/                    # iris_inference experiment
│   ├── <run_id>/         # Prediction run artifacts
│   │   ├── metrics/
│   │   ├── params/
│   │   ├── artifacts/
│   │   │   └── predictions.csv (batch only)
│   └── ...
└── models/
    ├── iris_model_*.pkl
    └── iris_model_*_metadata.json
```

## 🔍 Example Metrics View

### After Training:
- Accuracy: 96.67%
- Precision: 96.75%
- Recall: 96.67%
- F1-Score: 96.65%

### After Batch Prediction:
- Total Samples: 5
- Avg Confidence: 98.5%
- Min Confidence: 95.2%
- Max Confidence: 99.8%
- Distribution:
  - setosa: 2 (40%)
  - versicolor: 2 (40%)
  - virginica: 1 (20%)

## 🎨 Dashboard Features

### Visual Indicators:
- **Green left border**: Training runs
- **Blue left border**: Inference runs
- **Active tab**: Purple gradient background
- **Hover effect**: Slides right with border highlight

### Summary Stats:
- Real-time calculation from all runs
- Average model accuracy across training runs
- Count of each run type

## 🔧 Technical Details

### API Changes:
1. `/mlflow/runs` endpoint now accepts `experiment_name` parameter
2. Returns `experiment_name` and `run_name` for each run
3. Better error handling for missing experiments

### Frontend Changes:
1. Added experiment tab switching
2. Summary statistics calculation
3. Enhanced run display with type badges
4. Better responsive layout for full-width MLflow card

## 📈 Benefits

1. **Complete Observability**: Every action is now tracked
2. **Better Debugging**: See exactly what happened in each prediction
3. **Performance Monitoring**: Track model confidence over time
4. **Experiment Separation**: Clean separation of training vs serving
5. **Artifact Storage**: Keep prediction results for auditing
6. **Professional Dashboard**: Modern, intuitive UI

## 🎯 Next Steps

You can now:
- Compare different training runs to find best model
- Monitor prediction confidence trends
- Audit batch predictions via saved artifacts
- Track model performance degradation
- Export data for further analysis

All MLflow data persists in the `mlruns` directory and can be viewed in both the web interface and the native MLflow UI.
