# Iris ML Model Platform - Enhanced Edition

FastAPI web app with **comprehensive MLflow tracking and observability** for Iris flower classification.

![Version](https://img.shields.io/badge/version-2.0-blue)
![MLflow](https://img.shields.io/badge/mlflow-enabled-green)
![Status](https://img.shields.io/badge/status-production-success)

---

## 🆕 What's New in v2.0

### ✨ Major Enhancements:
- 🎯 **Separate Experiments**: Training runs vs Inference runs
- 📊 **Batch Prediction Tracking**: Complete observability for batch jobs
- 📈 **Enhanced Dashboard**: Tabbed interface with real-time statistics
- 💾 **Artifact Storage**: Automatic saving of prediction results
- 🎨 **Visual Indicators**: Color-coded run types and badges
- 📉 **Confidence Metrics**: Track prediction confidence over time
- 🔍 **Distribution Analysis**: See prediction distribution across classes

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Open Browser
```
http://localhost:8230
```

---

## ✨ Features

### Core ML Operations
- ✅ Train model with default Iris dataset or custom CSV
- ✅ Save and load models from filesystem
- ✅ Single predictions with probability distributions
- ✅ Batch predictions with statistics
- ✅ Model version management

### MLflow Integration
- ✅ **Two separate experiments**:
  - `iris_classification` - Training runs
  - `iris_inference` - Prediction runs
- ✅ **Comprehensive logging**:
  - All hyperparameters
  - Training metrics (accuracy, precision, recall, F1)
  - Prediction metrics (confidence, distributions)
  - Input features and outputs
- ✅ **Artifact management**:
  - Trained models
  - Model metadata
  - Batch prediction results (CSV)
- ✅ **Run tracking**:
  - Unique run IDs
  - Timestamps
  - Status monitoring

### Dashboard Features
- ✅ **Tabbed Interface**:
  - All Runs view
  - Training Runs only
  - Inference Runs only
- ✅ **Summary Statistics**:
  - Total run count
  - Average model accuracy
  - Run type breakdown
- ✅ **Visual Elements**:
  - Color-coded badges (green for training, blue for inference)
  - Hover effects for better UX
  - Click to view full details
- ✅ **Real-time Updates**:
  - Refresh button
  - Auto-load on page load
  - Dynamic metric calculation

---

## 📊 What Gets Tracked

### Training Runs (`iris_classification`)

**Parameters Logged:**
```python
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 100,
  "test_size": 0.2,
  "n_features": 4,
  "n_samples": 150,
  "n_classes": 3,
  "data_source": "sklearn_iris"
}
```

**Metrics Logged:**
```python
{
  "accuracy": 0.9667,
  "precision": 0.9675,
  "recall": 0.9667,
  "f1_score": 0.9665
}
```

**Artifacts Saved:**
- `model.pkl` - Trained model
- `metadata.json` - Model metadata

---

### Single Prediction Runs (`iris_inference`)

**Parameters Logged:**
```python
{
  "prediction_type": "single",
  "model_path": "./models/iris_model_20241121.pkl",
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2,
  "predicted_class": "setosa",
  "timestamp": "2024-11-21T10:30:00"
}
```

**Metrics Logged:**
```python
{
  "predicted_class_index": 0,
  "prediction_confidence": 0.997,
  "probability_setosa": 0.997,
  "probability_versicolor": 0.002,
  "probability_virginica": 0.001
}
```

---

### Batch Prediction Runs (`iris_inference`)

**Parameters Logged:**
```python
{
  "prediction_type": "batch",
  "model_path": "./models/iris_model_20241121.pkl",
  "input_file": "test_data.csv",
  "timestamp": "2024-11-21T10:32:00"
}
```

**Metrics Logged:**
```python
{
  "total_samples": 100,
  "avg_confidence": 0.982,
  "min_confidence": 0.891,
  "max_confidence": 0.998,
  "unique_classes_predicted": 3,
  "count_setosa": 35,
  "count_versicolor": 32,
  "count_virginica": 33,
  "percentage_setosa": 35.0,
  "percentage_versicolor": 32.0,
  "percentage_virginica": 33.0
}
```

**Artifacts Saved:**
- `predictions.csv` - Complete results with all probabilities

---

## 🎯 API Endpoints

### Model Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Train new model |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch predictions |
| `/models` | GET | List saved models |
| `/load_model` | POST | Load specific model |

### MLflow Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mlflow/experiments` | GET | List all experiments |
| `/mlflow/runs` | GET | Get runs (supports filtering) |
| `/mlflow/run/{run_id}` | GET | Get run details |

---

## 📁 Project Structure

```
iris-ml-platform/
├── main.py                 # FastAPI backend with MLflow
├── index.html              # Enhanced dashboard UI
├── requirements.txt        # Python dependencies
├── models/                 # Saved models directory
│   ├── iris_model_*.pkl
│   └── iris_model_*_metadata.json
└── mlruns/                 # MLflow tracking data
    ├── 0/                  # iris_classification experiment
    │   └── <run_id>/
    │       ├── metrics/
    │       ├── params/
    │       └── artifacts/
    └── 1/                  # iris_inference experiment
        └── <run_id>/
            ├── metrics/
            ├── params/
            └── artifacts/
                └── predictions/
                    └── predictions.csv
```

---

## 🔍 MLflow UI (Optional)

For the full native MLflow experience:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open: **http://localhost:5000**

### Features in MLflow UI:
- Side-by-side run comparison
- Detailed charts and visualizations
- Artifact download
- Advanced filtering
- Run parameter search

---

## 📸 Dashboard Screenshots

### All Runs View
```
┌─────────────────────────────────────────────┐
│ 📈 MLFlow Tracking Dashboard                │
├─────────────────────────────────────────────┤
│ [All Runs] [Training] [Inference]           │
│                                             │
│ Summary Statistics                          │
│ ┌─────────┬──────────┬───────────┐         │
│ │ Total 8 │ Train 3  │ Infer 5   │         │
│ └─────────┴──────────┴───────────┘         │
│ Average Accuracy: 96.5%                     │
│                                             │
│ ┌────────────────────────────────┐         │
│ │ Run: 3a2f... [TRAINING] ──────│         │
│ │ Time: 2024-11-21 10:30        │         │
│ │ Accuracy: 96.67%              │         │
│ │ F1-Score: 96.65%              │         │
│ └────────────────────────────────┘         │
│                                             │
│ ┌────────────────────────────────┐         │
│ │ Run: 7b4e... [INFERENCE] ─────│         │
│ │ Type: Batch Prediction         │         │
│ │ Samples: 100                   │         │
│ │ Avg Confidence: 98.2%          │         │
│ └────────────────────────────────┘         │
└─────────────────────────────────────────────┘
```

---

## 🎨 Color Coding

| Element | Color | Meaning |
|---------|-------|---------|
| Green Badge | 🟢 | Training Run |
| Blue Badge | 🔵 | Inference Run |
| Purple Gradient | 🟣 | Active Tab |
| Left Border (Green) | ⬜🟩 | Training Run (hover) |
| Left Border (Blue) | ⬜🔵 | Inference Run (hover) |

---

## 🧪 Testing

See `QUICK_START_TESTING.md` for:
- Step-by-step testing scenarios
- Expected results
- Verification checklist
- Troubleshooting tips

---

## 📚 Documentation

- `MLFLOW_ENHANCEMENTS.md` - Detailed feature documentation
- `BEFORE_AFTER_COMPARISON.md` - What changed and why
- `QUICK_START_TESTING.md` - Testing guide

---

## 🔧 Configuration

### Port Configuration
Change port in `main.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8230)  # Change here
```

### MLflow Configuration
MLflow tracking URI is set to local file system:
```python
mlflow.set_tracking_uri("file:./mlruns")
```

For remote tracking server:
```python
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
```

---

## 🚀 Production Deployment

### Using Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Using Gunicorn
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8230
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- FastAPI for the web framework
- MLflow for experiment tracking
- scikit-learn for ML capabilities
- The Iris dataset from UCI ML Repository

---

## 📞 Support

For issues and questions:
- 📧 Email: support@example.com
- 💬 Issues: GitHub Issues
- 📖 Docs: Project Wiki

---

**Built with ❤️ using FastAPI, MLflow, and scikit-learn**
