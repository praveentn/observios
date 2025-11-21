from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from datetime import datetime
from typing import List

app = FastAPI(title="Iris ML Model App")

# Setup MLFlow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_classification")

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variable to store current model path and metadata
current_model_path = None
current_model_metadata = None

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/train")
async def train_model(file: UploadFile = File(None)):
    """Train model with uploaded CSV or default Iris dataset"""
    global current_model_path, current_model_metadata
    
    try:
        with mlflow.start_run():
            # Load data
            if file:
                # User uploaded CSV
                contents = await file.read()
                df = pd.read_csv(pd.io.common.BytesIO(contents))
                
                # Assume last column is target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                feature_names = df.columns[:-1].tolist()
                target_name = df.columns[-1]
                
                # Get unique class labels and create mapping
                unique_classes = sorted(np.unique(y))
                class_names = [str(cls) for cls in unique_classes]
                
                # Convert y to numeric if it's not already
                if not np.issubdtype(y.dtype, np.number):
                    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
                    y = np.array([class_to_idx[cls] for cls in y])
                else:
                    y = y.astype(int)
                
                mlflow.log_param("data_source", "uploaded_csv")
                mlflow.log_param("file_name", file.filename)
                
                # Store metadata
                model_metadata = {
                    "data_source": "custom_csv",
                    "class_names": class_names,
                    "feature_names": feature_names,
                    "n_classes": len(unique_classes),
                    "n_features": len(feature_names)
                }
            else:
                # Use default Iris dataset
                iris = load_iris()
                X, y = iris.data, iris.target
                feature_names = iris.feature_names
                target_name = "species"
                class_names = iris.target_names.tolist()
                
                mlflow.log_param("data_source", "sklearn_iris")
                
                # Store metadata
                model_metadata = {
                    "data_source": "sklearn_iris",
                    "class_names": class_names,
                    "feature_names": feature_names,
                    "n_classes": len(class_names),
                    "n_features": len(feature_names)
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_classes", len(class_names))
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"iris_model_{timestamp}.pkl"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata alongside model
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            current_model_path = model_path
            current_model_metadata = model_metadata
            
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metadata_path)
            
            return JSONResponse({
                "status": "success",
                "message": "Model trained successfully",
                "model_path": model_path,
                "accuracy": float(accuracy),
                "classification_report": report,
                "confusion_matrix": conf_matrix,
                "run_id": mlflow.active_run().info.run_id,
                "model_metadata": model_metadata
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    """Make prediction using the current model"""
    global current_model_path, current_model_metadata
    
    try:
        # Check if model exists
        if current_model_path is None or not os.path.exists(current_model_path):
            raise HTTPException(status_code=400, detail="No model available. Please train a model first.")
        
        # Load model
        model = joblib.load(current_model_path)
        
        # Load metadata
        if current_model_metadata is None:
            metadata_path = current_model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    current_model_metadata = json.load(f)
            else:
                # Fallback to Iris dataset metadata
                iris = load_iris()
                current_model_metadata = {
                    "data_source": "sklearn_iris",
                    "class_names": iris.target_names.tolist(),
                    "feature_names": iris.feature_names,
                    "n_classes": 3,
                    "n_features": 4
                }
        
        # Prepare input
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        # Get class name from metadata
        class_names = current_model_metadata["class_names"]
        
        # Validate prediction index
        if prediction < 0 or prediction >= len(class_names):
            raise HTTPException(
                status_code=500, 
                detail=f"Model predicted class {prediction}, but only {len(class_names)} classes available"
            )
        
        class_name = class_names[prediction]
        
        # Log prediction to MLFlow (Inference Experiment)
        mlflow.set_experiment("iris_inference")
        with mlflow.start_run(run_name=f"single_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log prediction type and model info
            mlflow.log_param("prediction_type", "single")
            mlflow.log_param("model_path", current_model_path)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Log input features
            mlflow.log_param("sepal_length", sepal_length)
            mlflow.log_param("sepal_width", sepal_width)
            mlflow.log_param("petal_length", petal_length)
            mlflow.log_param("petal_width", petal_width)
            
            # Log prediction output
            mlflow.log_param("predicted_class", class_name)
            mlflow.log_metric("predicted_class_index", prediction)
            mlflow.log_metric("prediction_confidence", float(probabilities[prediction]))
            
            # Log all probabilities
            for i, prob in enumerate(probabilities):
                mlflow.log_metric(f"probability_{class_names[i]}", float(prob))
        
        # Switch back to classification experiment
        mlflow.set_experiment("iris_classification")
        
        # Build probabilities dictionary
        probabilities_dict = {
            class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return JSONResponse({
            "status": "success",
            "prediction": prediction,
            "class_name": class_name,
            "probabilities": probabilities_dict
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from CSV file"""
    global current_model_path, current_model_metadata
    
    try:
        if current_model_path is None or not os.path.exists(current_model_path):
            raise HTTPException(status_code=400, detail="No model available. Please train a model first.")
        
        # Load model
        model = joblib.load(current_model_path)
        
        # Load metadata
        if current_model_metadata is None:
            metadata_path = current_model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    current_model_metadata = json.load(f)
            else:
                # Fallback to Iris dataset metadata
                iris = load_iris()
                current_model_metadata = {
                    "data_source": "sklearn_iris",
                    "class_names": iris.target_names.tolist(),
                    "feature_names": iris.feature_names,
                    "n_classes": 3,
                    "n_features": 4
                }
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Make predictions
        predictions = model.predict(df.values)
        probabilities = model.predict_proba(df.values)
        
        # Get class names from metadata
        class_names = current_model_metadata["class_names"]
        
        # Calculate batch statistics
        unique_preds, counts = np.unique(predictions, return_counts=True)
        pred_distribution = {
            class_names[int(pred)]: int(count) 
            for pred, count in zip(unique_preds, counts)
        }
        
        avg_confidence = np.mean([prob.max() for prob in probabilities])
        min_confidence = np.min([prob.max() for prob in probabilities])
        max_confidence = np.max([prob.max() for prob in probabilities])
        
        # Log batch prediction to MLFlow (Inference Experiment)
        mlflow.set_experiment("iris_inference")
        with mlflow.start_run(run_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log prediction metadata
            mlflow.log_param("prediction_type", "batch")
            mlflow.log_param("model_path", current_model_path)
            mlflow.log_param("input_file", file.filename)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Log batch metrics
            mlflow.log_metric("total_samples", len(predictions))
            mlflow.log_metric("avg_confidence", float(avg_confidence))
            mlflow.log_metric("min_confidence", float(min_confidence))
            mlflow.log_metric("max_confidence", float(max_confidence))
            mlflow.log_metric("unique_classes_predicted", len(unique_preds))
            
            # Log prediction distribution
            for class_name, count in pred_distribution.items():
                mlflow.log_metric(f"count_{class_name}", count)
                mlflow.log_metric(f"percentage_{class_name}", (count / len(predictions)) * 100)
            
            # Save predictions as artifact
            predictions_df = df.copy()
            predictions_df['predicted_class'] = [class_names[int(p)] for p in predictions]
            predictions_df['confidence'] = [float(prob.max()) for prob in probabilities]
            
            # Add probability columns
            for i, class_name in enumerate(class_names):
                predictions_df[f'prob_{class_name}'] = [float(prob[i]) for prob in probabilities]
            
            # Save to temporary file
            temp_path = f"./temp_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path, "predictions")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Switch back to classification experiment
        mlflow.set_experiment("iris_classification")
        
        # Build results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            pred_int = int(pred)
            
            # Validate prediction index
            if pred_int < 0 or pred_int >= len(class_names):
                class_name_str = f"Unknown (Class {pred_int})"
            else:
                class_name_str = class_names[pred_int]
            
            results.append({
                "sample_index": i,
                "prediction": pred_int,
                "class_name": class_name_str,
                "confidence": float(probs[pred_int]),
                "probabilities": {
                    class_names[j]: float(prob) 
                    for j, prob in enumerate(probs)
                }
            })
        
        return JSONResponse({
            "status": "success",
            "predictions": results,
            "total_samples": len(results),
            "batch_statistics": {
                "avg_confidence": float(avg_confidence),
                "min_confidence": float(min_confidence),
                "max_confidence": float(max_confidence),
                "prediction_distribution": pred_distribution
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all saved models"""
    try:
        models = []
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pkl'):
                filepath = os.path.join(MODEL_DIR, filename)
                
                # Try to load metadata
                metadata_path = filepath.replace('.pkl', '_metadata.json')
                metadata = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                models.append({
                    "filename": filename,
                    "path": filepath,
                    "size_kb": os.path.getsize(filepath) / 1024,
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_current": filepath == current_model_path,
                    "metadata": metadata
                })
        
        return JSONResponse({
            "status": "success",
            "models": models,
            "current_model": current_model_path
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
async def load_model(model_path: str = Form(...)):
    """Load a specific model for predictions"""
    global current_model_path, current_model_metadata
    
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Try to load the model to verify it's valid
        joblib.load(model_path)
        current_model_path = model_path
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                current_model_metadata = json.load(f)
        else:
            current_model_metadata = None
        
        return JSONResponse({
            "status": "success",
            "message": f"Model loaded successfully",
            "model_path": current_model_path,
            "metadata": current_model_metadata
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/experiments")
async def get_experiments():
    """Get all MLFlow experiments"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            })
        
        return JSONResponse({
            "status": "success",
            "experiments": exp_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/runs")
async def get_runs(experiment_name: str = None, max_results: int = 10):
    """Get MLFlow runs for an experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all experiments if no specific one is requested
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
            experiment_ids = [experiment.experiment_id]
        else:
            # Get all experiments
            experiments = client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments if exp.lifecycle_stage == "active"]
        
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        runs_data = []
        for run in runs:
            # Get experiment name
            exp = client.get_experiment(run.info.experiment_id)
            
            runs_data.append({
                "run_id": run.info.run_id,
                "experiment_name": exp.name,
                "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(
                    run.info.start_time / 1000
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "artifact_uri": run.info.artifact_uri
            })
        
        return JSONResponse({
            "status": "success",
            "runs": runs_data,
            "total": len(runs_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/run/{run_id}")
async def get_run_details(run_id: str):
    """Get detailed information about a specific run"""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        return JSONResponse({
            "status": "success",
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(
                run.info.start_time / 1000
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(
                run.info.end_time / 1000
            ).strftime("%Y-%m-%d %H:%M:%S") if run.info.end_time else None,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifact_uri": run.info.artifact_uri
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8230)
