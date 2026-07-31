from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


INPUT_PATH = Path("data/processed/immune_finegrained_split_trainfit_pca.h5ad")

FEATURE_KEY = "X_pca_trainfit"
LABEL_COL = "fine_label"
SPLIT_COL = "split"

MODEL_PATH = Path("results/models/random_forest_finegrained.joblib")
METRICS_PATH = Path("results/metrics/random_forest_finegrained_metrics.csv")
RUNTIME_PATH = Path("results/metrics/random_forest_finegrained_runtime.csv")

VAL_REPORT_PATH = Path("results/tables/random_forest_finegrained_val_classification_report.csv")
TEST_REPORT_PATH = Path("results/tables/random_forest_finegrained_test_classification_report.csv")

TEST_PREDICTIONS_PATH = Path("results/tables/random_forest_finegrained_test_predictions.csv")
CONFUSION_MATRIX_PATH = Path("results/figures/random_forest_finegrained_test_confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = Path("results/tables/random_forest_pca_feature_importances.csv")


def evaluate_model(model, X, y_true, split_name, labels):
    predict_start = time.perf_counter()
    y_pred = model.predict(X)
    predict_time_seconds = time.perf_counter() - predict_start

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    prediction_time_per_cell_ms = (
        (predict_time_seconds / len(y_true)) * 1000 if len(y_true) else np.nan
    )

    print(f"\n=== {split_name.upper()} RESULTS ===", flush=True)
    print(f"Accuracy: {accuracy:.4f}", flush=True)
    print(f"Macro F1: {macro_f1:.4f}", flush=True)
    print(f"Weighted F1: {weighted_f1:.4f}", flush=True)
    print(f"Prediction time: {predict_time_seconds:.4f} seconds", flush=True)
    print(
        f"Prediction time per cell: {prediction_time_per_cell_ms:.4f} ms",
        flush=True,
    )

    print("\nClassification report:", flush=True)
    print(classification_report(y_true, y_pred, zero_division=0), flush=True)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    metrics = {
        "split": split_name,
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "prediction_time_seconds": predict_time_seconds,
        "prediction_time_per_cell_ms": prediction_time_per_cell_ms,
    }

    return metrics, report_df, y_pred


def main():
    script_start_time = time.perf_counter()

    print("Loading dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if FEATURE_KEY not in adata.obsm:
        raise ValueError(
            f"Feature key '{FEATURE_KEY}' not found. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if SPLIT_COL not in adata.obs.columns:
        raise ValueError(f"Missing split column: {SPLIT_COL}")

    X = np.asarray(adata.obsm[FEATURE_KEY])
    y = adata.obs[LABEL_COL].astype(str).to_numpy()
    splits = adata.obs[SPLIT_COL].astype(str).to_numpy()

    labels = sorted(np.unique(y))

    print("\nFeature matrix shape:", X.shape, flush=True)
    print("Labels:", labels, flush=True)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n=== SPLIT SHAPES ===", flush=True)
    print("Train:", X_train.shape, flush=True)
    print("Val:", X_val.shape, flush=True)
    print("Test:", X_test.shape, flush=True)

    print("\n=== TRAINING RANDOM FOREST ===", flush=True)

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    train_start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - train_start_time
    training_time_per_cell_ms = (
        (training_time_seconds / len(y_train)) * 1000 if len(y_train) else np.nan
    )

    print(f"Training time: {training_time_seconds:.4f} seconds", flush=True)
    print(f"Training time per cell: {training_time_per_cell_ms:.4f} ms", flush=True)

    val_metrics, val_report, val_pred = evaluate_model(
        model,
        X_val,
        y_val,
        split_name="val",
        labels=labels,
    )

    test_metrics, test_report, test_pred = evaluate_model(
        model,
        X_test,
        y_test,
        split_name="test",
        labels=labels,
    )

    for metrics in (val_metrics, test_metrics):
        metrics.update(
            {
                "training_time_seconds": training_time_seconds,
                "training_time_per_cell_ms": training_time_per_cell_ms,
                "n_train_samples": len(y_train),
                "n_features": X_train.shape[1],
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
                "max_features": model.max_features,
                "n_jobs": model.n_jobs,
            }
        )

    print("\nSaving outputs...", flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_start_time = time.perf_counter()

    joblib.dump(model, MODEL_PATH)

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

    val_report.to_csv(VAL_REPORT_PATH)
    test_report.to_csv(TEST_REPORT_PATH)

    # Save test predictions with confidence scores
    if hasattr(model, "predict_proba"):
        test_probs = model.predict_proba(X_test)
        max_prob = test_probs.max(axis=1)
    else:
        max_prob = np.full(len(test_pred), np.nan)

    test_predictions = pd.DataFrame(
        {
            "cell_id": adata.obs_names[test_mask],
            "true_label": y_test,
            "predicted_label": test_pred,
            "max_prediction_probability": max_prob,
        }
    )
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Random Forest - Fine-Grained Cell Type Classification")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    # PCA component feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": [f"PC{i+1}" for i in range(X.shape[1])],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    save_time_seconds = time.perf_counter() - save_start_time
    total_time_seconds = time.perf_counter() - script_start_time
    total_time_per_cell_ms = (
        (total_time_seconds / len(y)) * 1000 if len(y) else np.nan
    )

    runtime_df = pd.DataFrame(
        [
            {
                "stage": "training",
                "seconds": training_time_seconds,
                "n_samples": len(y_train),
                "milliseconds_per_cell": training_time_per_cell_ms,
            },
            {
                "stage": "validation_prediction",
                "seconds": val_metrics["prediction_time_seconds"],
                "n_samples": val_metrics["n_samples"],
                "milliseconds_per_cell": val_metrics["prediction_time_per_cell_ms"],
            },
            {
                "stage": "test_prediction",
                "seconds": test_metrics["prediction_time_seconds"],
                "n_samples": test_metrics["n_samples"],
                "milliseconds_per_cell": test_metrics["prediction_time_per_cell_ms"],
            },
            {
                "stage": "save_outputs",
                "seconds": save_time_seconds,
                "n_samples": np.nan,
                "milliseconds_per_cell": np.nan,
            },
            {
                "stage": "total_script",
                "seconds": total_time_seconds,
                "n_samples": len(y),
                "milliseconds_per_cell": total_time_per_cell_ms,
            },
        ]
    )
    runtime_df["n_samples"] = runtime_df["n_samples"].astype("Int64")
    runtime_df.to_csv(RUNTIME_PATH, index=False)

    print(f"Saved model to: {MODEL_PATH.resolve()}", flush=True)
    print(f"Saved metrics to: {METRICS_PATH.resolve()}", flush=True)
    print(f"Saved runtime summary to: {RUNTIME_PATH.resolve()}", flush=True)
    print(f"Saved validation report to: {VAL_REPORT_PATH.resolve()}", flush=True)
    print(f"Saved test report to: {TEST_REPORT_PATH.resolve()}", flush=True)
    print(f"Saved test predictions to: {TEST_PREDICTIONS_PATH.resolve()}", flush=True)
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH.resolve()}", flush=True)
    print(f"Saved PCA feature importances to: {FEATURE_IMPORTANCE_PATH.resolve()}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
