from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
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

MODEL_PATH = Path("results/models/logreg_finegrained_baseline.joblib")
METRICS_PATH = Path("results/metrics/logreg_finegrained_baseline_metrics.csv")
RUNTIME_PATH = Path("results/metrics/logreg_finegrained_baseline_runtime.csv")

VAL_REPORT_PATH = Path("results/tables/logreg_finegrained_val_classification_report.csv")
TEST_REPORT_PATH = Path("results/tables/logreg_finegrained_test_classification_report.csv")

TEST_PREDICTIONS_PATH = Path("results/tables/logreg_finegrained_test_predictions.csv")
CONFUSION_MATRIX_PATH = Path("results/figures/logreg_finegrained_test_confusion_matrix.png")


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
    print(classification_report(y_true, y_pred), flush=True)

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

    print("Loading split dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if FEATURE_KEY not in adata.obsm:
        raise ValueError(
            f"Feature key '{FEATURE_KEY}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if SPLIT_COL not in adata.obs.columns:
        raise ValueError(f"Missing split column: {SPLIT_COL}")

    print("\nUsing feature matrix:", FEATURE_KEY, flush=True)
    print("Using label column:", LABEL_COL, flush=True)

    X = np.asarray(adata.obsm[FEATURE_KEY])
    y = adata.obs[LABEL_COL].astype(str).to_numpy()
    splits = adata.obs[SPLIT_COL].astype(str).to_numpy()

    print("\nFeature matrix shape:", X.shape, flush=True)
    print("Labels:", sorted(np.unique(y)), flush=True)

    if np.isnan(X).any():
        raise ValueError("Feature matrix contains NaN values.")

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

    print("\n=== TRAINING LOGISTIC REGRESSION BASELINE ===", flush=True)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    train_start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - train_start_time
    training_time_per_cell_ms = (
        (training_time_seconds / len(y_train)) * 1000 if len(y_train) else np.nan
    )
    n_iter = int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else np.nan

    print(f"Training time: {training_time_seconds:.4f} seconds", flush=True)
    print(f"Training time per cell: {training_time_per_cell_ms:.4f} ms", flush=True)
    print(f"Solver iterations: {n_iter}", flush=True)

    labels = sorted(np.unique(y))

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
                "solver": model.solver,
                "max_iter": model.max_iter,
                "n_iter": n_iter,
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

    save_start_time = time.perf_counter()

    joblib.dump(model, MODEL_PATH)

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

    val_report.to_csv(VAL_REPORT_PATH)
    test_report.to_csv(TEST_REPORT_PATH)

    test_predictions = pd.DataFrame(
        {
            "cell_id": adata.obs_names[test_mask],
            "true_label": y_test,
            "predicted_label": test_pred,
        }
    )
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    cm = confusion_matrix(y_test, test_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Logistic Regression Baseline - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

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

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
