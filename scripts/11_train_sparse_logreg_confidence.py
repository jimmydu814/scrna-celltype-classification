from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.sparsefuncs import mean_variance_axis


INPUT_PATH = Path("data/processed/immune_finegrained_split.h5ad")

LABEL_COL = "fine_label"
SPLIT_COL = "split"

N_TOP_GENES = 3000
CONFIDENCE_THRESHOLD = 0.95
RANDOM_SEED = 42

OUTPUT_DIR = Path("results/sparse_logreg")

MODEL_PATH = OUTPUT_DIR / "sparse_logreg_model.joblib"
SCALER_PATH = OUTPUT_DIR / "sparse_logreg_scaler.joblib"
GENE_LIST_PATH = OUTPUT_DIR / "sparse_logreg_selected_genes.txt"

METRICS_PATH = OUTPUT_DIR / "sparse_logreg_metrics.csv"
RUNTIME_PATH = OUTPUT_DIR / "sparse_logreg_runtime.csv"
VAL_REPORT_PATH = OUTPUT_DIR / "sparse_logreg_val_report.csv"
TEST_REPORT_PATH = OUTPUT_DIR / "sparse_logreg_test_report.csv"

VAL_PREDICTIONS_PATH = OUTPUT_DIR / "sparse_logreg_val_predictions_threshold_095.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "sparse_logreg_test_predictions_threshold_095.csv"

CONFUSION_MATRIX_PATH = OUTPUT_DIR / "sparse_logreg_test_confusion_matrix_threshold_095.png"
TOP_GENES_PATH = OUTPUT_DIR / "sparse_logreg_top_genes_by_class.csv"
COEFFICIENT_MATRIX_PATH = OUTPUT_DIR / "sparse_logreg_coefficients.csv"


def to_dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def select_top_variable_genes_train_only(adata, train_mask, n_top_genes):
    """
    Select top variable genes using only training cells.
    This avoids feature-selection leakage.
    """
    print("Selecting top variable genes using training cells only...", flush=True)

    X_train = adata.X[train_mask, :]

    if sparse.issparse(X_train):
        _, variances = mean_variance_axis(X_train, axis=0)
    else:
        variances = np.var(X_train, axis=0)

    variances = np.asarray(variances).ravel()

    n_top = min(n_top_genes, adata.n_vars)
    top_gene_indices = np.argsort(variances)[-n_top:]
    top_gene_indices = np.sort(top_gene_indices)

    selected_genes = adata.var_names[top_gene_indices].to_numpy()

    print(f"Selected {len(selected_genes)} genes.", flush=True)

    return top_gene_indices, selected_genes


def make_confidence_predictions(model, X, y_true, obs_names, threshold, split_name):
    predict_start = time.perf_counter()
    probs = model.predict_proba(X)
    prediction_time_seconds = time.perf_counter() - predict_start
    prediction_time_per_cell_ms = (
        (prediction_time_seconds / len(y_true)) * 1000 if len(y_true) else np.nan
    )

    class_labels = model.classes_
    pred_idx = np.argmax(probs, axis=1)

    raw_pred = class_labels[pred_idx]
    max_prob = probs[np.arange(len(probs)), pred_idx]

    confidence_pred = raw_pred.copy().astype(object)
    confidence_pred[max_prob < threshold] = "Uncertain"

    pred_df = pd.DataFrame(
        {
            "cell_id": obs_names,
            "true_label": y_true,
            "raw_prediction": raw_pred,
            "confidence_prediction": confidence_pred,
            "max_probability": max_prob,
            "accepted": max_prob >= threshold,
        }
    )

    for i, cls in enumerate(class_labels):
        pred_df[f"prob_{cls}"] = probs[:, i]

    print(f"\n=== {split_name.upper()} PREDICTION RUNTIME ===", flush=True)
    print(f"Prediction time: {prediction_time_seconds:.4f} seconds", flush=True)
    print(
        f"Prediction time per cell: {prediction_time_per_cell_ms:.4f} ms",
        flush=True,
    )

    prediction_runtime = {
        "n_samples": len(y_true),
        "prediction_time_seconds": prediction_time_seconds,
        "prediction_time_per_cell_ms": prediction_time_per_cell_ms,
    }

    return pred_df, prediction_runtime


def evaluate_predictions(pred_df, split_name):
    y_true = pred_df["true_label"].astype(str)
    raw_pred = pred_df["raw_prediction"].astype(str)
    confidence_pred = pred_df["confidence_prediction"].astype(str)
    accepted = pred_df["accepted"].to_numpy()

    raw_accuracy = accuracy_score(y_true, raw_pred)
    raw_macro_f1 = f1_score(y_true, raw_pred, average="macro", zero_division=0)
    raw_weighted_f1 = f1_score(y_true, raw_pred, average="weighted", zero_division=0)

    coverage = accepted.mean()
    rejection_rate = 1.0 - coverage

    if accepted.sum() > 0:
        accepted_true = y_true[accepted]
        accepted_pred = raw_pred[accepted]

        accepted_accuracy = accuracy_score(accepted_true, accepted_pred)
        accepted_macro_f1 = f1_score(
            accepted_true,
            accepted_pred,
            average="macro",
            zero_division=0,
        )
        accepted_weighted_f1 = f1_score(
            accepted_true,
            accepted_pred,
            average="weighted",
            zero_division=0,
        )
    else:
        accepted_accuracy = np.nan
        accepted_macro_f1 = np.nan
        accepted_weighted_f1 = np.nan

    full_accuracy_with_uncertain = accuracy_score(y_true, confidence_pred)

    metrics = {
        "split": split_name,
        "threshold": CONFIDENCE_THRESHOLD,
        "n_total": len(pred_df),
        "n_accepted": int(accepted.sum()),
        "n_rejected": int((~accepted).sum()),
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "raw_accuracy": raw_accuracy,
        "raw_macro_f1": raw_macro_f1,
        "raw_weighted_f1": raw_weighted_f1,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "accepted_weighted_f1": accepted_weighted_f1,
        "full_accuracy_with_uncertain": full_accuracy_with_uncertain,
    }

    print(f"\n=== {split_name.upper()} RAW PREDICTION RESULTS ===", flush=True)
    print(f"Accuracy: {raw_accuracy:.4f}", flush=True)
    print(f"Macro F1: {raw_macro_f1:.4f}", flush=True)
    print(f"Weighted F1: {raw_weighted_f1:.4f}", flush=True)

    print(f"\n=== {split_name.upper()} CONFIDENCE RESULTS, threshold={CONFIDENCE_THRESHOLD} ===", flush=True)
    print(f"Coverage: {coverage:.4f}", flush=True)
    print(f"Rejection rate: {rejection_rate:.4f}", flush=True)
    print(f"Accepted accuracy: {accepted_accuracy:.4f}", flush=True)
    print(f"Accepted macro F1: {accepted_macro_f1:.4f}", flush=True)

    report_dict = classification_report(
        y_true,
        raw_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    print("\nClassification report using raw predictions:", flush=True)
    print(report_df, flush=True)

    return metrics, report_df


def extract_top_genes(model, selected_genes, top_n=20):
    """
    Extract top positive and negative coefficients per class.

    Positive coefficients push the model toward that class.
    Negative coefficients push the model away from that class.
    """
    coef = model.coef_
    classes = model.classes_

    rows = []

    for class_idx, class_name in enumerate(classes):
        class_coef = coef[class_idx]

        top_pos_idx = np.argsort(class_coef)[-top_n:][::-1]
        top_neg_idx = np.argsort(class_coef)[:top_n]

        for rank, idx in enumerate(top_pos_idx, start=1):
            rows.append(
                {
                    "class": class_name,
                    "direction": "positive",
                    "rank": rank,
                    "gene": selected_genes[idx],
                    "coefficient": class_coef[idx],
                }
            )

        for rank, idx in enumerate(top_neg_idx, start=1):
            rows.append(
                {
                    "class": class_name,
                    "direction": "negative",
                    "rank": rank,
                    "gene": selected_genes[idx],
                    "coefficient": class_coef[idx],
                }
            )

    return pd.DataFrame(rows)


def main():
    script_start_time = time.perf_counter()

    print("Loading fine-grained split dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if SPLIT_COL not in adata.obs.columns:
        raise ValueError(f"Missing split column: {SPLIT_COL}")

    y_all = adata.obs[LABEL_COL].astype(str).to_numpy()
    splits = adata.obs[SPLIT_COL].astype(str).to_numpy()

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    print("\nSplit counts:", flush=True)
    print(adata.obs[SPLIT_COL].value_counts(), flush=True)

    print("\nLabel counts:", flush=True)
    print(adata.obs[LABEL_COL].value_counts(), flush=True)

    # 1. Select genes using training set only
    top_gene_indices, selected_genes = select_top_variable_genes_train_only(
        adata=adata,
        train_mask=train_mask,
        n_top_genes=N_TOP_GENES,
    )

    # 2. Build feature matrices
    print("\nSubsetting to selected genes...", flush=True)

    X_train = adata.X[train_mask, :][:, top_gene_indices]
    X_val = adata.X[val_mask, :][:, top_gene_indices]
    X_test = adata.X[test_mask, :][:, top_gene_indices]

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    obs_val = adata.obs_names[val_mask].to_numpy()
    obs_test = adata.obs_names[test_mask].to_numpy()

    print("Train shape before scaling:", X_train.shape, flush=True)
    print("Val shape before scaling:", X_val.shape, flush=True)
    print("Test shape before scaling:", X_test.shape, flush=True)

    # 3. Convert to dense and scale
    print("\nConverting to dense float32...", flush=True)

    X_train = to_dense_float32(X_train)
    X_val = to_dense_float32(X_val)
    X_test = to_dense_float32(X_test)

    print("\nFitting scaler on training data only...", flush=True)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train sparse logistic regression
    print("\nTraining sparse logistic regression...", flush=True)

    model = LogisticRegression(
        solver="saga",
        C=0.5,
        l1_ratio=1,
        max_iter=5000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    train_start_time = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    training_time_seconds = time.perf_counter() - train_start_time
    training_time_per_cell_ms = (
        (training_time_seconds / len(y_train)) * 1000 if len(y_train) else np.nan
    )
    n_iter = int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else np.nan

    print(f"Training time: {training_time_seconds:.4f} seconds", flush=True)
    print(f"Training time per cell: {training_time_per_cell_ms:.4f} ms", flush=True)
    print(f"Solver iterations: {n_iter}", flush=True)

    print("\nModel classes:", flush=True)
    print(list(model.classes_), flush=True)

    n_nonzero = np.sum(model.coef_ != 0)
    n_total = model.coef_.size
    sparsity = 1.0 - (n_nonzero / n_total)

    print("\n=== SPARSITY INFO ===", flush=True)
    print(f"Nonzero coefficients: {n_nonzero}", flush=True)
    print(f"Total coefficients: {n_total}", flush=True)
    print(f"Sparsity: {sparsity:.4f}", flush=True)

    # 5. Predict with confidence threshold
    print("\nGenerating validation predictions...", flush=True)
    val_pred_df, val_prediction_runtime = make_confidence_predictions(
        model=model,
        X=X_val_scaled,
        y_true=y_val,
        obs_names=obs_val,
        threshold=CONFIDENCE_THRESHOLD,
        split_name="val",
    )

    print("\nGenerating test predictions...", flush=True)
    test_pred_df, test_prediction_runtime = make_confidence_predictions(
        model=model,
        X=X_test_scaled,
        y_true=y_test,
        obs_names=obs_test,
        threshold=CONFIDENCE_THRESHOLD,
        split_name="test",
    )

    val_metrics, val_report = evaluate_predictions(val_pred_df, "val")
    test_metrics, test_report = evaluate_predictions(test_pred_df, "test")

    val_metrics.update(val_prediction_runtime)
    test_metrics.update(test_prediction_runtime)

    n_genes_used = int(np.sum(np.any(model.coef_ != 0, axis=0)))

    for metrics in (val_metrics, test_metrics):
        metrics.update(
            {
                "training_time_seconds": training_time_seconds,
                "training_time_per_cell_ms": training_time_per_cell_ms,
                "n_train_samples": len(y_train),
                "n_features": X_train_scaled.shape[1],
                "solver": model.solver,
                "penalty": model.penalty,
                "C": model.C,
                "l1_ratio": model.l1_ratio,
                "max_iter": model.max_iter,
                "n_iter": n_iter,
                "n_jobs": model.n_jobs,
                "n_nonzero_coefficients": int(n_nonzero),
                "n_total_coefficients": int(n_total),
                "sparsity": sparsity,
                "n_genes_used": n_genes_used,
            }
        )

    # 6. Extract gene coefficients
    print("\nExtracting top genes by class...", flush=True)

    top_genes_df = extract_top_genes(
        model=model,
        selected_genes=selected_genes,
        top_n=20,
    )

    coef_df = pd.DataFrame(
        model.coef_,
        index=model.classes_,
        columns=selected_genes,
    )

    # 7. Confusion matrix for raw predictions
    labels = list(model.classes_)
    cm = confusion_matrix(
        test_pred_df["true_label"],
        test_pred_df["raw_prediction"],
        labels=labels,
    )

    # 8. Save everything
    print("\nSaving outputs...", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_start_time = time.perf_counter()

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(GENE_LIST_PATH, "w") as f:
        for gene in selected_genes:
            f.write(f"{gene}\n")

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

    val_report.to_csv(VAL_REPORT_PATH)
    test_report.to_csv(TEST_REPORT_PATH)

    val_pred_df.to_csv(VAL_PREDICTIONS_PATH, index=False)
    test_pred_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    top_genes_df.to_csv(TOP_GENES_PATH, index=False)
    coef_df.to_csv(COEFFICIENT_MATRIX_PATH)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Sparse Logistic Regression - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    save_time_seconds = time.perf_counter() - save_start_time
    total_time_seconds = time.perf_counter() - script_start_time
    total_time_per_cell_ms = (
        (total_time_seconds / len(y_all)) * 1000 if len(y_all) else np.nan
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
                "n_samples": len(y_all),
                "milliseconds_per_cell": total_time_per_cell_ms,
            },
        ]
    )
    runtime_df["n_samples"] = runtime_df["n_samples"].astype("Int64")
    runtime_df.to_csv(RUNTIME_PATH, index=False)

    print(f"Saved model to: {MODEL_PATH.resolve()}", flush=True)
    print(f"Saved scaler to: {SCALER_PATH.resolve()}", flush=True)
    print(f"Saved selected genes to: {GENE_LIST_PATH.resolve()}", flush=True)
    print(f"Saved metrics to: {METRICS_PATH.resolve()}", flush=True)
    print(f"Saved runtime summary to: {RUNTIME_PATH.resolve()}", flush=True)
    print(f"Saved top genes to: {TOP_GENES_PATH.resolve()}", flush=True)
    print(f"Saved coefficient matrix to: {COEFFICIENT_MATRIX_PATH.resolve()}", flush=True)
    print(f"Saved test confusion matrix to: {CONFUSION_MATRIX_PATH.resolve()}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
