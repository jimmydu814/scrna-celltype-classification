from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


INPUT_PATH = Path("data/processed/immune_finegrained_split_trainfit_pca.h5ad")
MODEL_PATH = Path("results/models/logreg_finegrained_baseline.joblib")

FEATURE_KEY = "X_pca_trainfit"
LABEL_COL = "fine_label"
SPLIT_COL = "split"

OUTPUT_DIR = Path("results/confidence_logreg")

VAL_THRESHOLD_RESULTS_PATH = OUTPUT_DIR / "val_threshold_sweep.csv"
TEST_THRESHOLD_RESULTS_PATH = OUTPUT_DIR / "test_threshold_sweep.csv"

VAL_PREDICTIONS_PATH = OUTPUT_DIR / "val_confidence_predictions.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_confidence_predictions.csv"

TEST_CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "test_classification_report_with_uncertain.csv"
TEST_CONFUSION_MATRIX_PATH = OUTPUT_DIR / "test_confusion_matrix_with_uncertain.png"

# Try a range of confidence thresholds.
# A prediction is accepted only if max probability >= threshold.
THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]

# When automatically choosing a threshold, require at least this much coverage.
# Coverage = fraction of cells not marked as Uncertain.
MIN_VALIDATION_COVERAGE = 0.90


def make_confidence_predictions(model, X, y_true, obs_names, threshold):
    """
    Run logistic regression probabilities and apply confidence-based rejection.
    """
    probs = model.predict_proba(X)

    class_labels = model.classes_
    pred_idx = np.argmax(probs, axis=1)

    raw_pred = class_labels[pred_idx]
    max_prob = probs[np.arange(len(probs)), pred_idx]

    confidence_pred = raw_pred.copy().astype(object)
    confidence_pred[max_prob < threshold] = "Uncertain"

    df = pd.DataFrame(
        {
            "cell_id": obs_names,
            "true_label": y_true,
            "raw_prediction": raw_pred,
            "confidence_prediction": confidence_pred,
            "max_probability": max_prob,
            "accepted": max_prob >= threshold,
        }
    )

    # Add probability for each class
    for i, cls in enumerate(class_labels):
        df[f"prob_{cls}"] = probs[:, i]

    return df


def evaluate_threshold(model, X, y_true, obs_names, threshold):
    """
    Evaluate a confidence threshold.

    Metrics:
    - coverage: fraction of cells accepted
    - rejection_rate: fraction marked Uncertain
    - accepted_accuracy: accuracy only among accepted predictions
    - accepted_macro_f1: macro F1 only among accepted predictions
    - full_accuracy_with_uncertain: accuracy if Uncertain counts as incorrect
    """
    pred_df = make_confidence_predictions(
        model=model,
        X=X,
        y_true=y_true,
        obs_names=obs_names,
        threshold=threshold,
    )

    accepted_mask = pred_df["accepted"].to_numpy()
    coverage = accepted_mask.mean()
    rejection_rate = 1.0 - coverage
    n_total = len(pred_df)
    n_accepted = int(accepted_mask.sum())
    n_rejected = n_total - n_accepted

    # Accuracy if Uncertain is treated as incorrect
    full_accuracy = accuracy_score(
        pred_df["true_label"],
        pred_df["confidence_prediction"],
    )

    if n_accepted > 0:
        accepted_true = pred_df.loc[accepted_mask, "true_label"]
        accepted_pred = pred_df.loc[accepted_mask, "raw_prediction"]

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

    result = {
        "threshold": threshold,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "accepted_weighted_f1": accepted_weighted_f1,
        "full_accuracy_with_uncertain": full_accuracy,
    }

    return result, pred_df


def choose_threshold(val_results):
    """
    Choose a threshold using validation results only.

    Rule:
    - only consider thresholds with coverage >= MIN_VALIDATION_COVERAGE
    - among those, choose the one with highest accepted_macro_f1
    - if tied, choose the higher threshold because it is more conservative
    """
    candidates = val_results[
        val_results["coverage"] >= MIN_VALIDATION_COVERAGE
    ].copy()

    if len(candidates) == 0:
        print(
            "\nNo threshold met minimum validation coverage. "
            "Choosing threshold with highest accepted_macro_f1 regardless of coverage.",
            flush=True,
        )
        candidates = val_results.copy()

    candidates = candidates.sort_values(
        by=["accepted_macro_f1", "threshold"],
        ascending=[False, False],
    )

    chosen = candidates.iloc[0]
    return float(chosen["threshold"])


def main():
    print("Loading dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {INPUT_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model: {MODEL_PATH}\n"
            "Train the logistic regression fine-grained baseline first."
        )

    adata = sc.read_h5ad(INPUT_PATH)
    model = joblib.load(MODEL_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if FEATURE_KEY not in adata.obsm:
        raise ValueError(
            f"Feature key '{FEATURE_KEY}' not found. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if SPLIT_COL not in adata.obs.columns:
        raise ValueError(f"Missing split column: {SPLIT_COL}")

    X = np.asarray(adata.obsm[FEATURE_KEY])
    y = adata.obs[LABEL_COL].astype(str).to_numpy()
    splits = adata.obs[SPLIT_COL].astype(str).to_numpy()

    val_mask = splits == "val"
    test_mask = splits == "test"

    X_val = X[val_mask]
    y_val = y[val_mask]
    obs_val = adata.obs_names[val_mask].to_numpy()

    X_test = X[test_mask]
    y_test = y[test_mask]
    obs_test = adata.obs_names[test_mask].to_numpy()

    print("\nModel classes:", flush=True)
    print(list(model.classes_), flush=True)

    print("\nValidation shape:", X_val.shape, flush=True)
    print("Test shape:", X_test.shape, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSweeping thresholds on validation set...", flush=True)

    val_results = []
    val_prediction_dfs = {}

    for threshold in THRESHOLDS:
        result, pred_df = evaluate_threshold(
            model=model,
            X=X_val,
            y_true=y_val,
            obs_names=obs_val,
            threshold=threshold,
        )
        val_results.append(result)
        val_prediction_dfs[threshold] = pred_df

    val_results_df = pd.DataFrame(val_results)
    val_results_df.to_csv(VAL_THRESHOLD_RESULTS_PATH, index=False)

    print("\n=== VALIDATION THRESHOLD RESULTS ===", flush=True)
    print(val_results_df, flush=True)

    chosen_threshold = choose_threshold(val_results_df)

    print(f"\nChosen threshold from validation set: {chosen_threshold}", flush=True)

    # Save validation predictions at chosen threshold
    val_chosen_df = val_prediction_dfs[chosen_threshold]
    val_chosen_df.to_csv(VAL_PREDICTIONS_PATH, index=False)

    print("\nApplying thresholds to test set...", flush=True)

    test_results = []
    test_prediction_dfs = {}

    for threshold in THRESHOLDS:
        result, pred_df = evaluate_threshold(
            model=model,
            X=X_test,
            y_true=y_test,
            obs_names=obs_test,
            threshold=threshold,
        )
        test_results.append(result)
        test_prediction_dfs[threshold] = pred_df

    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(TEST_THRESHOLD_RESULTS_PATH, index=False)

    print("\n=== TEST THRESHOLD RESULTS ===", flush=True)
    print(test_results_df, flush=True)

    # Save test predictions at chosen validation threshold
    test_chosen_df = test_prediction_dfs[chosen_threshold]
    test_chosen_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print("\n=== TEST RESULTS AT CHOSEN THRESHOLD ===", flush=True)
    chosen_test_result = test_results_df[
        test_results_df["threshold"] == chosen_threshold
    ].iloc[0]
    print(chosen_test_result, flush=True)

    print("\nClassification report with Uncertain predictions:", flush=True)

    labels_with_uncertain = list(model.classes_) + ["Uncertain"]

    report_dict = classification_report(
        test_chosen_df["true_label"],
        test_chosen_df["confidence_prediction"],
        labels=labels_with_uncertain,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df, flush=True)

    report_df.to_csv(TEST_CLASSIFICATION_REPORT_PATH)

    # Confusion matrix including Uncertain column/row
    cm = confusion_matrix(
        test_chosen_df["true_label"],
        test_chosen_df["confidence_prediction"],
        labels=labels_with_uncertain,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels_with_uncertain,
    )
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(
        f"Confidence-Aware Logistic Regression\n"
        f"Chosen threshold = {chosen_threshold}"
    )
    plt.tight_layout()
    plt.savefig(TEST_CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print("\nSaved outputs:", flush=True)
    print(f"Validation threshold sweep: {VAL_THRESHOLD_RESULTS_PATH.resolve()}", flush=True)
    print(f"Test threshold sweep: {TEST_THRESHOLD_RESULTS_PATH.resolve()}", flush=True)
    print(f"Validation predictions: {VAL_PREDICTIONS_PATH.resolve()}", flush=True)
    print(f"Test predictions: {TEST_PREDICTIONS_PATH.resolve()}", flush=True)
    print(f"Test classification report: {TEST_CLASSIFICATION_REPORT_PATH.resolve()}", flush=True)
    print(f"Test confusion matrix: {TEST_CONFUSION_MATRIX_PATH.resolve()}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()