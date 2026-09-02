from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.sparsefuncs import mean_variance_axis


INPUT_PATH = Path("data/processed/immune_finegrained_split.h5ad")
OUTPUT_PATH = Path("results/sparse_logreg/sparse_logreg_c_tuning_results.csv")

LABEL_COL = "fine_label"
SPLIT_COL = "split"

N_TOP_GENES = 3000
CONFIDENCE_THRESHOLD = 0.95
RANDOM_SEED = 42

C_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def to_dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def select_top_variable_genes_train_only(adata, train_mask, n_top_genes):
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


def evaluate_with_confidence(model, X, y_true, threshold):
    probs = model.predict_proba(X)
    class_labels = model.classes_

    pred_idx = np.argmax(probs, axis=1)
    raw_pred = class_labels[pred_idx]
    max_prob = probs[np.arange(len(probs)), pred_idx]

    accepted = max_prob >= threshold

    raw_accuracy = accuracy_score(y_true, raw_pred)
    raw_macro_f1 = f1_score(y_true, raw_pred, average="macro", zero_division=0)
    raw_weighted_f1 = f1_score(y_true, raw_pred, average="weighted", zero_division=0)

    coverage = accepted.mean()
    rejection_rate = 1.0 - coverage

    if accepted.sum() > 0:
        accepted_accuracy = accuracy_score(y_true[accepted], raw_pred[accepted])
        accepted_macro_f1 = f1_score(
            y_true[accepted],
            raw_pred[accepted],
            average="macro",
            zero_division=0,
        )
        accepted_weighted_f1 = f1_score(
            y_true[accepted],
            raw_pred[accepted],
            average="weighted",
            zero_division=0,
        )
    else:
        accepted_accuracy = np.nan
        accepted_macro_f1 = np.nan
        accepted_weighted_f1 = np.nan

    return {
        "raw_accuracy": raw_accuracy,
        "raw_macro_f1": raw_macro_f1,
        "raw_weighted_f1": raw_weighted_f1,
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "accepted_weighted_f1": accepted_weighted_f1,
    }


def main():
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

    print("\nSplit counts:", flush=True)
    print(adata.obs[SPLIT_COL].value_counts(), flush=True)

    print("\nLabel counts:", flush=True)
    print(adata.obs[LABEL_COL].value_counts(), flush=True)

    # Select genes using training data only
    top_gene_indices, selected_genes = select_top_variable_genes_train_only(
        adata=adata,
        train_mask=train_mask,
        n_top_genes=N_TOP_GENES,
    )

    print("\nBuilding train and validation matrices...", flush=True)

    X_train = adata.X[train_mask, :][:, top_gene_indices]
    X_val = adata.X[val_mask, :][:, top_gene_indices]

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]

    print("Train shape:", X_train.shape, flush=True)
    print("Validation shape:", X_val.shape, flush=True)

    print("\nConverting to dense float32...", flush=True)
    X_train = to_dense_float32(X_train)
    X_val = to_dense_float32(X_val)

    print("\nFitting scaler on training data only...", flush=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = []

    for C in C_VALUES:
        print(f"\n=== Training sparse logistic regression with C={C} ===", flush=True)

        model = LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=1,
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_SEED,
        )

        model.fit(X_train_scaled, y_train)

        eval_metrics = evaluate_with_confidence(
            model=model,
            X=X_val_scaled,
            y_true=y_val,
            threshold=CONFIDENCE_THRESHOLD,
        )

        n_nonzero_coefficients = int(np.sum(model.coef_ != 0))
        n_total_coefficients = int(model.coef_.size)
        sparsity = 1.0 - (n_nonzero_coefficients / n_total_coefficients)

        n_genes_used = int(np.sum(np.any(model.coef_ != 0, axis=0)))
        n_iter = int(np.max(model.n_iter_))

        result = {
            "C": C,
            "n_top_genes": N_TOP_GENES,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "n_iter": n_iter,
            "n_nonzero_coefficients": n_nonzero_coefficients,
            "n_total_coefficients": n_total_coefficients,
            "sparsity": sparsity,
            "n_genes_used": n_genes_used,
            **eval_metrics,
        }

        results.append(result)

        print("Validation raw macro F1:", f"{eval_metrics['raw_macro_f1']:.4f}", flush=True)
        print("Validation accepted macro F1:", f"{eval_metrics['accepted_macro_f1']:.4f}", flush=True)
        print("Coverage:", f"{eval_metrics['coverage']:.4f}", flush=True)
        print("Genes used:", n_genes_used, flush=True)
        print("Sparsity:", f"{sparsity:.4f}", flush=True)

    results_df = pd.DataFrame(results)

    # Suggested ranking:
    # 1. accepted macro F1
    # 2. raw macro F1
    # 3. fewer genes used
    results_df = results_df.sort_values(
        by=["accepted_macro_f1", "raw_macro_f1", "n_genes_used"],
        ascending=[False, False, True],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== TUNING RESULTS SORTED ===", flush=True)
    print(results_df, flush=True)

    best = results_df.iloc[0]

    print("\n=== SUGGESTED BEST C ===", flush=True)
    print(f"C = {best['C']}", flush=True)
    print(f"Validation raw macro F1 = {best['raw_macro_f1']:.4f}", flush=True)
    print(f"Validation accepted macro F1 = {best['accepted_macro_f1']:.4f}", flush=True)
    print(f"Coverage at threshold {CONFIDENCE_THRESHOLD} = {best['coverage']:.4f}", flush=True)
    print(f"Genes used = {int(best['n_genes_used'])}", flush=True)
    print(f"Sparsity = {best['sparsity']:.4f}", flush=True)

    print(f"\nSaved tuning results to: {OUTPUT_PATH.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()