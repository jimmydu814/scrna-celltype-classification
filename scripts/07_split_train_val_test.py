from pathlib import Path

import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/immune_finegrained_balanced.h5ad")
OUTPUT_PATH = Path("data/processed/immune_finegrained_split.h5ad")

SPLIT_COUNTS_PATH = Path("results/tables/finegrained_split_counts.csv")

LABEL_COL = "fine_label"
RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def main():
    print("Loading merged dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    print("\n=== LABEL COUNTS ===", flush=True)
    print(adata.obs[LABEL_COL].value_counts(), flush=True)

    obs_names = adata.obs_names.to_numpy()
    y = adata.obs[LABEL_COL].astype(str).to_numpy()

    # First split: train vs temporary set
    train_obs, temp_obs, y_train, y_temp = train_test_split(
        obs_names,
        y,
        test_size=(1 - TRAIN_SIZE),
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Second split: validation vs test
    # Since temp is 30%, splitting it 50/50 gives 15% validation and 15% test.
    val_obs, test_obs, y_val, y_test = train_test_split(
        temp_obs,
        y_temp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    # Add split column to AnnData metadata
    adata.obs["split"] = "unassigned"
    adata.obs.loc[train_obs, "split"] = "train"
    adata.obs.loc[val_obs, "split"] = "val"
    adata.obs.loc[test_obs, "split"] = "test"

    print("\n=== SPLIT COUNTS ===", flush=True)
    split_counts = pd.crosstab(adata.obs["split"], adata.obs[LABEL_COL])
    print(split_counts, flush=True)

    print("\n=== SPLIT PROPORTIONS ===", flush=True)
    print(adata.obs["split"].value_counts(normalize=True), flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    adata.write(OUTPUT_PATH)
    split_counts.to_csv(SPLIT_COUNTS_PATH)

    print(f"\nSaved split dataset to: {OUTPUT_PATH.resolve()}", flush=True)
    print(f"Saved split counts to: {SPLIT_COUNTS_PATH.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()