from pathlib import Path
import numpy as np
import scanpy as sc

INPUT_PATH = Path("data/processed/immune_4class_merged.h5ad")
OUTPUT_PATH = Path("data/processed/immune_finegrained_balanced.h5ad")

LABEL_COL = "cell_type"
MIN_COUNT = 500
CELLS_PER_CLASS = 750
RANDOM_SEED = 42

adata = sc.read_h5ad(INPUT_PATH)

counts = adata.obs[LABEL_COL].value_counts()
keep_classes = counts[counts >= MIN_COUNT].index.tolist()

print("Keeping classes:")
for cls in keep_classes:
    print(cls, counts[cls])

adata = adata[adata.obs[LABEL_COL].isin(keep_classes)].copy()

rng = np.random.default_rng(RANDOM_SEED)
selected_idx = []

for cls in keep_classes:
    idx = np.where(adata.obs[LABEL_COL].values == cls)[0]
    n_take = min(CELLS_PER_CLASS, len(idx))
    chosen = rng.choice(idx, size=n_take, replace=False)
    selected_idx.extend(chosen)

adata_small = adata[selected_idx].copy()
adata_small.obs["fine_label"] = adata_small.obs[LABEL_COL].astype(str)

print(adata_small)
print(adata_small.obs["fine_label"].value_counts())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
adata_small.write(OUTPUT_PATH)