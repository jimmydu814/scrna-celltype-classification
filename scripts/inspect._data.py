import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd
adata = sc.read_h5ad("data/processed/immune_4class_split_trainfit_pca.h5ad")

print("Fine-grained cell type counts:")
print(adata.obs["cell_type"].value_counts())

print("\nFine-grained cell types by broad label:")
print(pd.crosstab(adata.obs["broad_label"], adata.obs["cell_type"]))
print(adata)
print(type(adata.X))
print(adata.raw)
print(adata.var.columns)
print(adata.uns.keys())
print(adata.obsm.keys())

X_sample = adata.X[:10, :10]
print(X_sample)

if sparse.issparse(X_sample):
    vals = X_sample.data
else:
    vals = np.asarray(X_sample).ravel()

print("Sample values:", vals[:20])
print("Integer-like:", np.allclose(vals, np.round(vals)))