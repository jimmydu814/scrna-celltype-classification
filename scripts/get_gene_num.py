from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path("results/sparse_logreg/sparse_logreg_model.joblib")

model = joblib.load(MODEL_PATH)

coef = model.coef_

n_nonzero_coefficients = int(np.sum(coef != 0))
n_total_coefficients = int(coef.size)
sparsity = 1.0 - (n_nonzero_coefficients / n_total_coefficients)

# unique genes/features used at least once across all classes
n_genes_used = int(np.sum(np.any(coef != 0, axis=0)))

print("Number of classes:", coef.shape[0])
print("Number of input genes/features:", coef.shape[1])
print("Total coefficients:", n_total_coefficients)
print("Nonzero coefficients:", n_nonzero_coefficients)
print("Genes used:", n_genes_used)
print("Sparsity:", sparsity)