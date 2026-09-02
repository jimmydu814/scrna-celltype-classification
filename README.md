# scRNA-seq Immune Cell Classification

Machine learning pipeline for classifying immune cell types from single-cell RNA sequencing (scRNA-seq) data using Python, Scanpy, AnnData, and scikit-learn.

This project evaluates broad and fine-grained immune-cell classification using logistic regression, random forest, confidence-aware prediction, and sparse logistic regression for gene-level interpretability.

## Dataset

The project uses public PBMC scRNA-seq data from the **Immunobiology of Aging Cohort PBMC Profiling** collection available through **CZ CELLxGENE Discover**.

Four curated immune-cell subsets were used:

- Naive CD4 T cells
- CD8 T, gdT, MAIT, and dnT cells
- NK cells and ILCs
- B and plasma cells

The final fine-grained classification dataset contains:

- 6 immune-cell types
- 750 cells per class
- 4,500 total cells

Large `.h5ad` data files are excluded from the repository.

Link to datasets: https://cellxgene.cziscience.com/collections/60a2676d-9f37-46cc-9b02-c7370a53be9c 

## Technologies

- Python
- Scanpy
- AnnData
- scikit-learn
- NumPy
- pandas
- SciPy
- joblib
- Git / GitHub

## Machine Learning Workflow

The pipeline includes:

- scRNA-seq data inspection and downsampling
- dataset merging and label construction
- class balancing
- stratified train/validation/test splitting
- training-only feature selection, scaling, and PCA
- logistic regression
- random forest
- confidence-aware prediction
- L1-regularized sparse logistic regression
- hyperparameter tuning
- gene-level feature interpretation

To reduce data leakage, feature selection, scaling, and PCA are fitted using training data only.

## Pipeline Scripts

Scripts are numbered in execution order:

```text
00_inspect_data.py
01_downsample_naive_cd4.py
02_downsample_cd8_related_T.py
03_downsample_NK_ILC.py
04_downsample_B_plasma.py
05_merge_subsets.py
06_make_finegrained_dataset.py
07_split_train_val_test.py
08_fit_train_only_pca.py
09_train_logreg_baseline.py
10_train_random_forest.py
11_tune_logreg_confidence.py
12_tune_sparse_logreg_c.py
13_train_sparse_logreg_final.py
14_add_gene_symbols.py
```

## Repository Structure

```text
scrna-celltype-classification/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── scripts/
├── results/
│   ├── metrics/
│   ├── confidence_logreg/
│   └── sparse_logreg/
├── docs/
│   └── project_report.pdf
├── README.md
└── .gitignore
```

Large raw and processed data files are excluded from version control.

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the required CELLxGENE `.h5ad` files in `data/raw/`, then run the numbered scripts sequentially from the project root.

Example:

```bash
python scripts/01_downsample_naive_cd4.py
python scripts/02_downsample_cd8_related_T.py
python scripts/03_downsample_NK_ILC.py
```

Continue through the remaining scripts in numerical order.

Tested with Python 3.14.4

## Project Report

The full methodology, experiments, results, discussion, and limitations are available in this repo.
