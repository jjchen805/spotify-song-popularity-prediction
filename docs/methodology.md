# Methodology

## Pipeline separation
- **data_prep**: cleaning + deterministic label creation
- **features**: preprocessing steps (scaling/encoding) in a scikit-learn Pipeline
- **train**: model training + hyperparameter search (CV) and artifact saving
- **evaluate**: standardized evaluation against a held-out test set

This separation improves reproducibility and reduces the risk of data leakage.
