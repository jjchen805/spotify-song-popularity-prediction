from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def split_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = [c for c in X.columns if c not in num_features]
    return num_features, cat_features

def build_preprocessor_onehot(X: pd.DataFrame) -> ColumnTransformer:
    num_features, cat_features = split_features(X)
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_preprocessor_ordinal(X: pd.DataFrame) -> ColumnTransformer:
    # For PCA branch: ordinal-encode categoricals into numeric space, then scale + PCA
    num_features, cat_features = split_features(X)
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")