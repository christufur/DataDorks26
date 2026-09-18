"""Pipeline for CAHSI challenge. Regression, scored by R2.

Submission: answer.txt — one non-negative float per row, no header, no NaN/inf.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

RANDOM_STATE = 42


def load_data(train_path, test_path):
    for p in (Path(train_path), Path(test_path)):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — drop CSVs in data/")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def clean_data(train, test, target_col):
    """Median-impute numeric, mode-impute categorical. Fit on train only."""
    if target_col not in train.columns:
        raise KeyError(f"'{target_col}' not in train columns: {list(train.columns)}")
    train, test = train.copy(), test.copy()
    for col in [c for c in train.columns if c != target_col]:
        if col not in test.columns:
            raise KeyError(f"train column '{col}' missing from test")
        if pd.api.types.is_numeric_dtype(train[col]):
            fill = train[col].median()
        else:
            fill = train[col].mode(dropna=True).iloc[0]
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)
    return train.dropna(subset=[target_col]).reset_index(drop=True), test


def encode_features(train, test, target_col, id_col=None, max_onehot=10):
    """One-hot low-cardinality categoricals, frequency-encode the rest.

    Returns X_train, y_train, X_test.
    """
    y = train[target_col]
    drop = [target_col] + ([id_col] if id_col and id_col in train.columns else [])
    X_train = train.drop(columns=drop)
    X_test = test.drop(columns=[id_col] if id_col and id_col in test.columns else [])

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        if X_train[col].nunique() <= max_onehot:
            cats = sorted(X_train[col].astype(str).unique())
            X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats)
            X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)
        else:
            freq = X_train[col].value_counts(normalize=True)
            X_train[col] = X_train[col].map(freq).fillna(0.0)
            X_test[col] = X_test[col].map(freq).fillna(0.0)

    X_train = pd.get_dummies(X_train, dtype=float)
    X_test = pd.get_dummies(X_test, dtype=float).reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, y, X_test


def get_model(name):
    zoo = {
        "baseline": lambda: DummyRegressor(strategy="mean"),
        "linear": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        zoo["xgboost"] = lambda: XGBRegressor(
            n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0)
    if HAS_LGBM:
        zoo["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=-1)
    if name not in zoo:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(zoo)}")
    return zoo[name]()


def evaluate_models(X, y, model_names=None, n_splits=5):
    """K-fold CV, R2 (the official metric). Higher = better."""
    if model_names is None:
        model_names = ["baseline", "linear", "random_forest"]
        model_names += ["xgboost"] if HAS_XGB else []
        model_names += ["lightgbm"] if HAS_LGBM else []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in model_names:
        scores = cross_val_score(get_model(name), X, y, cv=cv, scoring="r2")
        rows.append({"model": name, "r2_mean": scores.mean(), "r2_std": scores.std()})
        print(f"  {name}: r2={scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows).set_index("model").sort_values("r2_mean", ascending=False)


def train_and_predict(model_name, X_train, y_train, X_test):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def export_submission(predictions, path="answer.txt"):
    """One non-negative float per row, no header, no NaN/inf."""
    preds = np.asarray(predictions, dtype=float)
    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN or inf")
    preds = np.clip(preds, 0, None)  # rule: non-negative
    np.savetxt(path, preds, fmt="%.6f")
    print(f"Wrote {path}: {len(preds)} rows, min={preds.min():.4f}, max={preds.max():.4f}")
    return preds
