"""Train and validate the CAHSI 2026 network-traffic classifier.

The official metric is unweighted Macro-F1 across 36 classes. The script
class-caps the training set for speed, preserves rare classes, validates on a
stratified holdout, then refits on all selected rows before predicting D2.

Run:
    python src/train.py --quick
    python src/train.py
"""

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_CLASSES = 36
DROP_COLS = ["Timestamp", "session_key", "flow_uid", "Label"]
CAT_COLS = [
    "Source IP",
    "Destination IP",
    "Protocol",
    "transport",
    "source_file",
    "capture_date",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer rows/trees for a fast end-to-end smoke test.",
    )
    parser.add_argument("--train", default="data/D1.pkl")
    parser.add_argument("--test", default="data/D2.pkl")
    parser.add_argument("--output", default="answer.txt")
    return parser.parse_args()


def select_training_rows(df, cap):
    """Cap common classes while retaining every example of rare classes."""
    parts = []
    for _, group in df.groupby("Label", sort=False, observed=True):
        if len(group) > cap:
            group = group.sample(cap, random_state=RANDOM_STATE)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def build_features(train, test):
    """Create aligned, LightGBM-compatible train and test matrices."""
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns]).copy()
    test = test.drop(columns=[c for c in DROP_COLS if c in test.columns]).copy()

    if list(train.columns) != list(test.columns):
        missing = sorted(set(train) - set(test))
        extra = sorted(set(test) - set(train))
        raise ValueError(f"Feature mismatch: missing={missing}, extra={extra}")

    # Source IP has 132 values and destination IP has 3,666. Keeping identity
    # is more informative than reducing each IP to its global frequency.
    for col in CAT_COLS:
        if col not in train:
            continue
        categories = pd.Index(
            pd.concat([train[col], test[col]], ignore_index=True)
            .dropna()
            .astype(str)
            .unique()
        )
        dtype = pd.CategoricalDtype(categories=categories)
        train[col] = train[col].astype(str).astype(dtype)
        test[col] = test[col].astype(str).astype(dtype)

    numeric = train.select_dtypes(include=[np.number]).columns
    train[numeric] = train[numeric].replace([np.inf, -np.inf], np.nan)
    test[numeric] = test[numeric].replace([np.inf, -np.inf], np.nan)
    return train, test


def make_model(n_estimators, class_weight):
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",
        n_estimators=n_estimators,
        learning_rate=0.08,
        num_leaves=63,
        min_child_samples=5,
        max_bin=255,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.2,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def macro_f1_eval(y_true, y_prob):
    pred = np.argmax(y_prob, axis=1)
    score = f1_score(
        y_true, pred, labels=np.arange(N_CLASSES), average="macro"
    )
    return "macro_f1", score, True


def validate_submission(predictions, n_rows):
    predictions = np.asarray(predictions)
    if predictions.shape != (n_rows,):
        raise ValueError(f"Expected {n_rows} predictions, got {predictions.shape}")
    if not np.issubdtype(predictions.dtype, np.integer):
        raise TypeError("Predictions must be integer class IDs")
    if predictions.min() < 0 or predictions.max() >= N_CLASSES:
        raise ValueError("Predictions must be in the range 0..35")


def write_submission(predictions, output_path):
    output = Path(output_path)
    output.write_text("\n".join(map(str, predictions)), encoding="utf-8")
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, arcname="answer.txt")
    return zip_path


def main():
    args = parse_args()
    t0 = time.time()

    # The official organizer files are trusted pickle files.
    d1 = pd.read_pickle(args.train)
    d2 = pd.read_pickle(args.test)
    print(f"Loaded D1 {d1.shape} and D2 {d2.shape} in {time.time() - t0:.1f}s")

    classes = sorted(d1["Label"].unique())
    if len(classes) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} labels, found {len(classes)}")
    label_to_id = {label: index for index, label in enumerate(classes)}
    print("Label map:")
    for class_id, label in enumerate(classes):
        print(f"  {class_id:2d}: {label}")

    cap = 8_000 if args.quick else 30_000
    selected = select_training_rows(d1, cap)
    del d1
    y = selected.pop("Label").map(label_to_id).astype("int16").to_numpy()
    print(f"Selected {len(selected):,} training rows (per-class cap={cap:,})")

    X, X_test = build_features(selected, d2)
    del selected, d2

    indices = np.arange(len(y))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    n_estimators = 120 if args.quick else 350
    validation_model = make_model(n_estimators, class_weight="balanced")
    validation_model.fit(
        X.iloc[train_idx],
        y[train_idx],
        eval_set=[(X.iloc[valid_idx], y[valid_idx])],
        eval_metric=macro_f1_eval,
        callbacks=[
            early_stopping(35, first_metric_only=True, verbose=True),
            log_evaluation(20),
        ],
        categorical_feature=[c for c in CAT_COLS if c in X],
    )

    valid_pred = validation_model.predict(X.iloc[valid_idx]).astype(int)
    macro = f1_score(
        y[valid_idx],
        valid_pred,
        labels=np.arange(N_CLASSES),
        average="macro",
    )
    print(f"\nHoldout Macro-F1: {macro:.6f}")
    print(
        classification_report(
            y[valid_idx],
            valid_pred,
            labels=np.arange(N_CLASSES),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    best_iteration = validation_model.best_iteration_ or n_estimators
    print(f"Refitting all selected rows with {best_iteration} boosting rounds")
    final_model = make_model(best_iteration, class_weight="balanced")
    final_model.fit(
        X,
        y,
        categorical_feature=[c for c in CAT_COLS if c in X],
        callbacks=[log_evaluation(0)],
    )

    predictions = final_model.predict(X_test).astype(int)
    validate_submission(predictions, len(X_test))
    zip_path = write_submission(predictions, args.output)
    print(
        f"Wrote {args.output} ({len(predictions):,} rows) and {zip_path} "
        f"in {time.time() - t0:.1f}s"
    )
    print("Prediction distribution:")
    counts = np.bincount(predictions, minlength=N_CLASSES)
    for class_id, (label, count) in enumerate(zip(classes, counts)):
        print(f"  {class_id:2d} {label:30s} {count:7,d}")


if __name__ == "__main__":
    main()
