"""High-recall CAHSI 2026 network-traffic classifier.

Combines flow deduplication, categorical identities, frequency features,
port categories, temporal features, natural-distribution validation, and a
final refit on all selected training flows.
"""

import argparse
import gc
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
BASE_CAT_COLS = [
    "Source IP",
    "Destination IP",
    "Protocol",
    "transport",
    "source_file",
    "capture_date",
]
ENGINEERED_CAT_COLS = ["ip_pair", "src_port_cat", "dst_port_cat"]
CAT_COLS = BASE_CAT_COLS + ENGINEERED_CAT_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--final-rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--train", default="data/D1.pkl")
    parser.add_argument("--test", default="data/D2.pkl")
    parser.add_argument("--output", default="answer.txt")
    parser.add_argument("--probabilities", default="proba.npy")
    return parser.parse_args()


def cap_classes(df, cap, seed=RANDOM_STATE):
    parts = []
    for _, group in df.groupby("Label", sort=False, observed=True):
        if len(group) > cap:
            group = group.sample(cap, random_state=seed)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _aligned_category(train_col, other_col):
    values = pd.concat([train_col, other_col], ignore_index=True)
    categories = pd.Index(values.dropna().astype(str).unique())
    dtype = pd.CategoricalDtype(categories=categories)
    return (
        train_col.fillna("__missing__").astype(str).astype(dtype),
        other_col.fillna("__missing__").astype(str).astype(dtype),
    )


def build_features(train, other):
    """Build aligned features without using labels from the other frame."""
    train = train.drop(columns=[c for c in DROP_COLS if c in train]).copy()
    other = other.drop(columns=[c for c in DROP_COLS if c in other]).copy()
    if list(train.columns) != list(other.columns):
        raise ValueError("Train and prediction feature columns do not match")

    for col in ("Source IP", "Destination IP"):
        frequency = train[col].value_counts(normalize=True)
        train[f"{col} frequency"] = train[col].map(frequency).fillna(0).astype("float32")
        other[f"{col} frequency"] = other[col].map(frequency).fillna(0).astype("float32")

    train["ip_pair"] = train["Source IP"].astype(str) + ">" + train["Destination IP"].astype(str)
    other["ip_pair"] = other["Source IP"].astype(str) + ">" + other["Destination IP"].astype(str)

    for port, feature in (
        ("Source Port", "src_port_cat"),
        ("Destination Port", "dst_port_cat"),
    ):
        common = set(train[port].value_counts().loc[lambda x: x >= 50].index)
        train[feature] = train[port].where(train[port].isin(common), -1).astype(str)
        other[feature] = other[port].where(other[port].isin(common), -1).astype(str)

    for frame in (train, other):
        start = pd.to_numeric(frame["flow_start_ts"], errors="coerce")
        frame["flow_hour"] = ((start // 3600) % 24).astype("float32")
        frame["flow_weekday"] = ((start // 86400 + 3) % 7).astype("float32")
        frame["duration_error"] = (
            (frame["flow_end_ts"] - frame["flow_start_ts"]) * 1_000_000
            - frame["Flow Duration"]
        ).astype("float32")

    for col in CAT_COLS:
        train[col], other[col] = _aligned_category(train[col], other[col])

    numeric = train.select_dtypes(include=[np.number]).columns
    train[numeric] = train[numeric].replace([np.inf, -np.inf], np.nan)
    other[numeric] = other[numeric].replace([np.inf, -np.inf], np.nan)
    return train, other


def make_model(n_estimators, seed=RANDOM_STATE):
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=5,
        max_bin=255,
        colsample_bytree=0.9,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def macro_f1_eval(y_true, y_probability):
    prediction = np.argmax(y_probability, axis=1)
    score = f1_score(
        y_true,
        prediction,
        labels=np.arange(N_CLASSES),
        average="macro",
        zero_division=0,
    )
    return "macro_f1", score, True


def fit_validation(d1, classes, cap, rounds):
    """Validate against the natural class distribution, then return rounds."""
    train_raw, valid_raw = train_test_split(
        d1,
        test_size=0.1,
        stratify=d1["Label"],
        random_state=RANDOM_STATE,
    )
    train_raw = cap_classes(train_raw, cap)
    label_to_id = {label: index for index, label in enumerate(classes)}
    y_train = train_raw.pop("Label").map(label_to_id).to_numpy()
    y_valid = valid_raw.pop("Label").map(label_to_id).to_numpy()
    X_train, X_valid = build_features(train_raw, valid_raw)
    del train_raw, valid_raw
    gc.collect()

    model = make_model(rounds)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=macro_f1_eval,
        callbacks=[
            early_stopping(30, first_metric_only=True),
            log_evaluation(20),
        ],
        categorical_feature=CAT_COLS,
    )
    prediction = model.predict(X_valid)
    score = f1_score(
        y_valid,
        prediction,
        labels=np.arange(N_CLASSES),
        average="macro",
        zero_division=0,
    )
    print(f"Natural-distribution holdout Macro-F1: {score:.6f}")
    print(
        classification_report(
            y_valid,
            prediction,
            labels=np.arange(N_CLASSES),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )
    best_round = model.best_iteration_ or rounds
    del model, X_train, X_valid, y_train, y_valid
    gc.collect()
    return best_round


def write_submission(predictions, path):
    if len(predictions) != 312_107:
        raise ValueError(f"Expected 312107 predictions, got {len(predictions)}")
    if predictions.min() < 0 or predictions.max() >= N_CLASSES:
        raise ValueError("Prediction outside official class range 0..35")
    output = Path(path)
    output.write_text("\n".join(map(str, predictions)), encoding="utf-8")
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, arcname="answer.txt")
    return archive_path


def main():
    args = parse_args()
    started = time.time()
    d1 = pd.read_pickle(args.train)
    d1 = d1.drop_duplicates("flow_uid").reset_index(drop=True)
    classes = sorted(d1["Label"].unique())
    if len(classes) != N_CLASSES:
        raise ValueError(f"Expected 36 classes, found {len(classes)}")
    print(f"Unique labeled flows: {len(d1):,}")

    cap = 10_000 if args.quick else 50_000
    rounds = 130 if args.quick else 280
    if args.skip_validation:
        if args.final_rounds is None:
            raise ValueError("--skip-validation requires --final-rounds")
        best_round = args.final_rounds
    else:
        best_round = fit_validation(d1, classes, cap, rounds)
        print(f"Best validation iteration: {best_round}")

    final_raw = cap_classes(d1, cap, seed=args.seed)
    del d1
    d2 = pd.read_pickle(args.test)
    label_to_id = {label: index for index, label in enumerate(classes)}
    y = final_raw.pop("Label").map(label_to_id).to_numpy()
    X, X_test = build_features(final_raw, d2)
    del final_raw, d2
    gc.collect()

    model = make_model(best_round, seed=args.seed)
    model.fit(
        X,
        y,
        categorical_feature=CAT_COLS,
        callbacks=[log_evaluation(0)],
    )
    probabilities = model.predict_proba(X_test)
    np.save(args.probabilities, probabilities)
    predictions = np.argmax(probabilities, axis=1).astype(int)
    archive = write_submission(predictions, args.output)
    print(f"Wrote {archive} in {time.time() - started:.1f}s")
    for class_id, count in enumerate(np.bincount(predictions, minlength=N_CLASSES)):
        print(f"{class_id:2d} {classes[class_id]:30s} {count:7,d}")


if __name__ == "__main__":
    main()
