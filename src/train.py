"""CAHSI 2026: classify network flows, 36 classes, metric = Macro-F1.

Trains LightGBM on D1.pkl, predicts D2.pkl, writes answer.txt (integer
class ids 0-35, one per row, no trailing newline) and answer.zip.

Run: python src/train.py [--quick]
  --quick  subsample 300k rows for a fast smoke run
"""

import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DROP_COLS = ["Timestamp", "session_key", "flow_uid", "Label"]
FREQ_COLS = ["Source IP", "Destination IP"]
CAT_COLS = ["Protocol", "transport", "source_file", "capture_date"]


def build_features(df, freq_maps=None):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if freq_maps is None:
        freq_maps = {c: X[c].value_counts(normalize=True) for c in FREQ_COLS}
    for c in FREQ_COLS:
        X[c] = X[c].map(freq_maps[c]).fillna(0.0).astype("float32")
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    return X, freq_maps


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()

    # .pkl is the official competition format from the CAHSI organizers'
    # Google Drive (trusted source) — pickle load is intentional here.
    d1 = pd.read_pickle("data/D1.pkl")
    d2 = pd.read_pickle("data/D2.pkl")
    print(f"loaded D1 {d1.shape} D2 {d2.shape} ({time.time()-t0:.0f}s)")

    # id mapping MUST come from the full dataset (rare classes vanish from
    # subsamples and shift every id after them)
    classes = sorted(d1["Label"].unique())  # alphabetical == PDF id mapping
    label_to_id = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} classes")

    # cap majority classes: faster + better for macro-F1
    cap = 15_000 if quick else 50_000
    d1 = pd.concat([g.sample(min(len(g), cap), random_state=RANDOM_STATE)
                    for _, g in d1.groupby("Label")]).reset_index(drop=True)
    y = d1["Label"].map(label_to_id).to_numpy()
    print(f"train rows after cap {cap}: {len(d1)}")

    X, freq_maps = build_features(d1)
    X2, _ = build_features(d2, freq_maps)
    X2 = X2[X.columns]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        metric="None",  # early-stop on custom macro-F1 only
        n_estimators=100 if quick else 300,
        learning_rate=0.1,
        num_leaves=127,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    def macro_f1_eval(y_true, y_prob):
        pred = np.argmax(y_prob, axis=1)
        return "macro_f1", f1_score(y_true, pred, average="macro"), True

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=macro_f1_eval,
        callbacks=[early_stopping(40, first_metric_only=True),
                   log_evaluation(25)],
    )

    val_pred = model.predict(X_val)
    macro = f1_score(y_val, val_pred, average="macro")
    print(f"holdout Macro-F1 = {macro:.4f}  ({time.time()-t0:.0f}s)")

    preds = model.predict(X2).astype(int)
    Path("answer.txt").write_text("\n".join(map(str, preds)))
    with zipfile.ZipFile("answer.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write("answer.txt")
    print(f"wrote answer.txt ({len(preds)} rows) + answer.zip "
          f"({time.time()-t0:.0f}s total)")
    print("pred class distribution:")
    ids, counts = np.unique(preds, return_counts=True)
    for i, n in zip(ids, counts):
        print(f"  {i:2d} {classes[i]:30s} {n}")


if __name__ == "__main__":
    main()
