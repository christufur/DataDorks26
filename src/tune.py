"""Per-class probability-multiplier tuning for macro-F1 (no full retrain).

1. Trains a fast proxy model (same features, small cap) for holdout probas.
2. Greedy grid-search of per-class multipliers w_c maximizing holdout macro-F1.
3. Applies w to the ensemble's proba.npy -> answer_tuned.txt / answer_tuned.zip.

Run: PYTHONPATH=src python src/tune.py
"""

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from train import build_features

GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def main():
    t0 = time.time()
    # official competition pickles (trusted organizer source)
    d1 = pd.read_pickle("data/D1.pkl").drop_duplicates("flow_uid").reset_index(drop=True)
    classes = sorted(d1["Label"].unique())
    lid = {c: i for i, c in enumerate(classes)}
    d1 = pd.concat([g.sample(min(len(g), 3000), random_state=1)
                    for _, g in d1.groupby("Label")]).reset_index(drop=True)
    y = d1["Label"].map(lid).to_numpy()
    X, _ = build_features(d1)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1)

    model = LGBMClassifier(
        objective="multiclass", num_class=len(classes), n_estimators=80,
        learning_rate=0.05, num_leaves=127, class_weight="balanced",
        random_state=1, n_jobs=4, verbosity=-1)
    model.fit(X_tr, y_tr)
    pv = model.predict_proba(X_val)
    base = f1_score(y_val, np.argmax(pv, axis=1), average="macro")
    print(f"proxy holdout macro-F1 (argmax) = {base:.4f} ({time.time()-t0:.0f}s)")

    w = np.ones(len(classes))
    best = base
    for _pass in range(2):
        for c in range(len(classes)):
            for m in GRID:
                trial = w.copy(); trial[c] = m
                f = f1_score(y_val, np.argmax(pv * trial, axis=1), average="macro")
                if f > best:
                    best, w = f, trial
    print(f"tuned holdout macro-F1 = {best:.4f} (+{best-base:.4f})")
    nz = {classes[i]: round(float(w[i]), 2)
          for i in range(len(classes)) if w[i] != 1.0}
    print("multipliers:", nz)
    Path("weights.json").write_text(json.dumps(list(w)))

    if Path("proba.npy").exists():
        proba = np.load("proba.npy")
        preds = np.argmax(proba * w, axis=1)
        Path("answer_tuned.txt").write_text("\n".join(map(str, preds)))
        with zipfile.ZipFile("answer_tuned.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write("answer_tuned.txt", arcname="answer.txt")
        changed = (preds != np.argmax(proba, axis=1)).sum()
        print(f"wrote answer_tuned.zip ({changed} predictions changed)")
    else:
        print("proba.npy not there yet — rerun after ensemble writes it")


if __name__ == "__main__":
    main()
