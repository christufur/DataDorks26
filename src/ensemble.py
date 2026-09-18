"""Seed-ensemble of balanced LightGBM models; averages predicted probs.

Writes answer.txt + answer.zip after EVERY seed finishes, so an upload
artifact always exists. Run: python src/ensemble.py
"""

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from train import build_features

SEEDS = [42, 7, 2026]
CAP = 50_000


def main():
    t0 = time.time()
    # official competition pickles from organizers' drive — trusted source
    d1_full = pd.read_pickle("data/D1.pkl")
    d2 = pd.read_pickle("data/D2.pkl")

    classes = sorted(d1_full["Label"].unique())
    label_to_id = {c: i for i, c in enumerate(classes)}

    proba_sum = None
    for si, seed in enumerate(SEEDS, 1):
        d1 = pd.concat([g.sample(min(len(g), CAP), random_state=seed)
                        for _, g in d1_full.groupby("Label")]).reset_index(drop=True)
        y = d1["Label"].map(label_to_id).to_numpy()
        X, freq_maps = build_features(d1)
        X2, _ = build_features(d2, freq_maps)
        X2 = X2[X.columns]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=seed)

        def macro_f1_eval(y_true, y_prob):
            pred = np.argmax(y_prob, axis=1)
            return "macro_f1", f1_score(y_true, pred, average="macro"), True

        model = LGBMClassifier(
            objective="multiclass", num_class=len(classes), metric="None",
            n_estimators=250, learning_rate=0.05, num_leaves=127,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric=macro_f1_eval,
                  callbacks=[early_stopping(30, first_metric_only=True),
                             log_evaluation(50)])

        macro = f1_score(y_val, model.predict(X_val), average="macro")
        print(f"seed {seed}: holdout Macro-F1 = {macro:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        p = model.predict_proba(X2)
        proba_sum = p if proba_sum is None else proba_sum + p

        preds = np.argmax(proba_sum, axis=1)
        Path("answer.txt").write_text("\n".join(map(str, preds)))
        with zipfile.ZipFile("answer.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write("answer.txt")
        print(f"wrote answer.txt/zip from {si} seed(s) "
              f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
