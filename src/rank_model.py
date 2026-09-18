"""Ranking model: 2-seed balanced LightGBM whose averaged probabilities rank
the donor rows in prior_balance.py. This is NOT the base-label model
(src/ensemble.py is); it only produces proba.npy.

Feature set differs from the base model on purpose: IP frequency encoding
(no identity categoricals), one destination-port category, no temporal
features. The winning submission's 716 prior-balance moves were ranked by
exactly these probabilities.

Run: python src/rank_model.py    (writes proba.npy; ~20 min CPU)
"""

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEEDS = [42, 7]
CAP = 50_000
DROP_COLS = ["Timestamp", "session_key", "flow_uid", "Label"]
FREQ_COLS = ["Source IP", "Destination IP"]
CAT_COLS = ["Protocol", "transport", "source_file", "capture_date"]


def build_features(df, freq_maps=None):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if freq_maps is None:
        freq_maps = {c: X[c].value_counts(normalize=True) for c in FREQ_COLS}
        # ports seen >=100 times in train become categories; the rest -> -1
        vc = X["Destination Port"].value_counts()
        freq_maps["_common_ports"] = set(vc[vc >= 100].index)
    common = freq_maps["_common_ports"]
    cats = [-1] + sorted(common)  # fixed category set so train/test codes align
    X["dst_port_cat"] = pd.Categorical(
        X["Destination Port"].where(X["Destination Port"].isin(common), -1),
        categories=cats)
    for c in FREQ_COLS:
        X[c] = X[c].map(freq_maps[c]).fillna(0.0).astype("float32")
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    return X, freq_maps


def main():
    t0 = time.time()
    # official competition pickles from the organizers (trusted source)
    d1_full = pd.read_pickle("data/D1.pkl")
    d2 = pd.read_pickle("data/D2.pkl")
    d1_full = d1_full.drop_duplicates("flow_uid").reset_index(drop=True)

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

        print(f"seed {seed}: holdout Macro-F1 = "
              f"{f1_score(y_val, model.predict(X_val), average='macro'):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        p = model.predict_proba(X2)
        proba_sum = p if proba_sum is None else proba_sum + p
        np.save("proba.npy", proba_sum / si)
        print(f"saved proba.npy from {si} seed(s)", flush=True)


if __name__ == "__main__":
    main()
