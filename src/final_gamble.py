"""Last-shot model: exact old 0.914 config (dups kept, no port-cat)
plus rare-class floor upsampling (classes < 1000 rows duplicated to 1000).
Writes answer_final.txt/zip and proba_final.npy. One seed (42).
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

CAP, FLOOR, SEED = 50_000, 1000, 42

t0 = time.time()
# official competition pickles (trusted organizer source); NO dedup on purpose
d1 = pd.read_pickle("data/D1.pkl")
d2 = pd.read_pickle("data/D2.pkl")
classes = sorted(d1["Label"].unique())
lid = {c: i for i, c in enumerate(classes)}

parts = []
for _, g in d1.groupby("Label"):
    g = g.sample(min(len(g), CAP), random_state=SEED)
    if len(g) < FLOOR:  # rare-class floor: duplicate up to FLOOR rows
        g = g.sample(FLOOR, replace=True, random_state=SEED)
    parts.append(g)
d1 = pd.concat(parts).reset_index(drop=True)
y = d1["Label"].map(lid).to_numpy()
print(f"train rows: {len(d1)}", flush=True)

X, fm = build_features(d1)
X2, _ = build_features(d2, fm)
X2 = X2[X.columns]
# revert the port-cat experiment — old winning config had no such column
X = X.drop(columns=["dst_port_cat"])
X2 = X2.drop(columns=["dst_port_cat"])

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=SEED)


def macro_f1_eval(y_true, y_prob):
    return "macro_f1", f1_score(y_true, np.argmax(y_prob, axis=1),
                                average="macro"), True


model = LGBMClassifier(
    objective="multiclass", num_class=len(classes), metric="None",
    n_estimators=250, learning_rate=0.05, num_leaves=127,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=macro_f1_eval,
          callbacks=[early_stopping(30, first_metric_only=True),
                     log_evaluation(50)])
print(f"holdout Macro-F1 = "
      f"{f1_score(y_val, model.predict(X_val), average='macro'):.4f} "
      f"({time.time()-t0:.0f}s)", flush=True)

proba = model.predict_proba(X2)
np.save("proba_final.npy", proba)
preds = np.argmax(proba, axis=1)
Path("answer_final.txt").write_text("\n".join(map(str, preds)))
with zipfile.ZipFile("answer_final.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("answer_final.txt", arcname="answer.txt")
print(f"wrote answer_final.zip ({time.time()-t0:.0f}s total)", flush=True)
