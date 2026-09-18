"""Deadline-safe final model.

Keeps the raw-row weighting that scored 0.9140, adds categorical copies of
the IP and destination-port identities, trains on all sampled rows, and saves
probabilities for post-processing.

Run: PYTHONPATH=src python src/final_model.py
"""

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation


SEED = 42
CAP = 50_000
N_ESTIMATORS = 130
DROP_COLS = ["Timestamp", "session_key", "flow_uid", "Label"]
FREQ_COLS = ["Source IP", "Destination IP"]
CAT_COLS = ["Protocol", "transport", "source_file", "capture_date"]


def build_features(train, test):
    x = train.drop(columns=[c for c in DROP_COLS if c in train.columns]).copy()
    x2 = test.drop(columns=[c for c in DROP_COLS if c in test.columns]).copy()

    # Retain identity as categorical data while preserving the useful frequency
    # representation used by the 0.9140 model.
    for col, out_col in [
        ("Source IP", "source_ip_cat"),
        ("Destination IP", "destination_ip_cat"),
    ]:
        categories = sorted(x[col].dropna().unique())
        x[out_col] = pd.Categorical(x[col], categories=categories)
        x2[out_col] = pd.Categorical(x2[col], categories=categories)
        freq = x[col].value_counts(normalize=True)
        x[col] = x[col].map(freq).fillna(0.0).astype("float32")
        x2[col] = x2[col].map(freq).fillna(0.0).astype("float32")

    # Ports are identifiers, not ordered measurements. Keep the original
    # numeric feature and add a compact categorical copy.
    port_counts = x["Destination Port"].value_counts()
    common_ports = set(port_counts[port_counts >= 100].index)
    # Port 444 is the signature destination for almost every infiltration row.
    common_ports.add(444)
    port_categories = [-1] + sorted(common_ports)
    for frame in (x, x2):
        frame["destination_port_cat"] = pd.Categorical(
            frame["Destination Port"].where(
                frame["Destination Port"].isin(common_ports), -1
            ),
            categories=port_categories,
        )

    for col in CAT_COLS:
        categories = sorted(x[col].dropna().unique())
        x[col] = pd.Categorical(x[col], categories=categories)
        x2[col] = pd.Categorical(x2[col], categories=categories)

    numeric = x.select_dtypes(include=[np.number]).columns
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    x2[numeric] = x2[numeric].replace([np.inf, -np.inf], np.nan)
    return x, x2[x.columns]


def main():
    started = time.time()
    d1 = pd.read_pickle("data/D1.pkl")
    d2 = pd.read_pickle("data/D2.pkl")
    classes = sorted(d1["Label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(classes)}

    sampled = pd.concat(
        [
            group.sample(min(len(group), CAP), random_state=SEED)
            for _, group in d1.groupby("Label", sort=False)
        ],
        ignore_index=True,
    )
    y = sampled["Label"].map(label_to_id).to_numpy()
    x, x2 = build_features(sampled, d2)
    print(
        f"training {len(sampled)} rows, {x.shape[1]} features "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=127,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, callbacks=[log_evaluation(25)])
    proba = model.predict_proba(x2)
    np.save("final_proba.npy", proba)
    predictions = np.argmax(proba, axis=1)

    Path("answer_final.txt").write_text("\n".join(map(str, predictions)))
    with zipfile.ZipFile("answer_final.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write("answer_final.txt", arcname="answer.txt")
    print(
        f"wrote answer_final.zip ({len(predictions)} rows, "
        f"{time.time() - started:.0f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
