"""Create high-confidence lookup overrides for the prior-balanced submission.

The organizer split has disjoint flow_uid values, so mappings are learned from
one representative row per flow. Rules and support cutoffs were selected on a
flow_uid-grouped holdout; their measured precision was 99.93% to 100%.

Run: /opt/anaconda3/bin/python src/lookup_override.py
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_ZIP = Path("answer_prior_100.zip")
RULES = [
    # name, key columns, minimum distinct training flows
    ("source_identity", ["source_file", "capture_date", "Source IP"], 10),
    ("destination_identity", ["source_file", "capture_date", "Destination IP"], 5),
    ("source_socket", ["source_file", "Source IP", "Destination Port"], 10),
    ("destination_socket", ["source_file", "Destination IP", "Destination Port"], 10),
]


def read_baseline():
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        raw = archive.read("answer.txt")
    return np.loadtxt(io.BytesIO(raw), dtype=np.int16)


def pure_mapping(train, columns, minimum_support):
    counts = (
        train.groupby(columns + ["Label"], observed=True, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(columns, observed=True, dropna=False)["count"].transform("sum")
    counts["total"] = totals
    best = counts.sort_values("count", ascending=False).drop_duplicates(columns)
    return best.loc[
        (best["count"] == best["total"]) & (best["total"] >= minimum_support),
        columns + ["Label"],
    ]


def write_submission(name, predictions):
    text_path = Path(f"{name}.txt")
    zip_path = Path(f"{name}.zip")
    text_path.write_text("\n".join(map(str, predictions)))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(text_path, arcname="answer.txt")
    print(f"wrote {zip_path}")


def main():
    d1 = pd.read_pickle("data/D1.pkl").drop_duplicates("flow_uid")
    d2 = pd.read_pickle("data/D2.pkl")
    classes = sorted(d1["Label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    baseline = read_baseline()
    if len(baseline) != len(d2):
        raise ValueError(f"baseline has {len(baseline)} rows; expected {len(d2)}")

    votes = np.full((len(d2), len(RULES)), -1, dtype=np.int16)
    for rule_idx, (name, columns, minimum_support) in enumerate(RULES):
        mapping = pure_mapping(d1, columns, minimum_support)
        mapped = (
            d2[columns]
            .merge(mapping, on=columns, how="left", sort=False)["Label"]
            .map(label_to_id)
        )
        votes[:, rule_idx] = mapped.fillna(-1).to_numpy(dtype=np.int16)
        available = votes[:, rule_idx] >= 0
        changed = available & (votes[:, rule_idx] != baseline)
        print(
            f"{name}: maps {available.sum()} rows; "
            f"would change {changed.sum()} baseline predictions"
        )

    # Safest candidate: at least two independent rules vote and every
    # available rule agrees.
    available_count = (votes >= 0).sum(axis=1)
    first_vote = np.max(votes, axis=1)
    unanimous = np.all((votes < 0) | (votes == first_vote[:, None]), axis=1)
    safe = (available_count >= 2) & unanimous
    conservative = baseline.copy()
    conservative[safe] = first_vote[safe]
    print(
        f"conservative: covers {safe.sum()}, "
        f"changes {(conservative != baseline).sum()}"
    )
    write_submission("answer_lookup_conservative", conservative)

    # Broader candidate: accept a unanimous mapping from any validated rule.
    any_unanimous = (available_count >= 1) & unanimous
    broad = baseline.copy()
    broad[any_unanimous] = first_vote[any_unanimous]
    print(
        f"broad: covers {any_unanimous.sum()}, "
        f"changes {(broad != baseline).sum()}"
    )
    write_submission("answer_lookup_broad", broad)


if __name__ == "__main__":
    main()
