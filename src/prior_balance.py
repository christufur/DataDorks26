"""Prior-balance the proven 0.9140 predictions using ranked probabilities.

D2 contains unique flow_uid values. Its large-class distribution closely
matches D1 after taking one row per flow_uid. The baseline's count errors are
concentrated in a small family of mutually confused UNSW/web labels. This
script moves only the lowest-cost donor rows, ranked by the deduplicated
model's probabilities, toward the unique-flow prior.
"""

import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


# Expected D2 counts from D1's unique-flow class proportions.
EXPECTED = np.array([
    84.7, 69.5, 15470.9, 439.7, 5920.8, 710.0, 1488.2, 26354.7, 919.5,
    1091.0, 24791.8, 20734.7, 20427.7, 22964.8, 20038.4, 25845.5,
    20175.7, 20222.9, 3982.8, 495.5, 3928.6, 2478.9, 6.0, 1.5, 4047.8,
    2325.4, 302.2, 584.0, 21909.1, 24206.3, 19992.9, 28.5, 1.7, 4.2,
    29.5, 31.5,
])

# Only correct discrepancies far beyond normal sampling variation.
ADJUST = np.array([0, 1, 5, 18, 21, 25, 31, 32])


def read_predictions(path):
    with zipfile.ZipFile(path) as archive:
        return np.loadtxt(io.BytesIO(archive.read("answer.txt")), dtype=np.int16)


def write_submission(name, predictions):
    txt = Path(f"{name}.txt")
    txt.write_text("\n".join(map(str, predictions)))
    with zipfile.ZipFile(f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(txt, arcname="answer.txt")


def balance(baseline, probabilities, fraction):
    predictions = baseline.copy()
    counts = np.bincount(predictions, minlength=36)
    desired = counts.copy()
    desired[ADJUST] = np.rint(
        counts[ADJUST] + fraction * (EXPECTED[ADJUST] - counts[ADJUST])
    ).astype(int)

    donors = {int(c): int(counts[c] - desired[c]) for c in ADJUST if counts[c] > desired[c]}
    receivers = {
        int(c): int(desired[c] - counts[c]) for c in ADJUST if counts[c] < desired[c]
    }
    donor_rows = np.flatnonzero(np.isin(predictions, list(donors)))

    # Rank every feasible move by relative probability. Each row and each
    # donor/receiver capacity can be used only once.
    pairs = []
    eps = 1e-12
    for receiver in receivers:
        score = (
            np.log(probabilities[donor_rows, receiver] + eps)
            - np.log(probabilities[donor_rows, predictions[donor_rows]] + eps)
        )
        pairs.extend(zip(score.tolist(), donor_rows.tolist(), [receiver] * len(donor_rows)))
    pairs.sort(reverse=True)

    used = np.zeros(len(predictions), dtype=bool)
    transitions = Counter()
    for _, row, receiver in pairs:
        donor = int(predictions[row])
        if used[row] or donors.get(donor, 0) <= 0 or receivers[receiver] <= 0:
            continue
        predictions[row] = receiver
        used[row] = True
        donors[donor] -= 1
        receivers[receiver] -= 1
        transitions[(donor, receiver)] += 1
        if not any(receivers.values()):
            break

    print(f"fraction={fraction}: changed={used.sum()} transitions={dict(transitions)}")
    return predictions


def main():
    # inputs come from src/ensemble.py: the averaged answer and per-seed probas
    baseline = read_predictions("answer.zip")
    probabilities = np.mean(
        [np.load(f"proba_seed{seed}.npy") for seed in (42, 7)], axis=0
    )
    if probabilities.shape != (len(baseline), 36):
        raise ValueError(f"unexpected probability shape {probabilities.shape}")

    predictions = balance(baseline, probabilities, 1.0)
    write_submission("answer_prior_100", predictions)


if __name__ == "__main__":
    main()
