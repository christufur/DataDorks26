"""Train two full models and average their class probabilities."""

import subprocess
import sys

import numpy as np

from train import write_submission

ROUNDS = 139
SEEDS = (42, 7)


def main():
    probability_paths = []
    for seed in SEEDS:
        probability_path = f"proba_seed{seed}.npy"
        probability_paths.append(probability_path)
        subprocess.run(
            [
                sys.executable,
                "src/train.py",
                "--skip-validation",
                "--final-rounds",
                str(ROUNDS),
                "--seed",
                str(seed),
                "--output",
                f"answer_seed{seed}.txt",
                "--probabilities",
                probability_path,
            ],
            check=True,
        )

    probabilities = [
        np.load(path, mmap_mode="r") for path in probability_paths
    ]
    mean_probability = np.mean(probabilities, axis=0)
    predictions = np.argmax(mean_probability, axis=1).astype(int)
    archive = write_submission(predictions, "answer.txt")
    print(f"Wrote two-seed ensemble to {archive}")


if __name__ == "__main__":
    main()
