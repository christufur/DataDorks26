"""Synthetic train.csv / test.csv for smoke-testing before real data drops.

Regression target (non-negative floats), mixed feature types, missing values.
Run: python make_fake_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
N_TRAIN, N_TEST, SEED = 500, 200, 42


def make_features(n, rng):
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": np.round(rng.normal(55_000, 18_000, n), 2),
        "score_a": np.round(rng.uniform(0, 100, n), 1),
        "score_b": np.round(rng.normal(50, 15, n), 1),
        "num_visits": rng.poisson(4, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1]),
        "plan": rng.choice(["free", "basic", "pro", "enterprise"], n),
        "city": rng.choice([f"city_{i:02d}" for i in range(30)], n),  # high-cardinality
    })


def inject_missing(df, rng, frac=0.05):
    df = df.copy()
    for col in df.columns:
        df.loc[rng.random(len(df)) < frac, col] = np.nan
    return df


def make_target(df, rng):
    signal = (
        0.03 * df["age"].fillna(df["age"].mean())
        + 0.00002 * df["income"].fillna(df["income"].mean())
        + 0.02 * df["score_a"].fillna(df["score_a"].mean())
        + df["device"].map({"mobile": 0.5, "desktop": -0.5, "tablet": 0.0}).fillna(0)
        + rng.normal(0, 0.5, len(df))
    )
    return pd.Series(np.round(np.clip(signal * 10 + 100, 0, None), 2), name="target")


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    train = inject_missing(make_features(N_TRAIN, rng), rng)
    train["target"] = make_target(train, rng)
    test = inject_missing(make_features(N_TEST, rng), rng)

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Wrote train {train.shape}, test {test.shape}")


if __name__ == "__main__":
    main()
