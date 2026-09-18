# CAHSI Data Analytics Challenge 2026 — Data Dorks

Classify network flows into 36 traffic classes (benign + 35 attack types).
Official metric: Macro-F1. **Best leaderboard score: 0.9328.**

## Software
- Python 3.13 (Anaconda), CPU only
- pandas, numpy, scikit-learn, lightgbm

## Setup
```bash
pip install -r requirements.txt
```
Place the competition data at `data/D1.pkl` (labeled train, 1,704,464 rows)
and `data/D2.pkl` (unlabeled test, 312,107 rows).

## Reproduce the best submission (0.9328)
Run the three stages from the repo root. Seeds are fixed, so with the
pinned library versions the output reproduces the scored file exactly.

```bash
# 1. Base labels (~25 min): deduplicated 2-seed LightGBM ensemble with
#    identity/temporal features -> answer.zip
PYTHONPATH=src python src/ensemble.py

# 2. Ranking model (~20 min): 2-seed LightGBM on frequency-encoded
#    features -> proba.npy (ranks the prior-balance moves)
python src/rank_model.py

# 3. Prior balancing (~1 min): largest single gain (+0.014 on the board)
#    -> answer_prior_100.zip == the winning 0.9328 submission
python src/prior_balance.py
```

The winning predictions are also committed directly as `answer.txt` /
`answer.zip` in the repo root (byte-identical to the scored submission).
`src/lookup_override.py` (optional stage) applies identity-rule fixes
measured at 99.93-100% precision; it scored equal at 4 decimal places.

## Method
1. **Base model** (`src/train.py`, driven twice by `src/ensemble.py`):
   drops repeated `flow_uid` rows, caps majority classes at 50k, keeps IP
   identity + frequency features, adds IP-pair, common-port, and time
   features as LightGBM categoricals, trains multiclass LightGBM
   (139 rounds, `class_weight=balanced`, lr 0.05, 127 leaves) on seeds
   42 and 7, and averages the predicted probabilities.
2. **Prior balancing** (`src/prior_balance.py`): D2's flow_uids are unique,
   so its class distribution follows D1's unique-flow proportions. The
   script moves the lowest-cost rows (ranked by log-probability ratio)
   among the mutually confused UNSW/web classes until predicted counts
   match that prior.
3. **Lookup overrides** (`src/lookup_override.py`): pure identity rules
   (source_file × capture_date × IP / socket) with support cutoffs whose
   precision measured 99.93–100% on a flow_uid-grouped holdout override
   the rows they map unambiguously.

## Submission format
`answer.txt`: one integer class id (0–35, labels in alphabetical order)
per D2 row, no header, no trailing newline, zipped for upload.

## Packaging for the code-submission form
```bash
sh make_package.sh First-Second-Third   # your team member names
```
Builds `submission_<names>.zip` containing `src/`, this README,
`requirements.txt`, and the winning `answer.txt`.
