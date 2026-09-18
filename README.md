# CAHSI Data Analytics Challenge 2026 — DataDorks

Classify network flows into 36 traffic classes (benign + 35 attack types).
Official metric: Macro-F1.

## Software
- Python 3.13 (Anaconda)
- pandas, numpy, scikit-learn, lightgbm

## Install
```bash
pip install -r requirements.txt
```

## Run
1. Put the competition data at `data/D1.pkl` (labeled train) and
   `data/D2.pkl` (unlabeled test).
2. ```bash
   python src/train.py
   ```
   (`--quick` runs a fast subsampled smoke test.)

The script:
- maps the 36 labels to ids 0–35 by alphabetical order (matches the
  official label mapping),
- downsamples majority classes to 50k rows each,
- frequency-encodes IPs, treats Protocol/transport/source_file/capture_date
  as categoricals, replaces inf with NaN (LightGBM handles NaN natively),
- trains a LightGBM multiclass model with early stopping on holdout
  Macro-F1 (10% stratified holdout),
- writes `answer.txt` (one integer class id per D2 row, no header, no
  trailing newline) and zips it into `answer.zip` for Codabench upload.
