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
2. Final model (2-seed LightGBM ensemble, averaged probabilities):
   ```bash
   PYTHONPATH=src python src/ensemble.py
   ```
   Single-model variant: `python src/train.py` (`--quick` = fast smoke test).

The script:
- maps the 36 labels to ids 0–35 by alphabetical order (matches the
  official label mapping),
- removes repeated `flow_uid` rows before sampling,
- downsamples majority classes to 50k rows each,
- keeps IP identity and frequency, adds IP-pair, common-port, and time
  features, and treats discrete features as LightGBM categoricals,
- validates on an uncapped, natural-distribution 10% holdout before
  refitting on all selected rows,
- writes `answer.txt` (one integer class id per D2 row, no header, no
  trailing newline) and zips it into `answer.zip` for Codabench upload.
