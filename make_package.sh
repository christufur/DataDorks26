#!/bin/sh
# Build the code-submission zip for the CAHSI Google Form.
# Usage: sh make_package.sh Name1-Name2-Name3
NAMES="${1:?usage: sh make_package.sh Name1-Name2-Name3}"
OUT="submission_${NAMES}.zip"
rm -f "$OUT"
zip -r "$OUT" src/train.py src/ensemble.py README.md requirements.txt answer.txt
echo "built $OUT"
