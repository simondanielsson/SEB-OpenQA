#!/bin/bash

# run with `./evaluate_nq.sh <datetime_of_experiment>`
# might have to `chmod +x evaluate_nq.sh`

echo "Running 'squad/evaluate-v2.0.py' on run $1..."
python3 evaluation/scripts/squad/evaluate-v2.0.py \
  data/nq_full/nq_dev_squad_fmt_v4.json \
  evaluation/output/nq_full/"$1"/inferences.json \
  --out-file evaluation/results/nq_full/"$1".json
echo "Results saved to 'evaluation/results/nq_full/$1.json'."

