#!/usr/bin/env bash
# Sequential γ sensitivity sweep + hard-loss control on the clean v2 recipe.
# (γ=1.0 already run separately.) Each soft run ~20 min on the 3060; the
# hard control uses the numba per-pair path and takes ~1.5-2 h.
set -e
cd "$(dirname "$0")/.."

for g in 0.5 2.0 0.1; do
    echo "=== gamma $g ==="
    python experiments/train_wlasl.py --loss soft --gamma "$g" \
        > "experiments/train_ftv2_soft_g${g}.log" 2>&1
    tail -3 "experiments/train_ftv2_soft_g${g}.log"
done

echo "=== hard control (same clean data/recipe) ==="
python experiments/train_wlasl.py --loss hard \
    > experiments/train_ftv2_hard.log 2>&1
tail -3 experiments/train_ftv2_hard.log

echo "=== sweep complete ==="
