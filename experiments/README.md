# Experiments

Evaluation and training code for the research described in
[`research/`](../research/). This is separate from the live application: the app matches
against signs you record yourself, whereas everything here is measured on WLASL
under a protocol where the evaluation classes are never seen during training.

## Getting the data

WLASL is not redistributed here. You have to obtain it from its maintainers at
<https://dxli94.github.io/WLASL/> under their terms. The videos are hosted by
third parties and some links rot, which is why class counts vary between runs.

Once you have `WLASL_v0.3.json` and the raw videos, extract landmarks:

```bash
python experiments/preprocess_wlasl.py \
    --json data/wlasl/WLASL_v0.3.json \
    --raw_videos data/wlasl/raw_videos \
    --out experiments/wlasl_landmarks
```

This writes one `.npy` per video plus a presence sidecar recording whether
MediaPipe actually found a hand in each frame. `data/` and
`experiments/*landmarks*/` are gitignored.

## Evaluation protocol

Classes with at least 10 recordings (85 of them) are split class-disjoint into
45 base and 40 novel with `SPLIT_SEED = 42`. Results are averaged over 1000
episodes with `n_query = 5`.

Two things are worth knowing before comparing any numbers:

- **`v1` vs `v2`.** `v1` is the original preprocessing, which zero-fills frames
  where no hand was detected and so contaminates the data with "ghost poses" at
  the origin. `v2` is the corrected version and is the default. Only use `v1`
  numbers to show the size of that bug.
- **`novel40` vs `clean11`.** The original fine-tuning set overlapped the novel
  split: 29 of the 40 novel classes had been trained on. `clean11` is
  `novel40` minus the WLASL top-100, so it is the only honest measure of
  transfer for those checkpoints. The canonical splits are in
  [`novel_classes.json`](novel_classes.json).

Confidence intervals are roughly ±0.8 points for 1-shot and ±0.35–0.55 for
5-shot, so differences under one point are not meaningful.

## Running things

Few-shot evaluation:

```bash
python experiments/eval_harness.py \
    --checkpoint server/model/weights.h5 \
    --landmarks_dir experiments/wlasl_landmarks \
    --n_way 5 --k_shot 1 --n_episodes 1000 \
    --prototype_strategy per_recording
```

`--prototype_strategy` is `per_recording` (one prototype per recording, most
accurate), `medoid` (one representative recording per class), or `dba` (one
DTW barycenter per class, fastest).

Training. The clean recipe is the default: novel classes excluded, class-disjoint
validation, AdamW with weight decay 1e-4, early stopping:

```bash
python experiments/train_wlasl.py --loss soft --gamma 0.1 --normalize l2 --tag myrun
```

`--normalize l2` matters. Without it the α term in `dtw_partition_loss` is
trivially minimized by shrinking the whole embedding scale, and training
collapses, with loss decreasing monotonically while accuracy falls apart.
Constraining frames to the unit sphere removes that degenerate solution. Always
pass `--tag`; without it, runs overwrite each other's checkpoints.

`run_gamma_sweep.sh` runs the γ sensitivity sweep plus a hard-DTW control on the
same data, which is what separates "the loss helped" from "fixing the data
helped".

Continuous-stream benchmark:

```bash
python experiments/continuous_bench.py \
    --strategy dba --auto_threshold 0.5 --debounce 3 --trim_blank
```

`--auto_threshold Q` calibrates the rejection threshold from leave-one-out
distances over the registered recordings instead of using a fixed
`--threshold`. Use `q = 0.5`; higher quantiles optimize for accepting true
matches and flood the output with insertions.

Latency, and the tests for the Soft-DTW implementation:

```bash
python experiments/latency_bench.py --class_counts 5,10,20,40,80
python experiments/soft_dtw.py    # gradcheck, γ→0 equals hard DTW, batched == unbatched
```

## Results

Measured on classes disjoint from the training vocabulary.

| Change | Effect |
|---|---|
| Detection fallback for missing hands | 5-way 1-shot 44.1% to 52.7% |
| DBA prototype aggregation | 4.6x faster inference |
| Soft-DTW loss on L2-normalized embeddings | 10-way 5-shot transfer 67.1% to 76.1% |
| Conformal threshold calibration | Continuous-stream WER 280% to 83.1%, no manual tuning |

### Results files

| File | Contents |
|---|---|
| `results_baseline.json` | v1 / v2 / presence / original fine-tuned checkpoints |
| `results_prototype_v2.json` | Prototype strategies on v2 |
| `results_ft_transfer.json` | Fine-tuned checkpoints on novel40 and clean11 |
| `results_continuous.json` | Continuous benchmark configurations |
| `results_latency.json` | Latency by database size |
| `novel_classes.json` | Canonical novel40 / clean11 / leaked29 / top100 lists |
| `checkpoints/history_*.json` | Training histories |

Trained checkpoints are not published. `server/model/weights.h5` is the model the
application ships with; the fine-tuned research checkpoints are not included.

## Known limitations

These apply to every number in this directory:

- Continuous recognition is unsolved: exact sequence match is 0%.
- WLASL has speaker confounds; no signer-disjoint split has been run.
- `clean11` has only 11 classes, so 10-way episodes draw nearly the same set
  every time and the confidence intervals understate class-level uncertainty.
- Synthetic continuous streams do not reproduce coarticulation.
- Training was run at a single seed (42); no multi-seed averages.
