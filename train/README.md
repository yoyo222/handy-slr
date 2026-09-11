# Original training notebooks

The two Colab notebooks in this directory are the original training pipeline for
Handy's embedding model. They are the work of the project's co-author (see
[NOTICE](../NOTICE)), included here for reference because they document how
`server/model/weights.h5` was produced.

| Notebook | Contents |
|---|---|
| `01_prototypical_dtw_training.ipynb` | The full pipeline: episodic dataset, CNN+TCN encoder, `dtw_partition_loss`, training loop. This is what produced the shipped weights. |
| `02_continuous_decoding_experiments.ipynb` | A later iteration exploring continuous decoding and Levenshtein-based scoring. Exploratory; several cells are inert. |

They expect a Google Colab GPU runtime with the dataset mounted from Drive under
`/content/drive/MyDrive/SignData/`, so they will not run as-is elsewhere.

## Licensing

**These notebooks are excluded from the Apache License 2.0 grant that covers the
rest of this repository, and from the CC BY 4.0 grant covering `research/`.** They are
reproduced with permission for reference. No license to use, modify, or
redistribute them is granted here. Ask the author.

## Relationship to `experiments/`

The research code in [`experiments/`](../experiments/) supersedes these for
anything measured in the paper. `experiments/train_wlasl.py` is adapted from
notebook 02 and differs in ways that change results:

- class-disjoint train/validation splitting with novel classes excluded
- a corrected mirror augmentation. The original flipped every sample
  unconditionally without swapping left/right hand slots, producing
  anatomically impossible data and a systematic orientation mismatch with
  evaluation
- presence-aware preprocessing, so frames with no detected hand are not
  zero-filled into "ghost poses"
- a batched Soft-DTW loss, roughly 197× faster than the per-pair path
- L2 normalisation of embedding frames, which removes the degenerate solution
  the original loss admits

Use `experiments/` if you want to reproduce the paper. Use these notebooks if you
want to see where the shipped model came from.
